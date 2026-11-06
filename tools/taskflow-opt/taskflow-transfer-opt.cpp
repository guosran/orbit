#include "Taskflow/DeviceTransferAudit.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace orbit::taskflow;
namespace cl=llvm::cl;
static cl::opt<std::string> input(cl::Positional,cl::Required);
static cl::opt<std::string> target("target-config",cl::Required),finalAIE("final-aie",cl::Required),
    artifact("artifact",cl::Required),metadata("runtime-binding",cl::Required),
    modelFile("cost-model"),output("output",cl::Required);
static cl::opt<bool> submitPriced("submit-already-priced"),waitPriced("wait-already-priced");
int main(int argc,char **argv) {
  cl::ParseCommandLineOptions(argc,argv,"ORBIT public communication to checked device interval cost\n");
  auto fail=[](const std::string &e){llvm::errs()<<e<<"\n";return 1;};std::string error;
  auto read=[&](const std::string &path,llvm::json::Value &v) {
    auto b=llvm::MemoryBuffer::getFile(path);if(!b){error=b.getError().message();return false;}
    auto p=llvm::json::parse((*b)->getBuffer());if(!p){error=llvm::toString(p.takeError());return false;}v=std::move(*p);return true;
  };
  auto digest=[](llvm::StringRef bytes){llvm::SHA256 h;h.update(bytes);return llvm::toHex(h.final(),true);};
  llvm::json::Value config=nullptr,meta=nullptr;if(!read(target,config) || !read(metadata,meta))return fail(error);
  auto *m=meta.getAsObject();if(!m)return fail("runtime binding must be object");
  DeviceTransferBackendBinding binding;
  const std::array<std::pair<const char *,std::string *>,5> identityFields={{
      {"runtime_identity",&binding.runtimeIdentity},
      {"toolchain_identity",&binding.toolchainIdentity},{"path_identity",&binding.pathIdentity},
      {"observer_identity",&binding.observerIdentity},{"condition_identity",&binding.conditionIdentity}}};
  for(const auto &field:identityFields) {
    auto value=m->getString(field.first);if(!value || value->empty())return fail("missing explicit runtime/path/observer/condition binding");*field.second=value->str();
  }
  auto elf=llvm::MemoryBuffer::getFile(artifact);if(!elf)return fail(elf.getError().message());
  binding.artifactSHA256=digest((*elf)->getBuffer());
  auto aieBytes=llvm::MemoryBuffer::getFile(finalAIE);if(!aieBytes)return fail(aieBytes.getError().message());
  mlir::DialectRegistry registry;registry.insert<mlir::taskflow::TaskflowDialect,
      mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::math::MathDialect,
      mlir::memref::MemRefDialect,mlir::func::FuncDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);context.allowUnregisteredDialects();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(input,&context);
  auto physical=mlir::parseSourceString<mlir::ModuleOp>((*aieBytes)->getBuffer(),&context);
  if(!module || !physical)return fail("valid source Taskflow and generic final AIE required");
  GenericAnalysis analysis;GenericMappingPlan plan;
  if(!analyzeGenericProgram(*module,analysis,error) || !planGenericMapping(analysis,config,plan,error))return fail(error);
  binding.graphId=programGraphIdentity(plan.communication);binding.nativePlanId=plan.planId;
  DeviceTransferBackendBinding derived;llvm::json::Value witness=nullptr;
  if(!deriveDeviceTransferBackendBinding(analysis,plan,*physical,binding,derived,witness,error))return fail(error);
  DeviceTransferFacts facts;
  if(!collectDeviceTransferFacts(analysis,plan,derived,facts,error) || !checkDeviceTransferFacts(analysis,plan,facts,error))return fail(error);
  DeviceTransferCostEvaluation evaluated;
  std::string effectiveModel=modelFile;
  if(effectiveModel.empty())if(auto object=config.getAsObject())
    if(auto path=object->getString("device_transfer_model"))effectiveModel=path->str();
  if(!effectiveModel.empty()) {
    auto data=llvm::MemoryBuffer::getFile(effectiveModel);if(!data)return fail(data.getError().message());
    DeviceTransferCostModel model;if(!parseDeviceTransferCostModel((*data)->getBuffer(),model,error))return fail(error);
    DeviceTransferComposition composition;composition.submitAlreadyPriced=submitPriced;composition.completionAlreadyPriced=waitPriced;
    if(!evaluateDeviceTransferCost(facts,model,composition,evaluated,error))return fail(error);
  } else {evaluated.reason="no calibrated device interval parameters supplied";evaluated.reservationResources=facts.reservationResources;}
  if(llvm::sys::fs::exists(output))return fail("output must be new");
  if(auto ec=llvm::sys::fs::create_directories(output))return fail(ec.message());
  auto write=[&](llvm::StringRef name,const llvm::json::Value &value) {
    std::error_code ec;llvm::raw_fd_ostream stream(output+"/"+name.str(),ec);if(ec){error=ec.message();return false;}stream<<llvm::formatv("{0:2}\n",value);return true;
  };
  llvm::json::Value receipt=llvm::json::Object{{"schema","orbit.device-transfer.receipt.v15"},
    {"final_aie_file_sha256",digest((*aieBytes)->getBuffer())},{"artifact_sha256",binding.artifactSHA256},
    {"graph_id",binding.graphId},{"native_plan_id",binding.nativePlanId},
    {"source_body_fingerprint",analysis.bodyFingerprint},{"physical_bytes",nullptr},
    {"completion_evidence","structural successful output-token await; runtime success must be bound separately"},
    {"cost_boundary","submit-to-successful-wait-return; includes necessary local copy, locks, runtime/configuration and polling"}};
  if(!write("facts.json",serializeDeviceTransferFacts(facts)) || !write("binding.json",serializeDeviceTransferBackendBinding(derived)) ||
     !write("witness.json",witness) || !write("cost.json",serializeDeviceTransferCost(evaluated)) ||
     !write("receipt.json",receipt) || !write("program-graph.json",serializeProgramGraph(plan.communication)))return fail(error);
  return 0;
}

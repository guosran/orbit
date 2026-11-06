#include "Taskflow/DeviceTransferCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
using namespace orbit::taskflow;
namespace {
std::string text(const llvm::json::Value &v){std::string s;llvm::raw_string_ostream o(s);o<<v;return o.str();}
const char *body=R"mlir(module {
 func.func @copy(%A: memref<1x2048xf32> {orbit.role="input"}, %C: memref<1x2048xf32> {orbit.role="output"}) attributes {orbit.noalias} {
  %done=taskflow.task @copy will_reads(%A:memref<1x2048xf32>) will_writes(%C:memref<1x2048xf32>) [original_read_memrefs(%A:memref<1x2048xf32>),original_write_memrefs(%C:memref<1x2048xf32>)] :(memref<1x2048xf32>,memref<1x2048xf32>)->memref<1x2048xf32> {
  ^bb0(%a:memref<1x2048xf32>,%c:memref<1x2048xf32>):
   affine.for %i=0 to 1 { affine.for %j=0 to 2048 {
    %v=affine.load %a[%i,%j]:memref<1x2048xf32>
    affine.store %v,%c[%i,%j]:memref<1x2048xf32>
   }}
   taskflow.yield done_writes(%c:memref<1x2048xf32>)
  }
  return
 }
})mlir";
}
int main(){
 mlir::DialectRegistry registry;registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
  mlir::arith::ArithDialect,mlir::memref::MemRefDialect,mlir::func::FuncDialect>();
 mlir::MLIRContext context(registry);std::string error;unsigned checks=0;bool ok=true;
 auto require=[&](bool value,const char *label){if(value)++checks;else {ok=false;llvm::errs()<<label<<": "<<error<<"\n";}};
 auto module=mlir::parseSourceString<mlir::ModuleOp>(body,&context);GenericAnalysis a;GenericMappingPlan p;
 llvm::json::Value config=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{1,256}}}}};
 if(!module || !analyzeGenericProgram(*module,a,error) || !planGenericMapping(a,config,p,error)){llvm::errs()<<error;return 1;}
 DeviceTransferBackendBinding b;b.graphId=programGraphIdentity(p.communication);b.nativePlanId=p.planId;
 b.artifactSHA256=std::string(64,'a');b.runtimeIdentity="synthetic-runtime";b.toolchainIdentity="synthetic-toolchain";
 b.pathIdentity="synthetic-final-AIE-path";b.observerIdentity="synthetic-start-wait-observer";b.conditionIdentity="synthetic-prepared-condition";
 for(const std::string direction:{"host_to_l1","l1_to_host"}){
  DeviceTransferStreamBinding s;s.direction=direction;
  for(const auto &t:p.transfers)if(t.direction==direction){s.storage=t.storage;s.publicEvents.push_back(t.event);}
  s.runtimeDescriptors.push_back({direction+".shim",0,2048,1,4});
  s.coreDescriptors.push_back({direction+".core",0,256,8,4});b.streams.push_back(s);
 }
 DeviceTransferFacts facts;
 require(collectDeviceTransferFacts(a,p,b,facts,error),"actual Taskflow copy plan plus synthetic descriptor witness");
 require(facts.publicTransferCount==16 && facts.runtimeDescriptorCount==2 && facts.coreTransactions==16,
         "public transfer count, coalesced runtime descriptors and core transactions are distinct");
 require(facts.requestedBytes==16384 && facts.descriptorBytes==16384 && facts.segmentCount==8 && facts.tileElements==256,
         "graph and both descriptor levels agree on payload");
 require(checkDeviceTransferFacts(a,p,facts,error),"independent facts reconstruction");
 require(facts.coveredComputeEvents.size()==8 && facts.reservationResources==std::vector<std::string>{
     p.memoryBinding.dmaResource,p.memoryBinding.computeResource,p.memoryBinding.submissionResource},
     "interval owns actual copy computations and reserves existing resources");
 auto reject=[&](DeviceTransferBackendBinding x,const char *label){DeviceTransferFacts ignored;require(!collectDeviceTransferFacts(a,p,x,ignored,error),label);};
 auto changed=b;changed.graphId="stale";reject(changed,"stale public graph rejected");
 changed=b;changed.nativePlanId="stale";reject(changed,"stale native plan rejected");
 changed=b;changed.artifactSHA256="wrong";reject(changed,"artifact digest required");
 changed=b;changed.conditionIdentity.clear();reject(changed,"runtime execution conditions required");
 changed=b;changed.completionBoundary="pure-DMA";reject(changed,"unobserved pure DMA timer not invented");
 changed=b;changed.streams.pop_back();reject(changed,"missing output stream rejected");
 changed=b;changed.streams[0].publicEvents.pop_back();reject(changed,"omitted public transfer rejected");
 changed=b;changed.streams[0].publicEvents.push_back(changed.streams[0].publicEvents[0]);reject(changed,"duplicate public transfer rejected");
 changed=b;changed.streams[0].storage=999;reject(changed,"wrong descriptor storage rejected");
 changed=b;changed.streams[0].stage=1;reject(changed,"wrong hardware stage rejected");
 changed=b;changed.streams[0].runtimeDescriptors[0].elements=1024;reject(changed,"coalesced runtime bytes mismatch rejected");
 changed=b;changed.streams[0].runtimeDescriptors[0].elements=INT64_MAX;reject(changed,"descriptor byte overflow rejected");
 changed=b;changed.streams[0].runtimeDescriptors[0].offsetBytes=4;reject(changed,"unaligned runtime offset rejected");
 changed=b;changed.streams[0].runtimeDescriptors[0].elementBytes=2;reject(changed,"descriptor dtype width rejected");
 changed=b;changed.streams[0].coreDescriptors[0].elements=512;reject(changed,"core tile length mismatch rejected");
 changed=b;changed.streams[0].coreDescriptors[0].repetitions=7;reject(changed,"core loop count mismatch rejected");
 changed=b;changed.streams[1].coreDescriptors[0].id=changed.streams[0].coreDescriptors[0].id;reject(changed,"duplicate descriptor identity rejected");
 auto tamper=facts;tamper.transfers[0].versions={999};require(!checkDeviceTransferFacts(a,p,tamper,error),"mutated version facts rejected");
 tamper=facts;tamper.transfers[0].resource="fake-DMA";require(!checkDeviceTransferFacts(a,p,tamper,error),"fabricated resource rejected");
 tamper=facts;tamper.runtimeDescriptorCount=16;require(!checkDeviceTransferFacts(a,p,tamper,error),"descriptor count cannot be replaced by phase count");
 // Source proof is structural and checks the actual stored SSA value.
 auto modified=mlir::parseSourceString<mlir::ModuleOp>(body,&context);
 mlir::affine::AffineStoreOp store;modified->walk([&](mlir::affine::AffineStoreOp x){store=x;});
 mlir::OpBuilder builder(store);auto value=store.getValue();
 auto sum=builder.create<mlir::arith::AddFOp>(store.getLoc(),value,value);store->setOperand(0,sum.getResult());
 GenericAnalysis other;GenericMappingPlan otherPlan;
 require(analyzeGenericProgram(*modified,other,error) && planGenericMapping(other,config,otherPlan,error),"arithmetic variant remains a real legal generic plan");
 auto otherBinding=b;otherBinding.graphId=programGraphIdentity(otherPlan.communication);otherBinding.nativePlanId=otherPlan.planId;
 DeviceTransferFacts ignored;require(!collectDeviceTransferFacts(other,otherPlan,otherBinding,ignored,error),"arbitrary compute body cannot inherit copy composite");
 llvm::json::Object domain{{"runtime_identity",b.runtimeIdentity},{"toolchain_identity",b.toolchainIdentity},
  {"path_identity",b.pathIdentity},{"observer_identity",b.observerIdentity},{"condition_identity",b.conditionIdentity},
  {"architecture_identity",facts.architectureIdentity},{"implementation","identity-copy-f32"},
  {"completion_boundary",b.completionBoundary},{"min_bytes",1024},{"max_bytes",32768},
  {"min_segments",1},{"max_segments",16},{"min_tile_elements",16},{"max_tile_elements",4096}};
 auto domainText=text(llvm::json::Value(std::move(domain)));
 auto makeModel=[&](bool calibrated){
  auto copyDomain=llvm::json::parse(domainText);if(!copyDomain){llvm::consumeError(copyDomain.takeError());return std::string();}
  llvm::json::Object row{{"domain",std::move(*copyDomain)}};
  if(calibrated){row["entry_ns"]=1;row["segment_ns"]=2;row["byte_ns"]=0.5;
   row["provenance"]=llvm::json::Object{{"kind","synthetic-unit"},{"source","not device calibration"}};
   row["raw_manifest_sha256"]=std::string(64,'b');}
  return text(llvm::json::Object{{"schema","orbit.device-transfer.model.v15"},{"model_id","synthetic-unit-only"},
    {"units","ns"},{"rows",llvm::json::Array{std::move(row)}}});
 };
 DeviceTransferCostModel model;DeviceTransferCostEvaluation cost;DeviceTransferComposition composition;
 require(parseDeviceTransferCostModel(makeModel(false),model,error) && evaluateDeviceTransferCost(facts,model,composition,cost,error) &&
         !cost.durationNs && cost.status=="unknown","unfitted coefficients remain null");
 require(parseDeviceTransferCostModel(makeModel(true),model,error) && evaluateDeviceTransferCost(facts,model,composition,cost,error) &&
         cost.durationNs && *cost.durationNs==8209,"guarded synthetic composite formula, not a physical DMA model");
 auto parameterIdentity=model.parameterFileSHA256;
 require(parseDeviceTransferCostModel(makeModel(true)+"\n",model,error) && model.parameterFileSHA256!=parameterIdentity,
         "exact parameter bytes bound separately from declared label");
 composition.submitAlreadyPriced=true;require(!evaluateDeviceTransferCost(facts,model,composition,cost,error),"submit cannot be charged twice");
 composition={};composition.completionAlreadyPriced=true;require(!evaluateDeviceTransferCost(facts,model,composition,cost,error),"completion/wait cannot be charged twice");
 composition={};composition.separatelyPricedEvents={facts.transfers[0].event};
 require(!evaluateDeviceTransferCost(facts,model,composition,cost,error),"native DMA cannot be charged beside owning interval");
 composition.separatelyPricedEvents={facts.coveredComputeEvents[0]};
 require(!evaluateDeviceTransferCost(facts,model,composition,cost,error),"local copy compute cannot be charged beside owning interval");
 composition={};auto old=facts;old.binding.runtimeIdentity="different";
 require(evaluateDeviceTransferCost(old,model,composition,cost,error) && !cost.durationNs && cost.status=="unknown","runtime mismatch is unknown not zero");
 old=facts;old.segmentCount=17;
 require(evaluateDeviceTransferCost(old,model,composition,cost,error) && !cost.durationNs,"unseen segmentation outside domain remains unknown");
 auto invalid=model;invalid.rows[0].entryNs=-1;
 require(!evaluateDeviceTransferCost(facts,invalid,composition,cost,error),"negative coefficients rejected");
 invalid=model;invalid.rows.push_back(invalid.rows[0]);
 require(!evaluateDeviceTransferCost(facts,invalid,composition,cost,error),"overlapping model rows rejected");
 invalid=model;invalid.rows[0].segmentNs.reset();
 require(!evaluateDeviceTransferCost(facts,invalid,composition,cost,error),"partial coefficient sets rejected");
 invalid=model;invalid.rows[0].rawManifestSHA256="bad";
 require(!evaluateDeviceTransferCost(facts,invalid,composition,cost,error),"raw evidence digest required for calibration");
 const auto report=serializeDeviceTransferFacts(facts);const auto *o=report.getAsObject();
 require(o && o->get("physical_ddr_bytes")->getAsNull().has_value(),"descriptor payload is not claimed physical DDR traffic");
 llvm::outs()<<"{\"passed\":"<<(ok?"true":"false")<<",\"checks\":"<<checks<<"}\n";return ok?0:1;
}

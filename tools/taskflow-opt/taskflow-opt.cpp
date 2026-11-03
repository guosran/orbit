#include "Taskflow/StructuredProgram.h"
#include "Taskflow/Search.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace orbit::taskflow;
namespace cl = llvm::cl;
static cl::opt<std::string>
    input(cl::Positional,
          cl::desc("source plan.json or authenticated Taskflow .mlir"),
          cl::Required);
static cl::opt<std::string>
    planFile("emit-plan", cl::desc("Write versioned execution plan"));
static cl::opt<std::string>
    mlirFile("emit-mlir", cl::desc("Write materialized canonical Taskflow"));
static cl::opt<int64_t> tm("tile-m", cl::init(0)), tn("tile-n", cl::init(0)),
    tk("tile-k", cl::init(0));
static cl::opt<bool> fuse("fuse", cl::init(false)),
    fission("fission", cl::init(false));
static cl::opt<bool>
    verifyOnly("verify-only", cl::init(false),
               cl::desc("Verify MLIR, including fail-closed legacy verifier, "
                        "without exchange export"));
static cl::opt<bool> search("search", cl::init(false)), searchResume("resume",cl::init(false));
static cl::opt<std::string> backend("backend",cl::init("npu")), evaluator("evaluator"), targetConfig("target-config"), searchOutput("search-output"), searchCheckpoint("checkpoint");
static cl::opt<int64_t> searchRounds("max-rounds",cl::init(2)), searchBudget("max-candidates",cl::init(9)), searchWidth("beam-width",cl::init(1));
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "ORBIT independent structured Taskflow core\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry);
  auto bytes = llvm::MemoryBuffer::getFile(input);
  if (!bytes) {
    llvm::errs() << bytes.getError().message() << "\n";
    return 1;
  }
  Program p;
  std::string error;
  llvm::StringRef text = (*bytes)->getBuffer();
  auto fatal = [&](llvm::StringRef e) {
    llvm::errs() << "taskflow-opt: " << e << "\n";
    return 1;
  };
  auto parse = [&](llvm::StringRef s) {
    return mlir::parseSourceString<mlir::ModuleOp>(s, &context);
  };
  if (text.ltrim().starts_with("{")) {
    auto json = llvm::json::parse(text);
    if (!json)
      return fatal(llvm::toString(json.takeError()));
    if (!parsePlan(*json, p, error))
      return fatal(error);
  } else {
    auto module = parse(text);
    if (!module || mlir::failed(mlir::verify(*module)))
      return fatal("MLIR verification failed");
    auto witness =
        module->getOperation()->getAttrOfType<mlir::StringAttr>("orbit.plan");
    if (!witness) {
      if (verifyOnly)
        return 0;
      return fatal("unsupported MLIR: missing orbit.plan structured witness");
    }
    auto json = llvm::json::parse(witness.getValue());
    if (!json)
      return fatal(llvm::toString(json.takeError()));
    if (!parsePlan(*json, p, error))
      return fatal(error);
    auto expected = parse(emitTaskflow(p));
    if (!expected)
      return fatal("internal witness materialization failure");
    module->getOperation()->removeAttr("orbit.plan");
    if (!mlir::OperationEquivalence::isEquivalentTo(
            module->getOperation(), expected->getOperation(),
            mlir::OperationEquivalence::Flags::IgnoreLocations))
      return fatal("Taskflow body, slices, storage or dependencies disagree "
                   "with the exchange witness");
    if (verifyOnly)
      return 0;
  }
  if (search) {
    SearchOptions options{backend,evaluator,searchOutput,targetConfig,searchCheckpoint,searchRounds,searchBudget,searchWidth,searchResume};
    return runStructuredSearch(p,options,error) ? 0 : fatal(error);
  }
  if (!transform(p, {tm, tn, tk, fuse, fission}, error))
    return fatal(error);
  auto module = parse(emitTaskflow(p));
  if (!module || mlir::failed(mlir::verify(*module)))
    return fatal("materialized Taskflow verification failed");
  std::string json;
  llvm::raw_string_ostream js(json);
  js << llvm::formatv("{0:2}", printPlan(p));
  module->getOperation()->setAttr("orbit.plan",
                                  mlir::StringAttr::get(&context, json));
  auto output = [&](const std::string &path, auto printer) -> bool {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      error = ec.message();
      return false;
    }
    printer(os);
    return true;
  };
  if (!planFile.empty() &&
      !output(planFile, [&](auto &os) { os << json << "\n"; }))
    return fatal(error);
  if (!mlirFile.empty() && !output(mlirFile, [&](auto &os) {
        module->print(os);
        os << "\n";
      }))
    return fatal(error);
  if (planFile.empty() && mlirFile.empty()) {
    module->print(llvm::outs());
    llvm::outs() << "\n";
  }
  return 0;
}

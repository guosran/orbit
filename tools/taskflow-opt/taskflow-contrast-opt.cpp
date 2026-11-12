#include "Taskflow/GenericProgramContrast.h"
#include "Taskflow/GenericProgramTiming.h"
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
using namespace orbit::taskflow;
namespace cl = llvm::cl;
static cl::opt<std::string> baseline(cl::Positional, cl::Required),
    candidate("candidate", cl::Required), baselineConfig("baseline-config"),
    candidateConfig("candidate-config"), modelPath("cost-model"),
    targetPath("target-binding"), output("output", cl::Required);
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "ORBIT paired checked data-organization implementation contrast\n");
  std::string error;
  auto fail = [](const std::string &s) {
    llvm::errs() << s << "\n";
    return 1;
  };
  auto read = [&](const std::string &path, llvm::json::Value &v) {
    auto b = llvm::MemoryBuffer::getFile(path);
    if (!b) {
      error = b.getError().message();
      return false;
    }
    auto p = llvm::json::parse((*b)->getBuffer());
    if (!p) {
      error = llvm::toString(p.takeError());
      return false;
    }
    v = std::move(*p);
    return true;
  };
  llvm::json::Value lc = llvm::json::Object{}, rc = llvm::json::Object{};
  if ((!baselineConfig.empty() && !read(baselineConfig, lc)) ||
      (!candidateConfig.empty() && !read(candidateConfig, rc)))
    return fail(error);
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto lm = mlir::parseSourceFile<mlir::ModuleOp>(baseline, &context);
  auto rm = mlir::parseSourceFile<mlir::ModuleOp>(candidate, &context);
  if (!lm || !rm)
    return fail("valid actual Taskflow modules required");
  GenericAnalysis la, ra;
  GenericMappingPlan lp, rp;
  if (!analyzeGenericProgram(*lm, la, error) ||
      !analyzeGenericProgram(*rm, ra, error) ||
      !planGenericMapping(la, lc, lp, error) ||
      !planGenericMapping(ra, rc, rp, error))
    return fail(error);
  GenericProgramContrastFacts facts;
  if (!collectGenericProgramContrastFacts(la, lp, ra, rp, facts, error))
    return fail(error);
  GenericProgramContrastCost cost;
  if (!modelPath.empty()) {
    llvm::json::Value model = nullptr, target = nullptr;
    if (targetPath.empty())
      return fail("explicit actual target binding required with model");
    if (!read(modelPath, model) || !read(targetPath, target) ||
        !evaluateGenericProgramContrast(facts, model, target, cost, error))
      return fail(error);
  } else
    cost.reason = "No frozen implementation-change parameters supplied";
  llvm::json::Value lt = nullptr, rt = nullptr;
  if (!evaluateGenericProgramTiming(la, lp, lt, error) ||
      !evaluateGenericProgramTiming(ra, rp, rt, error))
    return fail(error);
  if (llvm::sys::fs::exists(output))
    return fail("output must be new");
  if (auto ec = llvm::sys::fs::create_directories(output))
    return fail(ec.message());
  auto write = [&](llvm::StringRef file, const llvm::json::Value &v) {
    std::error_code ec;
    llvm::raw_fd_ostream s(output + "/" + file.str(), ec);
    if (ec) {
      error = ec.message();
      return false;
    }
    s << llvm::formatv("{0:2}\n", v);
    return true;
  };
  if (!write("facts.json", serializeGenericProgramContrastFacts(facts)) ||
      !write("cost.json", serializeGenericProgramContrastCost(cost)) ||
      !write("baseline-plan.json", serializeGenericMapping(la, lp)) ||
      !write("candidate-plan.json", serializeGenericMapping(ra, rp)) ||
      !write("baseline-timing.json", lt) || !write("candidate-timing.json", rt))
    return fail(error);
  return 0;
}

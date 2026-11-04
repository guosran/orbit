#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/GenericMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <limits>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis analysis;
  GenericMappingPlan base;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error) ||
      !planGenericMapping(analysis, llvm::json::Object{}, base, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }
  if (analysis.tasks.size() < 2 || base.transfers.size() < 2) {
    llvm::errs() << "requires multi-task fixture\n";
    return 1;
  }
  unsigned checks = 0;
  auto reject = [&](const char *name,
                    std::function<void(GenericMappingPlan &)> mutate) {
    GenericMappingPlan p = base;
    mutate(p);
    error.clear();
    // Deliberately discard identity: semantic failures must not rely on hashes.
    p.planId.clear();
    if (checkGenericMapping(analysis, p, error)) {
      llvm::errs() << "accepted illegal plan: " << name << "\n";
      return false;
    }
    llvm::outs() << name << ": " << error << "\n";
    ++checks;
    return true;
  };
  bool passed = true;
  passed &= reject("missing input transfer",
                   [](auto &p) { p.transfers.erase(p.transfers.begin()); });
  passed &= reject("missing output transfer",
                   [](auto &p) { p.transfers.pop_back(); });
  passed &= reject("duplicate transfer",
                   [](auto &p) { p.transfers.push_back(p.transfers.front()); });
  passed &= reject("wrong transfer extent",
                   [](auto &p) { p.transfers[0].bytes += 4; });
  passed &= reject("wrong transfer role",
                   [](auto &p) { p.transfers[0].direction = "l1_to_host"; });
  passed &= reject("invalid storage index", [&](auto &p) {
    p.transfers[0].storage = analysis.storages.size();
  });
  passed &= reject("unexpected temporary transfer", [&](auto &p) {
    for (size_t i = 0; i < analysis.storages.size(); ++i)
      if (analysis.storages[i].role == "temporary") {
        p.transfers.push_back({i, "host_to_l1", 64});
        return;
      }
    p.transfers[0].direction = "bad";
  });
  passed &= reject("wrong body identity", [](auto &p) {
    p.bodyFingerprint = "different actual body";
  });
  passed &= reject("unsupported strategy",
                   [](auto &p) { p.strategy = "parallel_magic"; });
  passed &=
      reject("underreported residency", [](auto &p) { --p.residentBytes; });
  passed &= reject("insufficient capacity",
                   [](auto &p) { p.capacityBytes = p.residentBytes - 1; });
  passed &=
      reject("unvalidated capacity", [](auto &p) { p.capacityBytes = 24577; });
  passed &= reject("missing task", [](auto &p) { p.orderedTasks.pop_back(); });
  passed &= reject("duplicate task", [](auto &p) {
    p.orderedTasks.push_back(p.orderedTasks.front());
  });
  passed &= reject("invalid task index",
                   [&](auto &p) { p.orderedTasks[0] = analysis.tasks.size(); });
  passed &= reject("dependency order", [](auto &p) {
    std::reverse(p.orderedTasks.begin(), p.orderedTasks.end());
  });
  passed &= reject("measured cost masquerade", [](auto &p) {
    p.hasCost = true;
    p.estimatedMs = 1;
    p.costProvenance = "measured chain model";
  });
  passed &= reject("nonfinite cost", [](auto &p) {
    p.hasCost = true;
    p.estimatedMs = std::numeric_limits<double>::infinity();
    p.costProvenance = "theoretical";
  });
  passed &= reject("unknown plan fabricated score",
                   [](auto &p) { p.estimatedMs = 1; });
  passed &= reject("unknown plan forged provenance",
                   [](auto &p) { p.costProvenance = "theoretical"; });
  GenericMappingPlan scored;
  llvm::json::Value cfg = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{
           {"theoretical",
            llvm::json::Object{{"launch_ms", 0.1},
                               {"per_scalar_operation_ms", 0.000001},
                               {"per_transfer_byte_ms", 0.0000001}}}}}};
  if (!planGenericMapping(analysis, cfg, scored, error) ||
      !checkGenericMapping(analysis, scored, error)) {
    llvm::errs() << "positive theoretical cost failed: " << error << "\n";
    passed = false;
  } else {
    ++checks;
    auto unknown = base;
    base = scored;
    passed &=
        reject("forged theoretical score", [](auto &p) { p.estimatedMs += 1; });
    passed &= reject("changed unaccounted coefficient",
                     [](auto &p) { p.theoreticalLaunchMs += 1; });
    passed &= reject("negative model coefficient",
                     [](auto &p) { p.theoreticalPerOperationMs = -1; });
    base = unknown;
  }
  if (!checkGenericMapping(analysis, base, error)) {
    llvm::errs() << "positive plan failed: " << error << "\n";
    passed = false;
  } else
    ++checks;
  llvm::outs() << "{\"passed\":" << (passed ? "true" : "false")
               << ",\"checks\":" << checks << ",\"hardware_executed\":false}\n";
  return passed ? 0 : 1;
}

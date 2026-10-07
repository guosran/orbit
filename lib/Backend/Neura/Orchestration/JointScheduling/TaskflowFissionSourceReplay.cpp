#include "TaskflowFissionSourceReplay.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Transforms/Optimizations/TaskflowFission.h"
#include "Backend/Neura/Orchestration/JointScheduling/GraphFactsIO.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <memory>
#include <optional>

using namespace mlir;
namespace mlir::amoeba::neura::joint_scheduling {

llvm::StringRef taskflowFissionCanonicalLoweringPipeline() {
  // Exact production input0 normalization suffix. It is source-owned; the
  // caller cannot supply a pipeline that drops predicates, effects or work.
  return "builtin.module(convert-taskflow-to-neura,cse,lower-affine,"
         "convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=0},"
         "assign-accelerator,lower-memref-to-neura,lower-arith-to-neura,"
         "lower-builtin-to-neura,lower-llvm-to-neura,promote-input-arg-to-const,"
         "fold-constant,canonicalize-return,canonicalize-live-in,"
         "leverage-predicated-value,transform-ctrl-to-data-flow,fold-constant,"
         "insert-data-mov,bind-source-iteration-domain)";
}

static llvm::StringRef taskflowFissionCanonicalLoweringPasses() {
  // The overload receiving an existing OpPassManager expects its contents.
  // Passing the anchored CLI form would nest another builtin.module manager
  // and leave a source module without nested modules completely untouched.
  return taskflowFissionCanonicalLoweringPipeline()
      .drop_front(llvm::StringRef("builtin.module(").size()).drop_back();
}

std::string printTaskflowFissionModuleWitness(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, OpPrintingFlags().printGenericOpForm());
  stream.flush();
  return text;
}

void registerTaskflowFissionCanonicalLoweringDependencies(
    DialectRegistry &registry) {
  OpPassManager manager(ModuleOp::getOperationName());
  if (failed(parsePassPipeline(taskflowFissionCanonicalLoweringPasses(),
                               manager, llvm::errs())))
    llvm::report_fatal_error("source-owned fission lowering pipeline is unavailable");
  manager.getDependentDialects(registry);
  mlir::amoeba::neura::createFissionTaskPass()->getDependentDialects(registry);
}

LogicalResult lowerPreparedTaskflowFissionSource(
    ModuleOp source, OwningOpRef<ModuleOp> &lowered, std::string &error) {
  lowered = nullptr;
  if (!source || failed(verify(source))) {
    error = "fission lowering requires a verified prepared source module";
    return failure();
  }
  bool hasTask = false, hasKernel = false;
  source.walk([&](taskflow::TaskflowTaskOp) { hasTask = true; });
  source.walk([&](mlir::neura::KernelOp) { hasKernel = true; });
  if (!hasTask || hasKernel) {
    error = "fission source must precede Taskflow-to-Neura conversion";
    return failure();
  }
  OwningOpRef<ModuleOp> trial = source.clone();
  PassManager manager(source.getContext());
  manager.enableVerifier(true);
  if (failed(parsePassPipeline(taskflowFissionCanonicalLoweringPasses(),
                               manager, llvm::errs()))) {
    error = "source-owned fission canonical lowering pipeline is unavailable";
    return failure();
  }
  ScopedDiagnosticHandler capture(source.getContext(), [&](Diagnostic &entry) {
    if (error.empty()) {
      llvm::raw_string_ostream stream(error);
      entry.print(stream);
    }
    return success();
  });
  if (failed(manager.run(*trial)) || failed(verify(*trial))) {
    if (error.empty())
      error = "fission candidate canonical lowering failed";
    return failure();
  }
  lowered = std::move(trial);
  return success();
}

LogicalResult verifyTaskflowFissionCanonicalLowering(
    ModuleOp source, ModuleOp canonicalNeura, std::string &error) {
  OwningOpRef<ModuleOp> expected;
  if (!canonicalNeura || failed(lowerPreparedTaskflowFissionSource(source, expected, error)))
    return failure();
  std::string expectedText = printTaskflowFissionModuleWitness(*expected);
  std::string canonicalText = printTaskflowFissionModuleWitness(canonicalNeura);
  if (expectedText != canonicalText) {
    size_t offset = 0;
    while (offset < expectedText.size() && offset < canonicalText.size() &&
           expectedText[offset] == canonicalText[offset])
      ++offset;
    error = "prepared fission source does not reproduce the exact canonical Neura module; first mismatch at byte " +
            std::to_string(offset) + "; expected: " + expectedText.substr(offset, 160) +
            "; actual: " + canonicalText.substr(offset, 160);
    return failure();
  }
  return success();
}

} // namespace mlir::amoeba::neura::joint_scheduling

namespace {
struct VerifyTaskflowFissionSourceReplayPass
    : PassWrapper<VerifyTaskflowFissionSourceReplayPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTaskflowFissionSourceReplayPass)
  VerifyTaskflowFissionSourceReplayPass() = default;
  VerifyTaskflowFissionSourceReplayPass(const VerifyTaskflowFissionSourceReplayPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const override { return "verify-taskflow-fission-source-replay"; }
  StringRef getDescription() const override { return "Verify exact source-owned DFG fission replay"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::amoeba::neura::joint_scheduling::
        registerTaskflowFissionCanonicalLoweringDependencies(registry);
  }
  Option<std::string> parentFile{*this, "parent-module", llvm::cl::init("")};
  Option<std::string> actionFile{*this, "actions", llvm::cl::init("")};
  Option<std::string> output{*this, "output", llvm::cl::init("")};
  Option<std::string> enumerateFunction{*this, "enumerate-function", llvm::cl::init("")};
  Option<int64_t> maxFissionActionsPerTask{
      *this, "max-fission-actions-per-task", llvm::cl::init(64)};

  void runOnOperation() override {
    auto fail = [&](const llvm::Twine &message) {
      getOperation().emitError(message);
      signalPassFailure();
    };
    if (!enumerateFunction.empty()) {
      if (!parentFile.empty() || !actionFile.empty() || output.empty() ||
          maxFissionActionsPerTask <= 0)
        return fail("source fission census requires output and a positive cap, without replay inputs");
      auto function = getOperation().lookupSymbol<mlir::func::FuncOp>(
          enumerateFunction.getValue());
      if (!function || function.isDeclaration())
        return fail("source fission census function is missing or external");
      llvm::json::Array tasks;
      std::string fatalError;
      function.walk([&](mlir::taskflow::TaskflowTaskOp task) {
        if (!fatalError.empty())
          return;
        std::string diagnostic;
        auto cuts = mlir::amoeba::neura::enumerateTaskflowFissionPartitions(
            task, static_cast<uint64_t>(maxFissionActionsPerTask.getValue()), diagnostic);
        if (failed(cuts)) {
          fatalError = task.getTaskName().str() + ":" + diagnostic;
          return;
        }
        llvm::json::Array partitions;
        for (const auto &cut : *cuts) {
          llvm::json::Array nodes;
          for (unsigned node : cut)
            nodes.push_back(static_cast<int64_t>(node));
          partitions.push_back(std::move(nodes));
        }
        tasks.push_back(llvm::json::Object{
            {"task", task.getTaskName().str()},
            {"status", cuts->empty() ? "unsupported" : "supported"},
            {"diagnostic", diagnostic},
            {"legal_cut_count", static_cast<int64_t>(cuts->size())},
            {"left_nodes", std::move(partitions)}});
      });
      if (!fatalError.empty())
        return fail("fission census overflow; no partial manifest published:" + fatalError);
      using namespace mlir::amoeba::neura::joint_scheduling;
      llvm::json::Object record{
          {"schema", "orbit-taskflow-fission-source-cut-census-v1"},
          {"status", "complete"}, {"function", enumerateFunction.getValue()},
          {"max_fission_actions_per_task", maxFissionActionsPerTask.getValue()},
          {"source_module_witness_bytes", printTaskflowFissionModuleWitness(getOperation())},
          {"tasks", std::move(tasks)}, {"mapper_invoked", false}};
      std::string error;
      if (!writeAtomically(output.getValue(), [&](llvm::raw_ostream &stream) {
            stream << llvm::json::Value(std::move(record)) << '\n'; return true;
          }, error))
        return fail(error);
      return;
    }
    if (parentFile.empty() || actionFile.empty())
      return fail("source fission verification requires parent-module and actions");
    auto parent = parseSourceFile<ModuleOp>(parentFile.getValue(), &getContext());
    auto bytes = llvm::MemoryBuffer::getFile(actionFile.getValue());
    if (!parent || !bytes)
      return fail("cannot read immutable parent or typed fission actions");
    auto document = llvm::json::parse((*bytes)->getBuffer());
    if (!document) {
      llvm::consumeError(document.takeError());
      return fail("typed fission actions contain invalid JSON");
    }
    auto *root = document->getAsObject();
    auto *rows = root ? root->getArray("actions") : nullptr;
    auto schema = root ? root->getString("schema") : std::optional<StringRef>();
    if (!root || root->size() != 2 ||
        !schema || *schema != "orbit-taskflow-fission-actions-v1" || !rows)
      return fail("unsupported typed source fission action schema");
    llvm::SmallVector<mlir::amoeba::neura::TaskflowFissionAction> actions;
    for (const auto &value : *rows) {
      auto *row = value.getAsObject();
      auto function = row ? row->getString("function") : std::optional<StringRef>();
      auto task = row ? row->getString("task") : std::optional<StringRef>();
      auto *cut = row ? row->getArray("left_nodes") : nullptr;
      if (!row || row->size() != 3 || !function || function->empty() ||
          !task || task->empty() || !cut || cut->empty())
        return fail("malformed typed source fission action");
      mlir::amoeba::neura::TaskflowFissionAction action;
      action.function = function->str();
      action.task = task->str();
      for (const auto &node : *cut) {
        auto index = node.getAsInteger();
        if (!index || *index < 0 ||
            static_cast<uint64_t>(*index) > std::numeric_limits<unsigned>::max())
          return fail("fission cut has an invalid source node ordinal");
        action.leftNodes.push_back(static_cast<unsigned>(*index));
      }
      actions.push_back(std::move(action));
    }
    std::string error;
    if (failed(mlir::amoeba::neura::verifyTaskflowFissionReplay(
            *parent, getOperation(), actions, error)))
      return fail(error);
    if (!output.empty()) {
      using namespace mlir::amoeba::neura::joint_scheduling;
      llvm::json::Object record{{"schema", "orbit-taskflow-fission-source-replay-v1"},
          {"status", "pass"}, {"source_replay_verified", true},
          {"parent_module", parentFile.getValue()}, {"actions_file", actionFile.getValue()},
          {"action_count", static_cast<int64_t>(actions.size())},
          {"comparison", "exact-generic-MLIR-without-debug-locations"},
          {"counter_domains", "retained-per-source-operation; not-disjoint-firing-partitions"},
          {"parent_module_witness_bytes", printTaskflowFissionModuleWitness(*parent)},
          {"candidate_module_witness_bytes", printTaskflowFissionModuleWitness(getOperation())}};
      if (!writeAtomically(output.getValue(), [&](llvm::raw_ostream &stream) {
            stream << llvm::json::Value(std::move(record)) << '\n'; return true;
          }, error))
        return fail(error);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::amoeba::neura::createVerifyTaskflowFissionSourceReplayPass() {
  return std::make_unique<VerifyTaskflowFissionSourceReplayPass>();
}

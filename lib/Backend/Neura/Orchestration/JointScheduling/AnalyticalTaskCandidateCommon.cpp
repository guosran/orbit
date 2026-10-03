//===- AnalyticalTaskCandidateCommon.cpp ---------------------*- C++ -*-===//
//
// Implements reusable task metadata, architecture fingerprints, and output
// helpers shared by analytical candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

// Infers the execution count recorded in analytical candidate manifests from
// constant Taskflow counter chains. Spatial and temporal candidate spaces use
// the same task metadata, so counter interpretation belongs to this shared
// candidate layer rather than either concrete orchestration strategy.
static FailureOr<std::optional<int64_t>>
inferStaticTaskTripCount(TaskflowTaskOp task, std::string &error) {
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (counters.empty()) {
    return std::optional<int64_t>{};
  }

  if (!task.getBody().hasOneBlock()) {
    error = "task " + task.getTaskName().str() +
            " must contain exactly one block to infer a static trip count";
    return failure();
  }

  SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex()) {
      children[parent].push_back(counter);
    } else {
      roots.push_back(counter);
    }
  }
  if (roots.empty()) {
    error = "task " + task.getTaskName().str() +
            " has counters but no root counter";
    return failure();
  }

  // Counter bounds in a Taskflow task are often task body block arguments.
  // Resolve those arguments through the corresponding execution input before
  // deciding whether the bound is static. Native preparation may also leave a
  // small signed arithmetic expression, such as `c63 + (-1)`, in the bound.
  // Keep every fold checked so an overflow or a dynamic operand fails closed.
  llvm::DenseSet<Value> activeStaticValues;
  std::function<FailureOr<int64_t>(Value)> constant_index =
      [&](Value value) -> FailureOr<int64_t> {
    if (!activeStaticValues.insert(value).second)
      return failure();

    auto clearActive = llvm::make_scope_exit(
        [&] { activeStaticValues.erase(value); });
    auto integer_attr = [](IntegerAttr integer) -> FailureOr<int64_t> {
      const llvm::APInt &value = integer.getValue();
      if (!value.isSignedIntN(64))
        return failure();
      return value.getSExtValue();
    };
    auto checked_add = [](int64_t lhs,
                          int64_t rhs) -> FailureOr<int64_t> {
      if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
          (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
        return failure();
      return lhs + rhs;
    };
    auto checked_sub = [](int64_t lhs,
                          int64_t rhs) -> FailureOr<int64_t> {
      if ((rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs) ||
          (rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs))
        return failure();
      return lhs - rhs;
    };

    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
      return constant.value();
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return integer_attr(integer);
      return failure();
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
      return constant_index(cast.getIn());
    if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
      return constant_index(cast.getIn());
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      FailureOr<int64_t> lhs = constant_index(add.getLhs());
      FailureOr<int64_t> rhs = constant_index(add.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return checked_add(*lhs, *rhs);
    }
    if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
      FailureOr<int64_t> lhs = constant_index(sub.getLhs());
      FailureOr<int64_t> rhs = constant_index(sub.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      return checked_sub(*lhs, *rhs);
    }
    // Static loop bounds may be normalized through an affine.apply, for
    // example `s0 - 1` on a compile-time task argument. Resolve all operands
    // recursively and fold the affine map without treating symbol-dynamic
    // values as static.
    if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
      SmallVector<Attribute> operandConstants;
      operandConstants.reserve(apply->getNumOperands());
      for (Value operand : apply->getOperands()) {
        FailureOr<int64_t> constant = constant_index(operand);
        if (failed(constant))
          return failure();
        operandConstants.push_back(IntegerAttr::get(
            IndexType::get(value.getContext()), *constant));
      }
      SmallVector<Attribute> folded;
      if (failed(apply.getAffineMap().constantFold(operandConstants, folded)) ||
          folded.size() != 1)
        return failure();
      if (auto integer = dyn_cast<IntegerAttr>(folded.front()))
        return integer_attr(integer);
      return failure();
    }

    auto blockArgument = dyn_cast<BlockArgument>(value);
    if (!blockArgument || blockArgument.getOwner() != &task.getBody().front())
      return failure();
    SmallVector<Value> executionInputs(task.getWillReads());
    executionInputs.append(task.getWillWrites().begin(),
                           task.getWillWrites().end());
    executionInputs.append(task.getValueInputs().begin(),
                           task.getValueInputs().end());
    unsigned argumentNumber = blockArgument.getArgNumber();
    if (argumentNumber >= executionInputs.size())
      return failure();
    return constant_index(executionInputs[argumentNumber]);
  };
  auto counter_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    FailureOr<int64_t> lower = constant_index(counter.getLowerBound());
    FailureOr<int64_t> upper = constant_index(counter.getUpperBound());
    FailureOr<int64_t> step = constant_index(counter.getStep());
    if (failed(lower) || failed(upper) || failed(step)) {
      return failure();
    }
    if (*step <= 0 || *upper <= *lower ||
        (*lower < 0 && *upper > std::numeric_limits<int64_t>::max() + *lower)) {
      return failure();
    }

    int64_t distance = *upper - *lower;
    return 1 + (distance - 1) / *step;
  };

  llvm::DenseSet<Operation *> active;
  llvm::DenseSet<Operation *> visited;
  std::function<FailureOr<int64_t>(TaskflowCounterOp)> chain_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    Operation *operation = counter.getOperation();
    if (!active.insert(operation).second || visited.contains(operation)) {
      error = "task " + task.getTaskName().str() +
              " has a cyclic or multiply referenced counter chain";
      return failure();
    }

    FailureOr<int64_t> count = counter_trip_count(counter);
    if (failed(count)) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    int64_t longest_child_chain = 1;
    auto found = children.find(counter.getCounterIndex());
    if (found != children.end()) {
      for (TaskflowCounterOp child : found->second) {
        FailureOr<int64_t> child_count = chain_trip_count(child);
        if (failed(child_count)) {
          return failure();
        }
        longest_child_chain = std::max(longest_child_chain, *child_count);
      }
    }
    if (*count > std::numeric_limits<int64_t>::max() / longest_child_chain) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    active.erase(operation);
    visited.insert(operation);
    return *count * longest_child_chain;
  };

  int64_t total = 1;
  for (TaskflowCounterOp root : roots) {
    FailureOr<int64_t> root_count = chain_trip_count(root);
    if (failed(root_count)) {
      return failure();
    }
    total = std::max(total, *root_count);
  }
  if (visited.size() != counters.size()) {
    error = "task " + task.getTaskName().str() +
            " has a counter disconnected from every root";
    return failure();
  }
  return std::optional<int64_t>{total};
}

// Resolves any compile-time trip count stored with each task. An explicit
// `trip_count` is authoritative; otherwise, constant Taskflow counter chains
// supply the count. Dynamic or invalid bounds fail instead of inventing a
// numeric value. A task without a counter represents one execution.
static FailureOr<int64_t> resolveAnalyticalTripCount(TaskflowTaskOp task,
                                                     std::string &error) {
  if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
    if (attr.getInt() <= 0) {
      error =
          "task " + task.getTaskName().str() + " has non-positive trip_count";
      return failure();
    }
    return attr.getInt();
  }

  FailureOr<std::optional<int64_t>> inferred =
      inferStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit positive trip_count or resolve the counter "
             "bounds first";
    return failure();
  }
  return inferred->value_or(1);
}

// Task names are the stable source task IDs in the candidate contract.
// Derived files bind them together with the source Git commit and graph ID;
// no file-content fingerprint is computed here.
// Collects task names, source-body identities, and available trip counts in
// walk order.
// The order is the task axis used by a spatial shape tuple, so duplicate names
// are rejected before they can make candidate records ambiguous.
FailureOr<SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadata(func::FuncOp func, std::string &error) {
  SmallVector<TaskMetadata> tasks;
  llvm::StringSet<> names;
  WalkResult walkResult = func.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    FailureOr<int64_t> tripCount = resolveAnalyticalTripCount(task, error);
    if (failed(tripCount))
      return WalkResult::interrupt();
    tasks.push_back({task, std::move(name), *tripCount});
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  if (tasks.empty()) {
    error = "function contains no taskflow.task operations";
    return failure();
  }
  return tasks;
}

std::string makeSequentialCandidateId(uint64_t index) {
  return "candidate-" + std::to_string(index);
}

void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error) {
  if (output.empty()) {
    error = "output path is required";
    return false;
  }
  SmallString<256> pattern(output);
  pattern += ".tmp-%%%%%%";
  SmallString<256> temporary;
  int descriptor = -1;
  std::error_code ec =
      llvm::sys::fs::createUniqueFile(pattern, descriptor, temporary);
  if (ec) {
    error = "cannot create temporary output " + temporary.str().str() + ": " +
            ec.message();
    return false;
  }

  bool ok = false;
  {
    llvm::raw_fd_ostream os(descriptor, /*shouldClose=*/true);
    ok = writeBody(os);
    // Check close errors as well as buffered writes.  Once handled, clear the
    // stream error so its destructor cannot terminate the compiler before the
    // caller records an incomplete result and removes the temporary file.
    os.close();
    if (os.has_error()) {
      error = "failed while writing " + output.str() + ": " +
              os.error().message();
      ok = false;
      os.clear_error();
    }
  }
  if (!ok) {
    llvm::sys::fs::remove(temporary);
    return false;
  }
  ec = llvm::sys::fs::rename(temporary, output);
  if (ec) {
    error = "cannot publish " + output.str() + ": " + ec.message();
    llvm::sys::fs::remove(temporary);
    return false;
  }
  return true;
}

// Selects the requested Taskflow function, or infers it when exactly one
// function contains tasks. This keeps every file-producing pass consistent.
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module, StringRef requested,
                                           std::string &error) {
  if (!requested.empty()) {
    auto function = module.lookupSymbol<func::FuncOp>(requested);
    if (!function) {
      error = "requested function " + requested.str() + " does not exist";
      return failure();
    }
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (!hasTask) {
      error = "requested function " + requested.str() +
              " contains no taskflow.task operations";
      return failure();
    }
    return function;
  }

  SmallVector<func::FuncOp> taskFunctions;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask)
      taskFunctions.push_back(function);
  }
  if (taskFunctions.size() != 1) {
    error = "expected exactly one function containing Taskflow tasks; use the "
            "function option when the module contains more than one";
    return failure();
  }
  return taskFunctions.front();
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

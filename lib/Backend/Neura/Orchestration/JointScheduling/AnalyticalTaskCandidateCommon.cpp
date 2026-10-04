//===- AnalyticalTaskCandidateCommon.cpp ---------------------*- C++ -*-===//
//
// Implements reusable task metadata, architecture fingerprints, and output
// helpers shared by analytical candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"

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

static bool validateTaskflowCounterAxisChain(TaskflowTaskOp task,
                                              unsigned expectedAxes,
                                              std::string &error) {
  SmallVector<TaskflowCounterOp> counters;
  SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  task.walk([&](TaskflowCounterOp counter) {
    counters.push_back(counter);
    if (Value parent = counter.getParentIndex())
      children[parent].push_back(counter);
    else
      roots.push_back(counter);
  });
  if (counters.size() != expectedAxes) {
    error = "task " + task.getTaskName().str() + " has " +
            std::to_string(counters.size()) +
            " current Taskflow counters for " + std::to_string(expectedAxes) +
            " source-represented axes";
    return false;
  }
  if (expectedAxes == 0)
    return counters.empty();
  if (roots.size() != 1) {
    error = "task " + task.getTaskName().str() +
            " source-represented counters do not form one chain";
    return false;
  }
  llvm::DenseSet<Operation *> visited;
  TaskflowCounterOp current = roots.front();
  while (current) {
    if (!visited.insert(current.getOperation()).second) {
      error = "task " + task.getTaskName().str() +
              " has a cyclic source-represented counter chain";
      return false;
    }
    auto found = children.find(current.getCounterIndex());
    if (found == children.end())
      break;
    if (found->second.size() != 1) {
      error = "task " + task.getTaskName().str() +
              " source-represented counters branch into multiple axes";
      return false;
    }
    current = found->second.front();
  }
  if (visited.size() != expectedAxes) {
    error = "task " + task.getTaskName().str() +
            " has disconnected source-represented counter axes";
    return false;
  }
  return true;
}

static bool hasUncertifiedSequentialControl(TaskflowTaskOp task) {
  bool found = false;
  task.walk([&](affine::AffineForOp) { found = true; });
  task.walk([&](scf::ForOp) { found = true; });
  task.walk([&](scf::WhileOp) { found = true; });
  task.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    if (name == "neura.phi" || name == "neura.counter" || name == "cf.br" ||
        name == "cf.cond_br" || name == "cf.switch")
      found = true;
  });
  return found;
}

static bool hasRetainedSequentialLoop(TaskflowTaskOp task) {
  bool found = false;
  task.walk([&](affine::AffineForOp) { found = true; });
  task.walk([&](scf::ForOp) { found = true; });
  task.walk([&](scf::WhileOp) { found = true; });
  return found;
}

static bool validateExplicitMapperFiringCount(
    TaskflowTaskOp task, int64_t expected, std::string &error,
    bool validateReplicaShardTripCount = true) {
  auto validateAttribute = [&](StringRef attribute) {
    if (!task->hasAttr(attribute))
      return true;
    auto count = task->getAttrOfType<IntegerAttr>(attribute);
    if (count && count.getInt() > 0 && count.getInt() == expected)
      return true;
    error = "task " + task.getTaskName().str() + " has " + attribute.str() +
            "=" + (count ? std::to_string(count.getInt()) : "<malformed>") +
            " but its current effective mapper-firing count is " +
            std::to_string(expected);
    return false;
  };
  if (!validateAttribute("trip_count") ||
      !validateAttribute("amoeba.selected_trip_count"))
    return false;
  // Replica shard counts describe the historical replica step. Source-owned
  // partition candidates validate that value against the authenticated ledger
  // volume after the latest replica step; it need not equal a later tiled
  // child's current mapper-firing count.
  return !validateReplicaShardTripCount ||
         validateAttribute("amoeba.replica.shard_trip_count");
}

static FailureOr<TaskIterationDomainMetadata>
resolveTaskIterationDomainImpl(TaskflowTaskOp task, std::string &error,
                               bool sourcePartitionCandidate) {
  TaskIterationDomainMetadata result;
  auto sourceAttr = task->getAttrOfType<DictionaryAttr>(
      kSourceIterationDomainAttr);
  if (task->hasAttr(kSourceIterationDomainAttr) && !sourceAttr) {
    error = "task " + task.getTaskName().str() +
            " has a malformed source-owned iteration-domain attribute";
    return failure();
  }
  if (!sourceAttr) {
    SmallVector<TaskflowCounterOp> counters;
    task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
    if (sourcePartitionCandidate) {
      if (hasRetainedSequentialLoop(task)) {
        error = "task " + task.getTaskName().str() +
                " retains a sequential loop outside source-domain coverage";
        return failure();
      }
      FailureOr<std::optional<int64_t>> inferred =
          inferStaticTaskTripCount(task, error);
      if (failed(inferred))
        return failure();
      int64_t currentCount = inferred->has_value() ? **inferred : 1;
      // The current trip-count fields are checked here. A replica shard
      // count may describe an earlier ledger step, so the enclosing complete
      // partition proof checks it against that step's authenticated volume.
      if (!validateExplicitMapperFiringCount(
              task, currentCount, error,
              /*validateReplicaShardTripCount=*/false))
        return failure();
      result.complete = false;
      result.countKnown = true;
      result.status = "source-partition-proof-candidate";
      result.taskflowTripCount = currentCount;
      result.effectiveMapperFiringCount = currentCount;
      result.sourceIterationWorkCount = currentCount;
      return result;
    }
    // Compatibility for uncertified historical inputs and small mapper fixtures.
    // These counts are diagnostic; certified searches reject such inputs at
    // their canonical boundary. Never substitute one for an unknown axis.
    if (hasRetainedSequentialLoop(task)) {
      error = "task " + task.getTaskName().str() +
              " has retained sequential control without source-domain coverage";
      return failure();
    }
    FailureOr<std::optional<int64_t>> inferred = inferStaticTaskTripCount(task, error);
    if (failed(inferred)) return failure();
    std::optional<int64_t> known = *inferred;
    if (!known) {
      if (auto explicitCount = task->getAttrOfType<IntegerAttr>("trip_count"))
        known = explicitCount.getInt();
      else if (!hasUncertifiedSequentialControl(task))
        known = 1;
    }
    if (!known || *known <= 0 ||
        !validateExplicitMapperFiringCount(task, *known, error)) {
      if (error.empty()) error = "task " + task.getTaskName().str() +
          " has no known mapper-firing count or source-domain proof";
      return failure();
    }
    result.complete = false;
    result.countKnown = true;
    result.status = "legacy-uncertified-diagnostic";
    result.taskflowTripCount = *known;
    result.effectiveMapperFiringCount = *known;
    result.sourceIterationWorkCount = *known;
    return result;
  }

  std::string parseError;
  FailureOr<SourceIterationDomainInfo> parsed =
      parseSourceIterationDomain(task, parseError);
  if (failed(parsed)) {
    error = std::move(parseError);
    return failure();
  }
  if (hasRetainedSequentialLoop(task)) {
    error = "task " + task.getTaskName().str() +
            " retains a sequential loop outside its complete source-domain "
            "certificate";
    return failure();
  }
  result.sourceCertified = true;
  result.complete = true;
  result.internalMultiplicity = parsed->internalMultiplicity;
  result.status = "certified-complete";
  for (const SourceIterationAxis &axis : parsed->axes)
    if (!axis.representedByTaskflow)
      result.expandedInternalExtents.push_back(axis.extent);

  unsigned representedAxes = llvm::count_if(
      parsed->axes, [](const SourceIterationAxis &axis) {
        return axis.representedByTaskflow;
      });
  if (!validateTaskflowCounterAxisChain(task, representedAxes, error))
    return failure();
  FailureOr<std::optional<int64_t>> inferred =
      inferStaticTaskTripCount(task, error);
  if (failed(inferred))
    return failure();
  if (representedAxes == 0) {
    result.taskflowTripCount = 1;
  } else {
    if (!inferred->has_value()) {
      error = "task " + task.getTaskName().str() +
              " has certified axes but no static current Taskflow extent";
      return failure();
    }
    result.taskflowTripCount = **inferred;
  }
  result.effectiveMapperFiringCount = result.taskflowTripCount;
  // The certificate records the original source-axis product, while the
  // counters above are the current mapper firing domain. They must agree for
  // a standalone task. A smaller product is valid only after a complete
  // source-owned shard-group proof establishes coverage of the original
  // domain; per-task lineage/count attributes cannot establish that proof.
  // Keep this resolver fail-closed until the enclosing C++ materialization
  // path has validated that complete group.
  const bool changedMapperDomain =
      result.taskflowTripCount != parsed->representedMultiplicity;
  if (changedMapperDomain && !sourcePartitionCandidate) {
    auto partitionProof = task->getAttrOfType<StringAttr>(
        kSourceIterationPartitionProofAttr);
    if (!partitionProof ||
        partitionProof.getValue() !=
            sourceIterationDomainCanonicalWitness(*parsed)) {
      error = "task " + task.getTaskName().str() +
              " has current Taskflow extent " +
              std::to_string(result.taskflowTripCount) +
              " but its source-owned represented domain is " +
              std::to_string(parsed->representedMultiplicity) +
              "; no complete source-owned partition proof is available";
      return failure();
    }
    result.status = "certified-source-partition";
  } else if (changedMapperDomain) {
    result.status = "source-partition-proof-candidate";
  }
  if (!changedMapperDomain && !sourcePartitionCandidate) {
    auto partitionProof = task->getAttrOfType<StringAttr>(
        kSourceIterationPartitionProofAttr);
    if (partitionProof &&
        partitionProof.getValue() !=
            sourceIterationDomainCanonicalWitness(*parsed)) {
      error = "task " + task.getTaskName().str() +
              " has a stale source iteration partition proof";
      return failure();
    }
    if (partitionProof)
      result.status = "certified-source-partition";
  }
  if (!checkedMultiply(result.taskflowTripCount,
                       result.internalMultiplicity,
                       result.sourceIterationWorkCount)) {
    error = "task " + task.getTaskName().str() +
            " source iteration work count exceeds int64";
    return failure();
  }
  if (!validateExplicitMapperFiringCount(
          task, result.effectiveMapperFiringCount, error))
    return failure();

  if (task->hasAttr(kSourceIterationCapturePendingAttr)) {
    error = "task " + task.getTaskName().str() +
            " source-domain certificate still needs trusted control binding";
    return failure();
  }
  auto binding = task->getAttrOfType<StringAttr>(
      kSourceIterationControlBindingAttr);
  auto sourceBinding = task->getAttrOfType<StringAttr>(
      kSourceIterationSourceControlBindingAttr);
  if (!binding || binding.getValue().empty() || !sourceBinding ||
      sourceBinding.getValue().empty()) {
    error = "task " + task.getTaskName().str() +
            " source-domain certificate has no current/source control binding";
    return failure();
  }
  if (!sourcePartitionCandidate) {
    std::string expectedBinding = currentSourceIterationControlBinding(
        task, sourceIterationDomainCanonicalWitness(*parsed));
    if (expectedBinding.empty() || binding.getValue() != expectedBinding) {
      size_t mismatch = 0;
      StringRef stored = binding.getValue();
      while (mismatch < stored.size() && mismatch < expectedBinding.size() &&
             stored[mismatch] == expectedBinding[mismatch]) ++mismatch;
      error = "task " + task.getTaskName().str() +
              " source-domain control binding is stale or forged at byte " +
              std::to_string(mismatch) + "; stored=" +
              stored.substr(mismatch, 180).str() + "; current=" +
              expectedBinding.substr(mismatch, 180);
      return failure();
    }
    if (!task->hasAttr(kSourceIterationPartitionProofAttr) &&
        sourceBinding.getValue() != expectedBinding) {
      error = "task " + task.getTaskName().str() +
              " source-origin control binding is stale without a verified "
              "partition rewrite";
      return failure();
    }
  }
  result.countKnown = true;
  return result;
}

FailureOr<TaskIterationDomainMetadata>
resolveTaskIterationDomain(TaskflowTaskOp task, std::string &error) {
  return resolveTaskIterationDomainImpl(task, error,
                                        /*sourcePartitionCandidate=*/false);
}

FailureOr<TaskIterationDomainMetadata>
resolveTaskIterationDomainForSourcePartitionProof(TaskflowTaskOp task,
                                                  std::string &error) {
  return resolveTaskIterationDomainImpl(task, error,
                                        /*sourcePartitionCandidate=*/true);
}

// Task names are the stable source task IDs in the candidate contract.
// Derived files bind them together with the source Git commit and graph ID;
// no file-content fingerprint is computed here.
// Collects task names, source-body identities, and available trip counts in
// walk order.
// The order is the task axis used by a spatial shape tuple, so duplicate names
// are rejected before they can make candidate records ambiguous.
static FailureOr<SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadataImpl(func::FuncOp func, std::string &error,
                                  bool sourcePartitionCandidate) {
  SmallVector<TaskMetadata> tasks;
  llvm::StringSet<> names;
  WalkResult walkResult = func.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    std::string domainError;
    FailureOr<TaskIterationDomainMetadata> domain =
        sourcePartitionCandidate
            ? resolveTaskIterationDomainForSourcePartitionProof(task,
                                                                domainError)
            : resolveTaskIterationDomain(task, domainError);
    if (failed(domain)) {
      error = std::move(domainError);
      return WalkResult::interrupt();
    }
    TaskMetadata metadata;
    metadata.op = task;
    metadata.name = std::move(name);
    metadata.tripCount = domain->effectiveMapperFiringCount;
    metadata.taskflowTripCount = domain->taskflowTripCount;
    metadata.sourceIterationMultiplicity = domain->internalMultiplicity;
    metadata.sourceIterationWorkCount = domain->sourceIterationWorkCount;
    metadata.sourceIterationDomainCertified = domain->sourceCertified;
    metadata.sourceIterationDomainComplete = domain->complete;
    metadata.tripCountKnown = domain->countKnown;
    metadata.sourceIterationDomainStatus = domain->status;
    metadata.sourceIterationDomainReason = domain->reason;
    metadata.expandedInternalExtents =
        std::move(domain->expandedInternalExtents);
    tasks.push_back(std::move(metadata));
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

FailureOr<SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadata(func::FuncOp func, std::string &error) {
  return collectAnalyticalTaskMetadataImpl(
      func, error, /*sourcePartitionCandidate=*/false);
}

FailureOr<SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadataForSourcePartitionProof(func::FuncOp func,
                                                     std::string &error) {
  return collectAnalyticalTaskMetadataImpl(
      func, error, /*sourcePartitionCandidate=*/true);
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

//===- ProveStaticActiveTransferShapesPass.cpp ----------------*- C++ -*-===//
// Derive conservative rectangular access bounds from Taskflow and Neura IR.
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Orchestration/JointScheduling/ProveStaticActiveTransferShapesPass.h"

#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace mlir::amoeba::neura::joint_scheduling {
namespace {

constexpr StringLiteral kLogicalTransferShapeAttr =
    "amoeba.logical_transfer_shape";
constexpr StringLiteral kActiveTransferShapeAttr =
    "amoeba.active_transfer_shape";
constexpr StringLiteral kActiveTransferCapacityShapeAttr =
    "amoeba.active_transfer_capacity_shape";
constexpr StringLiteral kActiveTransferProofAttr =
    "amoeba.active_transfer_proof";
constexpr StringLiteral kCallerNoAliasProvenAttr =
    "amoeba.input0_caller_noalias_proven";
constexpr StringLiteral kCallerProofModeAttr =
    "amoeba.input0_caller_noalias_proof_mode";
constexpr StringLiteral kCallerEvidencePathAttr =
    "amoeba.input0_caller_evidence_path";
constexpr StringLiteral kCallerFunctionAttr = "amoeba.input0_caller_function";
constexpr StringLiteral kCallerPreparedInputPathAttr =
    "amoeba.input0_caller_prepared_input_path";
constexpr StringLiteral kCallerStaticBoundAttr =
    "amoeba.input0_caller_static_bound";
constexpr StringLiteral kStaticBoundArg0Attr = "amoeba.static_bound.arg.0";
constexpr StringLiteral kProofSchema = "orbit-static-active-transfer-proof-v1";

using Shape = SmallVector<int64_t, 4>;
using namespace mlir::taskflow;
using namespace mlir::neura;

static std::optional<Shape> parseShapeAttribute(Attribute attribute) {
  if (auto dense = dyn_cast_or_null<DenseI64ArrayAttr>(attribute))
    return Shape(dense.asArrayRef().begin(), dense.asArrayRef().end());
  auto array = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!array)
    return std::nullopt;

  Shape shape;
  shape.reserve(array.size());
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer || !integer.getType().isInteger(64))
      return std::nullopt;
    shape.push_back(integer.getInt());
  }
  return shape;
}

static DenseI64ArrayAttr getShapeAttr(MLIRContext *context,
                                      ArrayRef<int64_t> shape) {
  return DenseI64ArrayAttr::get(context, shape);
}

static StringRef taskName(TaskflowTaskOp task) {
  auto name = task->getAttrOfType<StringAttr>("task_name");
  return name ? name.getValue() : StringRef("<unnamed>");
}

static bool containsMemRef(Type type) {
  if (isa<MemRefType>(type))
    return true;
  if (auto predicated = dyn_cast<PredicatedValue>(type))
    return containsMemRef(predicated.getValueType());
  return false;
}

static bool operationHasMemRefPayload(Operation *operation) {
  for (Value operand : operation->getOperands())
    if (containsMemRef(operand.getType()))
      return true;
  for (Value result : operation->getResults())
    if (containsMemRef(result.getType()))
      return true;
  return false;
}

static bool isMemRefLoadOrStore(Operation *operation) {
  return isa<memref::LoadOp, memref::StoreOp>(operation);
}

static bool isIndexedNeuraLoadOrStore(Operation *operation) {
  return isa<LoadIndexedOp, StoreIndexedOp>(operation);
}

static bool isKnownNeuraValueForwarder(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "neura.constant" || name == "neura.data_mov" ||
         name == "neura.grant_predicate" || name == "neura.grant_once" ||
         name == "neura.grant_always" || name == "neura.phi" ||
         name == "neura.phi_start" || name == "neura.reserve" ||
         name == "neura.ctrl_mov" || name == "neura.sel";
}

static bool isOpaqueMemoryOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  // Neura's unindexed/opaque operations and pointer arithmetic cannot be
  // assigned a complete rectangular range by this proof.
  return name == "neura.load" || name == "neura.store" ||
         name == "neura.memset" || name == "neura.gather" ||
         name == "neura.gep" || name == "llvm.load" || name == "llvm.store" ||
         name == "llvm.memcpy" || name == "llvm.memmove" ||
         name == "llvm.memset" || name == "llvm.getelementptr";
}

static std::optional<unsigned>
traceKernelInput(Value value, KernelOp kernel,
                 llvm::SmallDenseSet<Value, 16> &visiting, unsigned depth = 0) {
  if (!value || !kernel || !kernel.getBody().hasOneBlock() || depth >= 32 ||
      !visiting.insert(value).second)
    return std::nullopt;
  auto finish = [&](std::optional<unsigned> input) {
    visiting.erase(value);
    return input;
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() == &kernel.getBody().front() &&
        argument.getArgNumber() < kernel.getInputs().size())
      return finish(argument.getArgNumber());
    return finish(std::nullopt);
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return finish(std::nullopt);
  StringRef name = definition->getName().getStringRef();
  if (name == "neura.constant")
    return finish(detail::parseInputReference(definition->getAttr("value")));
  if (name == "neura.grant_once" || name == "neura.grant_always") {
    Attribute folded = definition->getAttr("constant_value");
    std::optional<unsigned> fromAttribute = detail::parseInputReference(folded);
    if (folded &&
        (!fromAttribute || *fromAttribute >= kernel.getInputs().size()))
      return finish(std::nullopt);
    if (definition->getNumOperands() == 0)
      return finish(fromAttribute);
    if (definition->getNumOperands() != 1)
      return finish(std::nullopt);
    auto fromOperand = traceKernelInput(definition->getOperand(0), kernel,
                                        visiting, depth + 1);
    if (!fromOperand || (fromAttribute && *fromAttribute != *fromOperand))
      return finish(std::nullopt);
    return finish(fromOperand);
  }
  if (name == "neura.data_mov" || name == "neura.grant_predicate" ||
      name == "neura.ctrl_mov" || name == "neura.phi_start") {
    if (definition->getNumOperands() == 0)
      return finish(std::nullopt);
    return finish(traceKernelInput(definition->getOperand(0), kernel, visiting,
                                   depth + 1));
  }
  if (name == "neura.phi" || name == "neura.sel") {
    unsigned firstValueOperand = name == "neura.sel" ? 1u : 0u;
    if (firstValueOperand >= definition->getNumOperands())
      return finish(std::nullopt);
    llvm::SmallDenseSet<Value, 16> firstPath = visiting;
    auto common = traceKernelInput(definition->getOperand(firstValueOperand),
                                   kernel, firstPath, depth + 1);
    if (!common)
      return finish(std::nullopt);
    for (Value operand :
         definition->getOperands().drop_front(firstValueOperand + 1)) {
      llvm::SmallDenseSet<Value, 16> branchPath = visiting;
      auto candidate = traceKernelInput(operand, kernel, branchPath, depth + 1);
      if (!candidate || *candidate != *common)
        return finish(std::nullopt);
    }
    return finish(common);
  }
  return finish(std::nullopt);
}

static std::optional<unsigned> traceKernelInput(Value value, KernelOp kernel) {
  llvm::SmallDenseSet<Value, 16> visiting;
  return traceKernelInput(value, kernel, visiting);
}

static std::optional<unsigned>
authenticatedAccessInput(Operation *operation, Value base, KernelOp kernel) {
  if (!kernel || !kernel.getBody().hasOneBlock())
    return std::nullopt;
  if (std::optional<unsigned> shared =
          detail::accessInput(operation, base, kernel))
    return shared;
  if (!kernel || !isa<LoadIndexedOp, StoreIndexedOp>(operation))
    return std::nullopt;
  bool isLoad = isa<LoadIndexedOp>(operation);
  StringRef expectedName = isLoad ? "lhs_value" : "rhs_value";
  StringRef oppositeName = isLoad ? "rhs_value" : "lhs_value";
  auto parseReference = [&](StringRef name,
                            bool &present) -> std::optional<unsigned> {
    Attribute attribute = operation->getAttr(name);
    present = static_cast<bool>(attribute);
    if (!attribute)
      return std::nullopt;
    auto input = detail::parseInputReference(attribute);
    if (!input || *input >= kernel.getInputs().size())
      return std::nullopt;
    return input;
  };
  bool expectedPresent = false;
  bool oppositePresent = false;
  auto expected = parseReference(expectedName, expectedPresent);
  auto opposite = parseReference(oppositeName, oppositePresent);
  if ((expectedPresent && !expected) || (oppositePresent && !opposite))
    return std::nullopt;
  std::optional<unsigned> actual;
  if (base) {
    actual = traceKernelInput(base, kernel);
    if (!actual)
      return std::nullopt;
  }
  if (actual && expected && *actual != *expected)
    return std::nullopt;
  if (actual)
    return actual;
  return expected;
}

static Attribute integerAttribute(MLIRContext *context, int64_t value) {
  return IntegerAttr::get(IntegerType::get(context, 64), value);
}

static bool hasNoAliasFact(func::FuncOp function, unsigned argumentIndex) {
  Attribute attribute = function.getArgAttr(argumentIndex, "amoeba.noalias");
  if (isa_and_nonnull<UnitAttr>(attribute))
    return true;
  auto boolean = dyn_cast_or_null<BoolAttr>(attribute);
  return boolean && boolean.getValue();
}

static ArrayAttr stringsAttribute(MLIRContext *context,
                                  ArrayRef<std::string> strings) {
  SmallVector<Attribute> values;
  values.reserve(strings.size());
  for (const std::string &value : strings)
    values.push_back(StringAttr::get(context, value));
  return ArrayAttr::get(context, values);
}

static DictionaryAttr axisToAttribute(MLIRContext *context,
                                      const ActiveTransferAxisRange &axis) {
  NamedAttrList fields;
  fields.set("proven", BoolAttr::get(context, axis.proven));
  fields.set("route", StringAttr::get(context, axis.route));
  fields.set("lower", integerAttribute(context, axis.lower));
  fields.set("upper", integerAttribute(context, axis.upper));
  if (axis.counterId)
    fields.set("counter_id", integerAttribute(context, *axis.counterId));
  if (!axis.reason.empty())
    fields.set("reason", StringAttr::get(context, axis.reason));
  return DictionaryAttr::get(context, fields);
}

static DictionaryAttr
accessToAttribute(MLIRContext *context,
                  const ActiveTransferAccessRecord &access) {
  NamedAttrList fields;
  fields.set("ordinal", integerAttribute(context, access.ordinal));
  fields.set("task", StringAttr::get(context, access.taskName));
  fields.set("operation", StringAttr::get(context, access.operation));
  if (access.inputIndex)
    fields.set("input_index", integerAttribute(context, *access.inputIndex));
  fields.set("proven", BoolAttr::get(context, access.proven));
  if (!access.reason.empty())
    fields.set("reason", StringAttr::get(context, access.reason));
  SmallVector<Attribute> axes;
  axes.reserve(access.axes.size());
  for (const ActiveTransferAxisRange &axis : access.axes)
    axes.push_back(axisToAttribute(context, axis));
  fields.set("axes", ArrayAttr::get(context, axes));
  return DictionaryAttr::get(context, fields);
}

struct CounterDomain {
  int64_t id = -1;
  int64_t lower = 0;
  int64_t upper = 0;
};

class StaticActiveTransferAnalyzer {
public:
  StaticActiveTransferAnalyzer(func::FuncOp function, unsigned argumentIndex)
      : function(function), argumentIndex(argumentIndex) {
    proof.argumentIndex = argumentIndex;
  }

  StaticActiveTransferShapeProof run() {
    if (!function || function.isDeclaration() || function.getBody().empty()) {
      fail("function must have a defined entry block");
      return std::move(proof);
    }
    if (argumentIndex >= function.getNumArguments()) {
      fail("argument index is outside the function signature");
      return std::move(proof);
    }
    root = function.getArgument(argumentIndex);
    auto rootType = dyn_cast<MemRefType>(root.getType());
    if (!rootType) {
      fail("selected argument is not a memref");
      return std::move(proof);
    }
    if (rootType.getRank() == 0)
      fail("active-transfer proof requires a positive-rank memref");

    std::optional<Shape> capacity = parseShapeAttribute(
        function.getArgAttr(argumentIndex, kLogicalTransferShapeAttr));
    if (!capacity) {
      fail("missing or malformed amoeba.logical_transfer_shape capacity");
    } else {
      proof.hasCapacity = true;
      proof.capacityShape = std::move(*capacity);
      if (proof.capacityShape.size() != static_cast<size_t>(rootType.getRank()))
        fail("logical transfer capacity rank disagrees with memref rank");
      if (llvm::any_of(proof.capacityShape,
                       [](int64_t extent) { return extent <= 0; }))
        fail("logical transfer capacity extents must be positive");
      if (proof.capacityShape.size() == static_cast<size_t>(rootType.getRank()))
        for (auto [axis, extent] : llvm::enumerate(proof.capacityShape)) {
          int64_t declaredExtent = rootType.getDimSize(axis);
          if (!ShapedType::isDynamic(declaredExtent) &&
              declaredExtent != extent)
            fail("logical transfer capacity disagrees with static memref "
                 "extent");
        }
    }

    validateCallerFacts();
    validateCallerBoundUse();
    auditFunctionArgumentUses();

    SmallVector<int64_t> lowerBounds(proof.capacityShape.size(),
                                     std::numeric_limits<int64_t>::max());
    SmallVector<int64_t> upperBounds(proof.capacityShape.size(),
                                     std::numeric_limits<int64_t>::min());
    SmallVector<bool> axisSeen(proof.capacityShape.size(), false);

    function.walk([&](Operation *operation) {
      bool indexed = isIndexedNeuraLoadOrStore(operation) ||
                     isMemRefLoadOrStore(operation);
      bool opaqueMemory = isOpaqueMemoryOperation(operation) ||
                          (operationHasMemRefPayload(operation) && !indexed &&
                           !isAllowedContainer(operation) &&
                           !isKnownNeuraValueForwarder(operation));
      if (!indexed && !opaqueMemory)
        return;

      ActiveTransferAccessRecord record;
      record.ordinal = accessOrdinal++;
      record.operation = operation->getName().getStringRef().str();
      TaskflowTaskOp task = operation->getParentOfType<TaskflowTaskOp>();
      if (task)
        record.taskName = taskName(task).str();
      else
        record.taskName = "<function>";

      Value accessRoot;
      SmallVector<Value> indices;
      std::optional<unsigned> inputIndex;
      std::string rootError;

      if (auto load = dyn_cast<LoadIndexedOp>(operation)) {
        auto kernel = load->getParentOfType<KernelOp>();
        inputIndex =
            authenticatedAccessInput(operation, load.getBase(), kernel);
        if (inputIndex && task && kernel)
          accessRoot =
              resolveKernelInputRoot(task, kernel, *inputIndex, rootError);
        else
          rootError =
              "indexed Neura load lacks an authenticated task/kernel input";
        for (Value index : load.getIndices())
          indices.push_back(index);
      } else if (auto store = dyn_cast<StoreIndexedOp>(operation)) {
        auto kernel = store->getParentOfType<KernelOp>();
        inputIndex =
            authenticatedAccessInput(operation, store.getBase(), kernel);
        if (inputIndex && task && kernel)
          accessRoot =
              resolveKernelInputRoot(task, kernel, *inputIndex, rootError);
        else
          rootError =
              "indexed Neura store lacks an authenticated task/kernel input";
        for (Value index : store.getIndices())
          indices.push_back(index);
      } else if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        accessRoot = resolveRoot(load.getMemref(), rootError).value_or(Value{});
        for (Value index : load.getIndices())
          indices.push_back(index);
      } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
        accessRoot =
            resolveRoot(store.getMemref(), rootError).value_or(Value{});
        for (Value index : store.getIndices())
          indices.push_back(index);
      } else {
        accessRoot = rootForOpaqueMemoryOperation(operation, rootError);
        if (!accessRoot) {
          record.reason =
              rootError.empty()
                  ? "opaque memory operation may alias selected storage"
                  : rootError;
          appendUnknownAccess(record);
          fail(record.reason);
          return;
        }
      }
      record.inputIndex = inputIndex;

      if (!accessRoot) {
        record.reason = rootError.empty()
                            ? "memory access storage root is not authenticated"
                            : rootError;
        appendUnknownAccess(record);
        fail(record.reason);
        return;
      }
      if (accessRoot != root) {
        if (!detail::provesDistinctStorage(accessRoot, root)) {
          record.reason = "other memory access may alias selected storage";
          appendUnknownAccess(record);
          fail(record.reason);
        }
        return;
      }

      if (opaqueMemory) {
        record.reason = "selected storage has an opaque memory operation";
        appendUnknownAccess(record);
        fail(record.reason);
        return;
      }

      if (indices.size() != proof.capacityShape.size()) {
        record.reason = "indexed access rank disagrees with capacity rank";
        appendUnknownAccess(record);
        fail(record.reason);
        return;
      }

      std::optional<KernelOp> kernel;
      if (auto parentKernel = operation->getParentOfType<KernelOp>())
        kernel = parentKernel;
      SmallVector<CounterDomain, 4> domains;
      SmallVector<CounterOp, 4> kernelCounters;
      bool counterProofAvailable = false;
      std::string counterError;
      if (task) {
        counterProofAvailable = getCounterDomains(task, kernel, domains,
                                                  kernelCounters, counterError);
      } else {
        counterProofAvailable = true;
      }
      if (!counterProofAvailable) {
        record.reason = counterError;
        appendUnknownAxes(record, indices.size(), counterError);
        appendUnknownAccess(record);
        fail(counterError);
        return;
      }

      bool accessProven = true;
      for (auto [axisIndex, index] : llvm::enumerate(indices)) {
        ActiveTransferAxisRange axis;
        if (kernel) {
          if (auto counterIndex = detail::counterRoute(index, kernelCounters)) {
            if (*counterIndex < domains.size()) {
              const CounterDomain &domain = domains[*counterIndex];
              axis.lower = domain.lower;
              axis.upper = domain.upper;
              axis.counterId = domain.id;
              axis.route = "matched-neura-counter";
              axis.proven = true;
            }
          }
        } else if (task) {
          for (const CounterDomain &domain : domains) {
            auto counters = getTaskCounters(task);
            if (domain.id >= 0 &&
                static_cast<size_t>(domain.id) < counters.size() &&
                index == counters[domain.id].getCounterIndex()) {
              axis.lower = domain.lower;
              axis.upper = domain.upper;
              axis.counterId = domain.id;
              axis.route = "taskflow-counter";
              axis.proven = true;
              break;
            }
          }
        }

        if (!axis.proven) {
          if (std::optional<int64_t> constant = detail::staticIndex(
                  index, task, kernel.value_or(KernelOp{}))) {
            if (*constant < 0 ||
                *constant == std::numeric_limits<int64_t>::max()) {
              axis.reason = "constant access index is negative or overflows";
            } else {
              axis.lower = *constant;
              axis.upper = *constant + 1;
              axis.route = "authenticated-constant";
              axis.proven = true;
            }
          } else {
            axis.reason =
                "index is neither a constant nor a matched counter route";
          }
        }

        if (axis.proven && axisIndex < proof.capacityShape.size() &&
            (axis.lower < 0 || axis.upper <= axis.lower ||
             axis.upper > proof.capacityShape[axisIndex])) {
          axis.proven = false;
          axis.reason = "access range is outside the original capacity";
        }
        if (!axis.proven) {
          accessProven = false;
          if (record.reason.empty())
            record.reason =
                "axis " + std::to_string(axisIndex) + ": " + axis.reason;
        } else {
          axisSeen[axisIndex] = true;
          lowerBounds[axisIndex] = std::min(lowerBounds[axisIndex], axis.lower);
          upperBounds[axisIndex] = std::max(upperBounds[axisIndex], axis.upper);
        }
        record.axes.push_back(std::move(axis));
      }
      record.proven = accessProven;
      if (!record.proven)
        fail(record.reason);
      proof.accesses.push_back(std::move(record));
    });

    if (proof.accesses.empty())
      fail("selected storage has no authenticated indexed load/store accesses");
    if (lowerBounds.size() != proof.capacityShape.size())
      fail("access rank does not match capacity rank");
    for (size_t axis = 0; axis < lowerBounds.size(); ++axis) {
      if (!axisSeen[axis]) {
        fail("no authenticated access covers capacity axis " +
             std::to_string(axis));
        continue;
      }
      if (lowerBounds[axis] != 0) {
        fail("active transfer box does not begin at zero on axis " +
             std::to_string(axis));
        continue;
      }
      proof.activeShape.push_back(upperBounds[axis]);
    }
    proof.proven = proof.reasons.empty() &&
                   proof.activeShape.size() == proof.capacityShape.size();
    if (!proof.proven)
      proof.activeShape.clear();
    return std::move(proof);
  }

private:
  void fail(StringRef reason) {
    if (reason.empty())
      return;
    if (!llvm::is_contained(proof.reasons, reason.str()))
      proof.reasons.push_back(reason.str());
  }

  void appendUnknownAccess(ActiveTransferAccessRecord record) {
    record.proven = false;
    if (record.axes.empty()) {
      ActiveTransferAxisRange axis;
      axis.reason = record.reason;
      record.axes.push_back(std::move(axis));
    }
    proof.accesses.push_back(std::move(record));
  }

  void appendUnknownAxes(ActiveTransferAccessRecord &record, size_t rank,
                         StringRef reason) {
    record.axes.clear();
    for (size_t index = 0; index < rank; ++index) {
      ActiveTransferAxisRange axis;
      axis.reason = reason.str();
      record.axes.push_back(std::move(axis));
    }
  }

  // A memref.cast may expose the same storage with dynamic/static shape
  // information only. The logical capacity authenticates each static extent;
  // layout, element type, address space, and rank must remain identical.
  bool isAuthenticatedIdentityMemRefView(Type sourceType, Type targetType,
                                         Value storageRoot,
                                         std::string &error,
                                         bool requireCapacity = false) {
    auto source = dyn_cast<MemRefType>(sourceType);
    auto target = dyn_cast<MemRefType>(targetType);
    if (!source || !target || source.getRank() != target.getRank() ||
        source.getElementType() != target.getElementType() ||
        source.getLayout() != target.getLayout() ||
        source.getMemorySpace() != target.getMemorySpace()) {
      error = "memref view changes rank, element type, layout, or memory "
              "space";
      return false;
    }
    if (source == target && !requireCapacity)
      return true;

    auto argument = dyn_cast<BlockArgument>(storageRoot);
    if (!argument || argument.getOwner() != &function.getBody().front() ||
        !isa<MemRefType>(storageRoot.getType())) {
      error = "memref view storage root is not a function memref argument";
      return false;
    }
    std::optional<Shape> capacity = parseShapeAttribute(function.getArgAttr(
        argument.getArgNumber(), kLogicalTransferShapeAttr));
    if (!capacity ||
        capacity->size() != static_cast<size_t>(source.getRank())) {
      error = "memref view has no well-formed logical transfer capacity";
      return false;
    }
    auto matchesCapacity = [&](MemRefType type) {
      for (auto [axis, extent] : llvm::enumerate(*capacity)) {
        int64_t declaredExtent = type.getDimSize(axis);
        if (!ShapedType::isDynamic(declaredExtent) && declaredExtent != extent)
          return false;
      }
      return true;
    };
    if (!matchesCapacity(source) || !matchesCapacity(target)) {
      error = "memref view static extents disagree with logical transfer "
              "capacity";
      return false;
    }
    return true;
  }

  bool isAllowedContainer(Operation *operation) {
    if (isa<TaskflowTaskOp, TaskflowYieldOp, TaskflowChannelOp,
            TaskflowCounterOp, KernelOp>(operation))
      return true;
    if (auto cast = dyn_cast<memref::CastOp>(operation)) {
      std::string error;
      std::optional<Value> sourceRoot = resolveRoot(cast.getSource(), error);
      return sourceRoot && isAuthenticatedIdentityMemRefView(
                               cast.getSource().getType(),
                               cast.getResult().getType(), *sourceRoot, error,
                               /*requireCapacity=*/true);
    }
    if (auto join = dyn_cast<TaskflowJoinOp>(operation)) {
      std::string error;
      return resolveRoot(join.getJoined(), error).has_value();
    }
    return false;
  }

  bool validateCallerFacts() {
    bool valid =
        isa_and_nonnull<UnitAttr>(function->getAttr(kCallerNoAliasProvenAttr));
    auto proofMode = function->getAttrOfType<StringAttr>(kCallerProofModeAttr);
    auto evidence =
        function->getAttrOfType<StringAttr>(kCallerEvidencePathAttr);
    auto caller = function->getAttrOfType<StringAttr>(kCallerFunctionAttr);
    auto preparedInput =
        function->getAttrOfType<StringAttr>(kCallerPreparedInputPathAttr);
    auto callerBound =
        function->getAttrOfType<IntegerAttr>(kCallerStaticBoundAttr);
    auto importedBound =
        function->getAttrOfType<IntegerAttr>(kStaticBoundArg0Attr);
    bool preparedCaller =
        proofMode && proofMode.getValue() == "prepared-external-caller";
    bool inModuleCaller =
        proofMode && proofMode.getValue() == "in-module-caller";
    valid &= preparedCaller || inModuleCaller;
    valid &= evidence && !evidence.getValue().empty();
    valid &= caller && !caller.getValue().empty();
    if (preparedCaller)
      valid &= preparedInput && !preparedInput.getValue().empty();
    else if (inModuleCaller)
      valid &= !preparedInput;
    if (callerBound || importedBound)
      valid &= callerBound && importedBound &&
               callerBound.getType().isInteger(64) &&
               importedBound.getType().isInteger(64) &&
               callerBound.getInt() == importedBound.getInt() &&
               callerBound.getInt() > 0;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (!isa<MemRefType>(function.getArgument(index).getType()))
        continue;
      if (!hasNoAliasFact(function, index))
        valid = false;
    }
    trustedCallerBound = valid && callerBound && importedBound;
    if (!valid)
      fail("caller no-alias/static-bound provenance is incomplete");
    return valid;
  }

  std::optional<Value> resolveRoot(Value value,
                                   llvm::SmallDenseSet<Value, 16> &visiting,
                                   std::string &error) {
    if (!value || !visiting.insert(value).second) {
      error = "memory root route is empty or cyclic";
      return std::nullopt;
    }
    auto finish = [&](std::optional<Value> result) {
      visiting.erase(value);
      return result;
    };

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      if (auto ownerFunction = dyn_cast_or_null<func::FuncOp>(parent)) {
        if (argument.getOwner() == &ownerFunction.getBody().front() &&
            isa<MemRefType>(value.getType()))
          return finish(value);
      }
      if (auto task = dyn_cast_or_null<TaskflowTaskOp>(parent)) {
        if (argument.getOwner() != &task.getBody().front()) {
          error = "Taskflow argument is not in the task entry block";
          return finish(std::nullopt);
        }
        unsigned input = argument.getArgNumber();
        unsigned stateInputs =
            task.getWillReads().size() + task.getWillWrites().size();
        if (input >= stateInputs) {
          error = "memref route crosses a Taskflow value input";
          return finish(std::nullopt);
        }
        std::optional<Value> rootValue =
            resolveTaskStateInput(task, input, visiting, error);
        return finish(rootValue);
      }
      error = "memref block argument is outside function/task memory state";
      return finish(std::nullopt);
    }

    Operation *definition = value.getDefiningOp();
    if (auto task = dyn_cast_or_null<TaskflowTaskOp>(definition)) {
      unsigned resultNumber = cast<OpResult>(value).getResultNumber();
      ValueRange readResults = task.getDoneReads();
      ValueRange writeResults = task.getDoneWrites();
      Value yielded;
      if (resultNumber < readResults.size()) {
        auto terminator =
            dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
        if (!terminator || resultNumber >= terminator.getDoneReads().size()) {
          error = "Taskflow read result has no corresponding yielded state";
          return finish(std::nullopt);
        }
        yielded = terminator.getDoneReads()[resultNumber];
      } else {
        unsigned writeNumber = resultNumber - readResults.size();
        if (writeNumber >= writeResults.size()) {
          error = "memory root is a non-state Taskflow result";
          return finish(std::nullopt);
        }
        auto terminator =
            dyn_cast<TaskflowYieldOp>(task.getBody().front().getTerminator());
        if (!terminator || writeNumber >= terminator.getDoneWrites().size()) {
          error = "Taskflow write result has no corresponding yielded state";
          return finish(std::nullopt);
        }
        yielded = terminator.getDoneWrites()[writeNumber];
      }
      if (!yielded) {
        error = "Taskflow state result has an empty yield route";
        return finish(std::nullopt);
      }
      if (yielded.getType() != value.getType()) {
        error = "Taskflow result changes the yielded memref type or layout";
        return finish(std::nullopt);
      }
      std::optional<Value> rootValue = resolveRoot(yielded, visiting, error);
      return finish(rootValue);
    }

    if (auto memrefCast = dyn_cast_or_null<memref::CastOp>(definition)) {
      std::optional<Value> sourceRoot =
          resolveRoot(memrefCast.getSource(), visiting, error);
      if (!sourceRoot)
        return finish(std::nullopt);
      if (!isAuthenticatedIdentityMemRefView(memrefCast.getSource().getType(),
                                             value.getType(), *sourceRoot,
                                             error,
                                             /*requireCapacity=*/true))
        return finish(std::nullopt);
      return finish(sourceRoot);
    }

    // A completion join is an identity alias only after its base and every
    // tile state independently resolve to the same original storage root.
    if (auto join = dyn_cast_or_null<TaskflowJoinOp>(definition)) {
      if (join.getJoined() != value || join.getTileStates().size() < 2) {
        error = "Taskflow join result or tile list is malformed";
        return finish(std::nullopt);
      }
      std::optional<Value> baseRoot =
          resolveRoot(join.getBase(), visiting, error);
      if (!baseRoot)
        return finish(std::nullopt);
      if (!isAuthenticatedIdentityMemRefView(join.getBase().getType(),
                                             baseRoot->getType(), *baseRoot,
                                             error) ||
          !isAuthenticatedIdentityMemRefView(
              value.getType(), baseRoot->getType(), *baseRoot, error))
        return finish(std::nullopt);

      for (Value tileState : join.getTileStates()) {
        std::optional<Value> tileRoot = resolveRoot(tileState, visiting, error);
        if (!tileRoot)
          return finish(std::nullopt);
        if (*tileRoot != *baseRoot) {
          error = "Taskflow join tile states do not resolve to the same "
                  "original storage root as its base";
          return finish(std::nullopt);
        }
        if (!isAuthenticatedIdentityMemRefView(
                tileState.getType(), baseRoot->getType(), *baseRoot, error))
          return finish(std::nullopt);
      }
      auto baseArgument = dyn_cast<BlockArgument>(*baseRoot);
      if (!baseArgument ||
          baseArgument.getOwner() != &function.getBody().front()) {
        error = "Taskflow join base does not resolve to a function argument";
        return finish(std::nullopt);
      }
      std::optional<Shape> capacity = parseShapeAttribute(function.getArgAttr(
          baseArgument.getArgNumber(), kLogicalTransferShapeAttr));
      int64_t axis = join.getAxis();
      ArrayRef<int64_t> lower = join.getRegionLower();
      ArrayRef<int64_t> upper = join.getRegionUpper();
      if (!capacity || lower.size() != capacity->size() ||
          upper.size() != capacity->size() || axis < 0 ||
          static_cast<size_t>(axis) >= capacity->size()) {
        error = "Taskflow join region does not match the logical capacity rank";
        return finish(std::nullopt);
      }
      for (auto [dimension, extent] : llvm::enumerate(*capacity))
        if (lower[dimension] < 0 || upper[dimension] <= lower[dimension] ||
            upper[dimension] > extent) {
          error = "Taskflow join region is outside the logical transfer "
                  "capacity";
          return finish(std::nullopt);
        }
      return finish(baseRoot);
    }

    error = "memory root is produced by an unsupported alias operation";
    return finish(std::nullopt);
  }

  std::optional<Value> resolveRoot(Value value, std::string &error) {
    llvm::SmallDenseSet<Value, 16> visiting;
    return resolveRoot(value, visiting, error);
  }

  std::optional<Value>
  resolveTaskStateInput(TaskflowTaskOp task, unsigned stateIndex,
                        llvm::SmallDenseSet<Value, 16> &visiting,
                        std::string &error) {
    unsigned readCount = task.getWillReads().size();
    unsigned writeCount = task.getWillWrites().size();
    unsigned valueCount = task.getValueInputs().size();
    if (stateIndex >= readCount + writeCount) {
      error = "Taskflow input index is not a memory-state input";
      return std::nullopt;
    }
    if (!task.getBody().hasOneBlock() ||
        task.getBody().front().getNumArguments() !=
            readCount + writeCount + valueCount) {
      error = "Taskflow body arguments do not match memory/value inputs";
      return std::nullopt;
    }

    Value state = task->getOperand(stateIndex);
    if (task.getBody().front().getArgument(stateIndex).getType() !=
        state.getType()) {
      error = "Taskflow memory-state input changes its memref type or layout";
      return std::nullopt;
    }
    Value declaredRoot;
    if (stateIndex < readCount) {
      if (stateIndex < task.getOriginalReadMemrefs().size())
        declaredRoot = task.getOriginalReadMemrefs()[stateIndex];
    } else {
      unsigned writeIndex = stateIndex - readCount;
      if (writeIndex < task.getOriginalWriteMemrefs().size())
        declaredRoot = task.getOriginalWriteMemrefs()[writeIndex];
    }
    llvm::SmallDenseSet<Value, 16> stateVisiting = visiting;
    std::string stateError;
    std::optional<Value> stateRoot =
        resolveRoot(state, stateVisiting, stateError);
    if (!stateRoot) {
      error = "Taskflow original-memory root does not authenticate its state";
      if (!stateError.empty())
        error += ": " + stateError;
      return std::nullopt;
    }
    if (!isAuthenticatedIdentityMemRefView(
            state.getType(), stateRoot->getType(), *stateRoot, stateError)) {
      error =
          "Taskflow state route changes the root memref view: " + stateError;
      return std::nullopt;
    }
    // Original-root sidecars are checked whenever present. Some Taskflow
    // tasks deliberately collapse several write states to one result and
    // consequently have fewer sidecars than states; in that case the exact
    // yielded block argument and corresponding state operand above provide the
    // provenance route, as in TaskEdgeContract::resolveOriginalMemrefRoot.
    if (declaredRoot) {
      llvm::SmallDenseSet<Value, 16> rootVisiting = visiting;
      std::string declaredError;
      std::optional<Value> sidecarRoot =
          resolveRoot(declaredRoot, rootVisiting, declaredError);
      if (!sidecarRoot || *stateRoot != *sidecarRoot ||
          !isAuthenticatedIdentityMemRefView(declaredRoot.getType(),
                                             state.getType(), *stateRoot,
                                             declaredError)) {
        error = "Taskflow original-memory root does not authenticate its state";
        if (!declaredError.empty())
          error += ": " + declaredError;
        return std::nullopt;
      }
    }
    return stateRoot;
  }

  std::optional<Value> resolveTaskStateInput(TaskflowTaskOp task,
                                             unsigned stateIndex,
                                             std::string &error) {
    llvm::SmallDenseSet<Value, 16> visiting;
    return resolveTaskStateInput(task, stateIndex, visiting, error);
  }

  Value resolveKernelInputRoot(TaskflowTaskOp task, KernelOp kernel,
                               unsigned inputIndex, std::string &error) {
    if (!kernel || inputIndex >= kernel.getInputs().size() ||
        !kernel.getBody().hasOneBlock()) {
      error = "kernel input index is out of range or kernel body is malformed";
      return {};
    }
    Value input = kernel.getInputs()[inputIndex];
    auto argument = dyn_cast<BlockArgument>(input);
    if (!argument || argument.getOwner() != &task.getBody().front()) {
      error = "kernel input does not map to a Taskflow state argument";
      return {};
    }
    unsigned stateIndex = argument.getArgNumber();
    unsigned stateCount =
        task.getWillReads().size() + task.getWillWrites().size();
    if (stateIndex >= stateCount) {
      error = "kernel memory input maps to a Taskflow value input";
      return {};
    }
    std::optional<Value> actualRoot =
        resolveTaskStateInput(task, stateIndex, error);
    std::optional<Value> annotatedRoot =
        detail::taskRootForKernelInput(inputIndex, kernel, task);
    if (!actualRoot) {
      if (error.empty())
        error = "kernel input has no authenticated Taskflow state root";
      return {};
    }
    if (!isAuthenticatedIdentityMemRefView(
            input.getType(), actualRoot->getType(), *actualRoot, error)) {
      error = "kernel input changes the Taskflow root memref view: " + error;
      return {};
    }
    if (!annotatedRoot) {
      // The Taskflow op may intentionally have fewer original-write sidecars
      // than state operands. Accept only that missing-entry case after the
      // exact body-argument -> task operand -> producer-yield chain above has
      // independently resolved the root. A present but malformed sidecar is
      // rejected by resolveTaskStateInput.
      unsigned readCount = task.getWillReads().size();
      bool hasSidecar =
          stateIndex < readCount
              ? stateIndex < task.getOriginalReadMemrefs().size()
              : stateIndex - readCount < task.getOriginalWriteMemrefs().size();
      if (!hasSidecar)
        return *actualRoot;
      error = "kernel input has no unique original Taskflow memory root";
      return {};
    }
    std::string annotationError;
    std::optional<Value> resolvedAnnotation =
        resolveRoot(*annotatedRoot, annotationError);
    if (!resolvedAnnotation || *resolvedAnnotation != *actualRoot) {
      error = "kernel input root disagrees with its Taskflow state route";
      return {};
    }
    return *actualRoot;
  }

  Value rootForOpaqueMemoryOperation(Operation *operation, std::string &error) {
    // An opaque operation is conservatively associated with a direct memref
    // operand where possible. Wrapped Neura memory pointers are ambiguous, so
    // the caller invalidates the selected root instead of guessing.
    if (auto join = dyn_cast<TaskflowJoinOp>(operation)) {
      std::optional<Value> joinedRoot = resolveRoot(join.getJoined(), error);
      return joinedRoot.value_or(Value{});
    }
    for (Value operand : operation->getOperands()) {
      if (!isa<MemRefType>(operand.getType()))
        continue;
      std::optional<Value> resolved = resolveRoot(operand, error);
      if (resolved)
        return *resolved;
    }
    error = "opaque operation's memory root cannot be resolved";
    return {};
  }

  bool auditFunctionArgumentUses() {
    bool valid = true;
    for (OpOperand &use : root.getUses()) {
      Operation *owner = use.getOwner();
      auto task = dyn_cast<TaskflowTaskOp>(owner);
      if (task) {
        unsigned readCount = task.getWillReads().size();
        unsigned writeCount = task.getWillWrites().size();
        unsigned valueCount = task.getValueInputs().size();
        unsigned originalReadCount = task.getOriginalReadMemrefs().size();
        unsigned operand = use.getOperandNumber();
        if (operand < readCount) {
          std::string error;
          if (!resolveTaskStateInput(task, operand, error)) {
            fail("function argument state route is unproven: " + error);
            valid = false;
          }
        } else if (operand < readCount + writeCount) {
          std::string error;
          if (!resolveTaskStateInput(task, operand, error)) {
            fail("function argument state route is unproven: " + error);
            valid = false;
          }
        } else if (operand < readCount + writeCount + valueCount) {
          fail("memref function argument is passed as a Taskflow value input");
          valid = false;
        } else if (operand <
                   readCount + writeCount + valueCount + originalReadCount) {
          unsigned stateIndex = operand - readCount - writeCount - valueCount;
          std::string error;
          if (stateIndex >= readCount ||
              !resolveTaskStateInput(task, stateIndex, error)) {
            fail("function argument original-read sidecar is unmatched");
            valid = false;
          }
        } else {
          unsigned stateIndex =
              operand - readCount - writeCount - valueCount - originalReadCount;
          std::string error;
          if (stateIndex >= writeCount ||
              !resolveTaskStateInput(task, readCount + stateIndex, error)) {
            fail("function argument original-write sidecar is unmatched");
            valid = false;
          }
        }
        continue;
      }

      if (auto memrefCast = dyn_cast<memref::CastOp>(owner)) {
        std::string error;
        std::optional<Value> castRoot =
            resolveRoot(memrefCast.getResult(), error);
        if (castRoot && *castRoot == root)
          continue;
        fail("function memref argument cast route is unproven: " + error);
        valid = false;
        continue;
      }

      if (auto join = dyn_cast<TaskflowJoinOp>(owner)) {
        std::string error;
        std::optional<Value> joinedRoot = resolveRoot(join.getJoined(), error);
        if (joinedRoot && *joinedRoot == root)
          continue;
        fail("function memref argument join route is unproven: " + error);
        valid = false;
        continue;
      }

      // Direct memref.load/store are supported by the same range proof.
      bool directLoad = false;
      bool directStore = false;
      if (auto load = dyn_cast<memref::LoadOp>(owner))
        directLoad = load.getMemref() == root;
      if (auto store = dyn_cast<memref::StoreOp>(owner))
        directStore = store.getMemref() == root;
      if (directLoad || directStore)
        continue;

      fail("function memref argument has an opaque or unindexed user: " +
           owner->getName().getStringRef().str());
      valid = false;
    }
    return valid;
  }

  SmallVector<TaskflowCounterOp, 4> getTaskCounters(TaskflowTaskOp task) const {
    SmallVector<TaskflowCounterOp, 4> counters;
    if (!task || !task.getBody().hasOneBlock())
      return counters;
    for (Operation &operation : task.getBody().front())
      if (auto counter = dyn_cast<TaskflowCounterOp>(&operation))
        counters.push_back(counter);
    llvm::sort(counters, [](TaskflowCounterOp lhs, TaskflowCounterOp rhs) {
      auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
      auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
      if (!lhsId || !rhsId)
        return static_cast<bool>(lhsId);
      return lhsId.getInt() < rhsId.getInt();
    });
    return counters;
  }

  bool getCounterDomains(TaskflowTaskOp task, std::optional<KernelOp> kernel,
                         SmallVectorImpl<CounterDomain> &domains,
                         SmallVectorImpl<CounterOp> &kernelCounters,
                         std::string &error) {
    SmallVector<TaskflowCounterOp, 4> taskCounters = getTaskCounters(task);
    for (auto [index, counter] : llvm::enumerate(taskCounters)) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      auto lower = detail::staticIndex(counter.getLowerBound(), task);
      auto upper = detail::staticIndex(counter.getUpperBound(), task);
      auto step = detail::staticIndex(counter.getStep(), task);
      bool parentOK = index == 0
                          ? !counter.getParentIndex()
                          : counter.getParentIndex() ==
                                taskCounters[index - 1].getCounterIndex();
      if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
          !upper || !step || *lower < 0 || *upper <= *lower || *step != 1 ||
          !parentOK) {
        error =
            "Taskflow counter IDs, parent chain, bounds, or unit step invalid";
        return false;
      }
      domains.push_back({id.getInt(), *lower, *upper});
    }

    if (!kernel)
      return true;
    if (!kernel->getBody().hasOneBlock()) {
      error = "Neura kernel must have one block";
      return false;
    }
    for (Operation &operation : kernel->getBody().front())
      if (auto counter = dyn_cast<CounterOp>(&operation))
        kernelCounters.push_back(counter);
    llvm::sort(kernelCounters, [](CounterOp lhs, CounterOp rhs) {
      auto lhsId = lhs->getAttrOfType<IntegerAttr>("counter_id");
      auto rhsId = rhs->getAttrOfType<IntegerAttr>("counter_id");
      if (!lhsId || !rhsId)
        return static_cast<bool>(lhsId);
      return lhsId.getInt() < rhsId.getInt();
    });
    if (kernelCounters.size() != taskCounters.size()) {
      error = "Taskflow and Neura counter counts differ";
      return false;
    }
    for (auto [index, counter] : llvm::enumerate(kernelCounters)) {
      auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
      auto lower = detail::staticCounterAttribute(
          counter.getOperation(), "lower_bound_value", counter.getLowerBound(),
          task, *kernel);
      auto upper = detail::staticCounterAttribute(
          counter.getOperation(), "upper_bound_value", counter.getUpperBound(),
          task, *kernel);
      auto step =
          detail::staticCounterAttribute(counter.getOperation(), "step_value",
                                         counter.getStep(), task, *kernel);
      if (!id || id.getInt() != static_cast<int64_t>(index) || !lower ||
          !upper || !step || *lower != domains[index].lower ||
          *upper != domains[index].upper || *step != 1) {
        error = "Neura counter bounds disagree with Taskflow counter routes";
        return false;
      }
    }
    return true;
  }

  void validateCallerBoundUse() {
    bool hasAnyBound = false;
    // The imported caller fact is specifically the input-0 bound. Reject
    // additional unverified static-bound annotations that the shared helper
    // might otherwise fold as though they were caller-authenticated.
    for (NamedAttribute attribute : function->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (!name.starts_with("amoeba.static_bound.arg."))
        continue;
      hasAnyBound = true;
      if (!trustedCallerBound || name != kStaticBoundArg0Attr ||
          !isa<IntegerAttr>(attribute.getValue()) ||
          cast<IntegerAttr>(attribute.getValue()).getInt() !=
              function->getAttrOfType<IntegerAttr>(kCallerStaticBoundAttr)
                  .getInt())
        fail("static-bound metadata is not bound to imported input-0 evidence");
    }
    if (hasAnyBound && !trustedCallerBound)
      fail(
          "static-bound metadata is present without imported input-0 evidence");
  }

  StaticActiveTransferShapeProof proof;
  func::FuncOp function;
  unsigned argumentIndex;
  Value root;
  uint64_t accessOrdinal = 0;
  bool trustedCallerBound = false;
};

} // namespace

StaticActiveTransferShapeProof
analyzeStaticActiveTransferShape(func::FuncOp function,
                                 unsigned argumentIndex) {
  StaticActiveTransferAnalyzer analyzer(function, argumentIndex);
  return analyzer.run();
}

DictionaryAttr serializeStaticActiveTransferShapeProof(
    MLIRContext *context, func::FuncOp function,
    const StaticActiveTransferShapeProof &proof) {
  NamedAttrList fields;
  fields.set("schema", StringAttr::get(context, kProofSchema));
  fields.set("status",
             StringAttr::get(context, proof.proven ? "proven" : "unknown"));
  fields.set("function", StringAttr::get(context, function.getSymName()));
  fields.set("argument", integerAttribute(context, proof.argumentIndex));
  fields.set("capacity_present", BoolAttr::get(context, proof.hasCapacity));
  fields.set("capacity_shape", getShapeAttr(context, proof.capacityShape));
  fields.set("caller_noalias_proven",
             BoolAttr::get(context, isa_and_nonnull<UnitAttr>(function->getAttr(
                                        kCallerNoAliasProvenAttr))));
  SmallVector<Attribute> noAliasArguments;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (!isa<MemRefType>(function.getArgument(index).getType()))
      continue;
    NamedAttrList fact;
    fact.set("index", integerAttribute(context, index));
    fact.set("proven", BoolAttr::get(context, hasNoAliasFact(function, index)));
    noAliasArguments.push_back(DictionaryAttr::get(context, fact));
  }
  fields.set("noalias_arguments", ArrayAttr::get(context, noAliasArguments));
  if (proof.proven)
    fields.set("active_shape", getShapeAttr(context, proof.activeShape));
  auto proofMode = function->getAttrOfType<StringAttr>(kCallerProofModeAttr);
  auto evidence = function->getAttrOfType<StringAttr>(kCallerEvidencePathAttr);
  auto caller = function->getAttrOfType<StringAttr>(kCallerFunctionAttr);
  auto staticBound =
      function->getAttrOfType<IntegerAttr>(kCallerStaticBoundAttr);
  if (proofMode)
    fields.set("caller_proof_mode", proofMode);
  if (evidence)
    fields.set("caller_evidence", evidence);
  if (caller)
    fields.set("caller_function", caller);
  auto preparedInput =
      function->getAttrOfType<StringAttr>(kCallerPreparedInputPathAttr);
  if (preparedInput)
    fields.set("caller_prepared_input", preparedInput);
  if (staticBound)
    fields.set("caller_static_bound", staticBound);
  fields.set("reasons", stringsAttribute(context, proof.reasons));
  SmallVector<Attribute> accesses;
  accesses.reserve(proof.accesses.size());
  for (const ActiveTransferAccessRecord &access : proof.accesses)
    accesses.push_back(accessToAttribute(context, access));
  fields.set("accesses", ArrayAttr::get(context, accesses));
  return DictionaryAttr::get(context, fields);
}

LogicalResult verifyStaticActiveTransferShapeProof(func::FuncOp function,
                                                   unsigned argumentIndex,
                                                   std::string *errorMessage) {
  auto failWith = [&](StringRef message) {
    if (errorMessage)
      *errorMessage = message.str();
    return failure();
  };
  if (!function || argumentIndex >= function.getNumArguments())
    return failWith("function argument index is invalid");

  StaticActiveTransferShapeProof proof =
      analyzeStaticActiveTransferShape(function, argumentIndex);
  Attribute proofAttribute =
      function.getArgAttr(argumentIndex, kActiveTransferProofAttr);
  auto storedProof = dyn_cast_or_null<DictionaryAttr>(proofAttribute);
  if (!storedProof)
    return failWith("active-transfer proof record is missing or malformed");
  DictionaryAttr expected = serializeStaticActiveTransferShapeProof(
      function.getContext(), function, proof);
  if (storedProof != expected)
    return failWith(
        "active-transfer proof record does not match re-derived IR facts");

  std::optional<Shape> logicalCapacity = parseShapeAttribute(
      function.getArgAttr(argumentIndex, kLogicalTransferShapeAttr));
  Attribute capacityAttribute =
      function.getArgAttr(argumentIndex, kActiveTransferCapacityShapeAttr);
  if (logicalCapacity) {
    auto storedCapacity = parseShapeAttribute(capacityAttribute);
    if (!storedCapacity || *storedCapacity != *logicalCapacity)
      return failWith(
          "retained active-transfer capacity differs from logical capacity");
  } else if (capacityAttribute) {
    return failWith("capacity was published despite malformed source capacity");
  }

  Attribute activeAttribute =
      function.getArgAttr(argumentIndex, kActiveTransferShapeAttr);
  if (proof.proven) {
    auto storedActive = parseShapeAttribute(activeAttribute);
    if (!storedActive || *storedActive != proof.activeShape)
      return failWith("active shape differs from the re-derived access box");
  } else if (activeAttribute) {
    return failWith("unknown proof must not carry an active shape");
  }
  return success();
}

bool hasStaticActiveTransferShapeFacts(func::FuncOp function,
                                       unsigned argumentIndex) {
  if (!function || argumentIndex >= function.getNumArguments())
    return false;
  return function.getArgAttr(argumentIndex, kActiveTransferShapeAttr) ||
         function.getArgAttr(argumentIndex, kActiveTransferCapacityShapeAttr) ||
         function.getArgAttr(argumentIndex, kActiveTransferProofAttr);
}

LogicalResult rederiveAndStoreStaticActiveTransferShapeProof(
    func::FuncOp function, unsigned argumentIndex, std::string *errorMessage) {
  auto failWith = [&](StringRef message) {
    if (errorMessage)
      *errorMessage = message.str();
    return failure();
  };
  if (!function || argumentIndex >= function.getNumArguments())
    return failWith("function argument index is invalid");
  if (!isa<MemRefType>(function.getArgument(argumentIndex).getType()))
    return failWith("function argument is not a memref");

  StaticActiveTransferShapeProof proof =
      analyzeStaticActiveTransferShape(function, argumentIndex);
  function.removeArgAttr(argumentIndex, kActiveTransferShapeAttr);
  function.removeArgAttr(argumentIndex, kActiveTransferCapacityShapeAttr);
  function.removeArgAttr(argumentIndex, kActiveTransferProofAttr);
  if (proof.hasCapacity)
    function.setArgAttr(
        argumentIndex, kActiveTransferCapacityShapeAttr,
        getShapeAttr(function.getContext(), proof.capacityShape));
  if (proof.proven)
    function.setArgAttr(argumentIndex, kActiveTransferShapeAttr,
                        getShapeAttr(function.getContext(), proof.activeShape));
  function.setArgAttr(argumentIndex, kActiveTransferProofAttr,
                      serializeStaticActiveTransferShapeProof(
                          function.getContext(), function, proof));
  return verifyStaticActiveTransferShapeProof(function, argumentIndex,
                                              errorMessage);
}

namespace {

static FailureOr<SmallVector<unsigned, 8>> parseArgumentList(StringRef text) {
  SmallVector<unsigned, 8> arguments;
  if (text.trim().empty())
    return failure();
  SmallVector<StringRef> pieces;
  text.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  for (StringRef piece : pieces) {
    unsigned index = 0;
    piece = piece.trim();
    if (piece.empty() || piece.getAsInteger(10, index) ||
        llvm::is_contained(arguments, index))
      return failure();
    arguments.push_back(index);
  }
  return arguments;
}

static llvm::json::Object
proofToJSON(func::FuncOp function,
            const StaticActiveTransferShapeProof &proof) {
  llvm::json::Object result;
  result["schema"] = kProofSchema.str();
  result["function"] = function.getSymName().str();
  result["argument"] = static_cast<int64_t>(proof.argumentIndex);
  result["status"] = proof.proven ? "proven" : "unknown";
  result["capacity_present"] = proof.hasCapacity;
  result["caller_noalias_proven"] =
      isa_and_nonnull<UnitAttr>(function->getAttr(kCallerNoAliasProvenAttr));
  llvm::json::Array noAliasArguments;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (!isa<MemRefType>(function.getArgument(index).getType()))
      continue;
    llvm::json::Object fact;
    fact["index"] = static_cast<int64_t>(index);
    fact["proven"] = hasNoAliasFact(function, index);
    noAliasArguments.push_back(std::move(fact));
  }
  result["noalias_arguments"] = std::move(noAliasArguments);
  llvm::json::Array capacity;
  for (int64_t extent : proof.capacityShape)
    capacity.push_back(extent);
  result["capacity_shape"] = std::move(capacity);
  if (proof.proven) {
    llvm::json::Array active;
    for (int64_t extent : proof.activeShape)
      active.push_back(extent);
    result["active_shape"] = std::move(active);
  }
  llvm::json::Array reasons;
  for (const std::string &reason : proof.reasons)
    reasons.push_back(reason);
  result["reasons"] = std::move(reasons);
  llvm::json::Array accesses;
  for (const ActiveTransferAccessRecord &access : proof.accesses) {
    llvm::json::Object accessObject;
    accessObject["ordinal"] = static_cast<int64_t>(access.ordinal);
    accessObject["task"] = access.taskName;
    accessObject["operation"] = access.operation;
    if (access.inputIndex)
      accessObject["input_index"] = static_cast<int64_t>(*access.inputIndex);
    accessObject["proven"] = access.proven;
    if (!access.reason.empty())
      accessObject["reason"] = access.reason;
    llvm::json::Array axes;
    for (const ActiveTransferAxisRange &axis : access.axes) {
      llvm::json::Object axisObject;
      axisObject["proven"] = axis.proven;
      axisObject["route"] = axis.route;
      axisObject["lower"] = axis.lower;
      axisObject["upper"] = axis.upper;
      if (axis.counterId)
        axisObject["counter_id"] = *axis.counterId;
      if (!axis.reason.empty())
        axisObject["reason"] = axis.reason;
      axes.push_back(std::move(axisObject));
    }
    accessObject["axes"] = std::move(axes);
    accesses.push_back(std::move(accessObject));
  }
  result["accesses"] = std::move(accesses);
  return result;
}

struct ProveStaticActiveTransferShapesPass
    : public PassWrapper<ProveStaticActiveTransferShapesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ProveStaticActiveTransferShapesPass)

  ProveStaticActiveTransferShapesPass() = default;
  ProveStaticActiveTransferShapesPass(
      const ProveStaticActiveTransferShapesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "prove-static-active-transfer-shapes";
  }
  StringRef getDescription() const override {
    return "Prove opt-in active transfer boxes from Taskflow and Neura "
           "accesses";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Function symbol to analyze."),
      llvm::cl::init("")};
  Option<std::string> argumentsText{
      *this, "arguments",
      llvm::cl::desc("Comma-separated function argument indices to prove."),
      llvm::cl::init("")};
  Option<std::string> reportOutput{
      *this, "report-output",
      llvm::cl::desc("Optional JSON proof report output path."),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (functionName.getValue().empty()) {
      module.emitError()
          << "prove-static-active-transfer-shapes requires a function name";
      return signalPassFailure();
    }
    FailureOr<SmallVector<unsigned, 8>> arguments =
        parseArgumentList(argumentsText.getValue());
    if (failed(arguments)) {
      module.emitError()
          << "arguments must be a non-empty comma-separated list "
             "of unique non-negative indices";
      return signalPassFailure();
    }
    func::FuncOp function = module.lookupSymbol<func::FuncOp>(functionName);
    if (!function) {
      module.emitError() << "function '" << functionName << "' was not found";
      return signalPassFailure();
    }
    if (function.isDeclaration() || function.getBody().empty()) {
      module.emitError() << "selected function must have a defined body";
      return signalPassFailure();
    }

    SmallVector<StaticActiveTransferShapeProof, 8> proofs;
    proofs.reserve(arguments->size());
    for (unsigned index : *arguments) {
      if (index >= function.getNumArguments()) {
        module.emitError() << "argument index " << index
                           << " is out of range for function '" << functionName
                           << "'";
        return signalPassFailure();
      }
      if (!isa<MemRefType>(function.getArgument(index).getType())) {
        module.emitError() << "argument " << index << " of function '"
                           << functionName << "' is not a memref";
        return signalPassFailure();
      }
      StaticActiveTransferShapeProof proof =
          analyzeStaticActiveTransferShape(function, index);
      bool hasAnyFact =
          function.getArgAttr(index, kActiveTransferShapeAttr) ||
          function.getArgAttr(index, kActiveTransferCapacityShapeAttr) ||
          function.getArgAttr(index, kActiveTransferProofAttr);
      if (hasAnyFact) {
        std::string error;
        if (failed(verifyStaticActiveTransferShapeProof(function, index,
                                                        &error))) {
          module.emitError() << "argument " << index
                             << " already carries a stale or unverified active "
                                "transfer fact: "
                             << error;
          return signalPassFailure();
        }
      }
      proofs.push_back(std::move(proof));
    }

    if (!reportOutput.getValue().empty()) {
      llvm::json::Array reports;
      for (const StaticActiveTransferShapeProof &proof : proofs)
        reports.push_back(proofToJSON(function, proof));
      llvm::json::Object document;
      document["schema"] = kProofSchema.str();
      document["proofs"] = std::move(reports);
      std::error_code error;
      llvm::raw_fd_ostream stream(reportOutput.getValue(), error,
                                  llvm::sys::fs::OF_Text);
      if (error) {
        module.emitError() << "cannot write active-transfer JSON report '"
                           << reportOutput << "': " << error.message();
        return signalPassFailure();
      }
      stream << llvm::json::Value(std::move(document)) << '\n';
      stream.flush();
      if (stream.has_error()) {
        module.emitError()
            << "failed while writing active-transfer JSON report '"
            << reportOutput << "'";
        return signalPassFailure();
      }
    }

    for (const StaticActiveTransferShapeProof &proof : proofs) {
      unsigned index = proof.argumentIndex;
      if (proof.hasCapacity)
        function.setArgAttr(
            index, kActiveTransferCapacityShapeAttr,
            getShapeAttr(function.getContext(), proof.capacityShape));
      if (proof.proven)
        function.setArgAttr(
            index, kActiveTransferShapeAttr,
            getShapeAttr(function.getContext(), proof.activeShape));
      // An unknown proof is required to have no active shape. Existing facts
      // were independently re-derived above, so there is nothing to clear.
      function.setArgAttr(index, kActiveTransferProofAttr,
                          serializeStaticActiveTransferShapeProof(
                              function.getContext(), function, proof));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createProveStaticActiveTransferShapesPass() {
  return std::make_unique<ProveStaticActiveTransferShapesPass>();
}

} // namespace mlir::amoeba::neura::joint_scheduling

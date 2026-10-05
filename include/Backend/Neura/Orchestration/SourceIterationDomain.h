//===- SourceIterationDomain.h ----------------------------------*- C++ -*-===//
//
// Source-owned iteration-domain certificates shared by Taskflow preparation
// and analytical cost metadata.  The certificate separates Taskflow counters
// (which candidate materialization may shard) from bounded loops retained in a
// kernel body (whose multiplicity remains part of each task firing).
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_H
#define AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_H

#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {
namespace joint_scheduling {

inline constexpr llvm::StringLiteral kSourceIterationDomainAttr =
    "amoeba.source_iteration_domain";
inline constexpr llvm::StringLiteral kSourceIterationControlBindingAttr =
    "amoeba.source_iteration_control_binding";
inline constexpr llvm::StringLiteral kSourceIterationSourceControlBindingAttr =
    "amoeba.source_iteration_source_control_binding";
inline constexpr llvm::StringLiteral kSourceIterationPartitionProofAttr =
    "amoeba.source_iteration_partition_verified";
inline constexpr llvm::StringLiteral kSourceIterationCapturePendingAttr =
    "amoeba.source_iteration_capture_pending";
inline constexpr llvm::StringLiteral kSourceIterationDomainSchema =
    "amoeba-source-iteration-domain-v1";

struct SourceIterationAxis {
  bool representedByTaskflow = false;
  bool expandedInsideMapperFiring = false;
  unsigned ordinal = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
  int64_t extent = 0;
  int64_t parentCounterOrdinal = -1;
  int64_t carriedValues = 0;
  int64_t resultUses = 0;
};

struct SourceIterationDomainInfo {
  bool complete = false;
  SmallVector<SourceIterationAxis> axes;
  int64_t representedMultiplicity = 1;
  int64_t internalMultiplicity = 1;
  int64_t sourceMultiplicity = 1;
  std::string reason;
};

inline bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  result = lhs * rhs;
  return true;
}

inline std::optional<int64_t>
sourceStaticIndex(Value value, taskflow::TaskflowTaskOp task,
                  llvm::DenseSet<Value> &active) {
  if (!value || !active.insert(value).second)
    return std::nullopt;
  auto eraseActive = llvm::make_scope_exit([&] { active.erase(value); });

  auto integerValue = [](IntegerAttr attr) -> std::optional<int64_t> {
    const llvm::APInt &value = attr.getValue();
    if (!value.isSignedIntN(64))
      return std::nullopt;
    return value.getSExtValue();
  };
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerValue(integer);
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return sourceStaticIndex(cast.getIn(), task, active);
  if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
    return sourceStaticIndex(cast.getIn(), task, active);

  auto checkedBinary = [&](Value lhs, Value rhs, char operation)
      -> std::optional<int64_t> {
    std::optional<int64_t> left = sourceStaticIndex(lhs, task, active);
    std::optional<int64_t> right = sourceStaticIndex(rhs, task, active);
    if (!left || !right)
      return std::nullopt;
    int64_t result = 0;
    if (operation == '+' && !__builtin_add_overflow(*left, *right, &result))
      return result;
    if (operation == '-' && !__builtin_sub_overflow(*left, *right, &result))
      return result;
    if (operation == '*' && !__builtin_mul_overflow(*left, *right, &result))
      return result;
    return std::nullopt;
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return checkedBinary(add.getLhs(), add.getRhs(), '+');
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    return checkedBinary(sub.getLhs(), sub.getRhs(), '-');
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return checkedBinary(mul.getLhs(), mul.getRhs(), '*');
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> constants;
    constants.reserve(apply->getNumOperands());
    for (Value operand : apply->getOperands()) {
      std::optional<int64_t> constant =
          sourceStaticIndex(operand, task, active);
      if (!constant)
        return std::nullopt;
      constants.push_back(IntegerAttr::get(
          IndexType::get(value.getContext()), *constant));
    }
    SmallVector<Attribute> folded;
    if (failed(apply.getAffineMap().constantFold(constants, folded)) ||
        folded.size() != 1)
      return std::nullopt;
    if (auto integer = dyn_cast<IntegerAttr>(folded.front()))
      return integerValue(integer);
    return std::nullopt;
  }

  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  if (argument.getOwner() == &task.getBody().front()) {
    SmallVector<Value> inputs(task.getWillReads());
    inputs.append(task.getWillWrites().begin(), task.getWillWrites().end());
    inputs.append(task.getValueInputs().begin(), task.getValueInputs().end());
    if (argument.getArgNumber() >= inputs.size())
      return std::nullopt;
    return sourceStaticIndex(inputs[argument.getArgNumber()], task, active);
  }
  if (auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp())) {
    std::string name = "amoeba.static_bound.arg." +
                       std::to_string(argument.getArgNumber());
    if (auto bound = function->getAttrOfType<IntegerAttr>(name))
      return bound.getInt() > 0 ? std::optional<int64_t>(bound.getInt())
                                : std::nullopt;
  }
  return std::nullopt;
}

inline std::optional<int64_t>
sourceStaticAffineBound(affine::AffineForOp loop, bool lower,
                        taskflow::TaskflowTaskOp task) {
  AffineMap map = lower ? loop.getLowerBoundMap() : loop.getUpperBoundMap();
  ValueRange operands = lower ? loop.getLowerBoundOperands()
                              : loop.getUpperBoundOperands();
  SmallVector<Attribute> constants;
  constants.reserve(operands.size());
  for (Value operand : operands) {
    llvm::DenseSet<Value> active;
    std::optional<int64_t> value = sourceStaticIndex(operand, task, active);
    if (!value)
      return std::nullopt;
    constants.push_back(
        IntegerAttr::get(IndexType::get(task.getContext()), *value));
  }
  SmallVector<Attribute> folded;
  if (failed(map.constantFold(constants, folded)) || folded.size() != 1)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(folded.front());
  if (!integer || !integer.getValue().isSignedIntN(64))
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

inline std::optional<int64_t>
sourceAffineTripCount(affine::AffineForOp loop, taskflow::TaskflowTaskOp task,
                      int64_t &lower, int64_t &upper, int64_t &step) {
  std::optional<int64_t> staticLower = sourceStaticAffineBound(loop, true, task);
  std::optional<int64_t> staticUpper = sourceStaticAffineBound(loop, false, task);
  if (!staticLower || !staticUpper || !loop.getStepAsInt() ||
      loop.getStepAsInt() <= 0 || *staticUpper <= *staticLower)
    return std::nullopt;
  lower = *staticLower;
  upper = *staticUpper;
  step = loop.getStepAsInt();
  if (lower < 0 && upper > std::numeric_limits<int64_t>::max() + lower)
    return std::nullopt;
  int64_t distance = upper - lower;
  return 1 + (distance - 1) / step;
}

inline SourceIterationDomainInfo
certifySourceIterationDomain(taskflow::TaskflowTaskOp task,
                             ArrayRef<affine::AffineForOp> representedLoops) {
  SourceIterationDomainInfo info;
  SmallVector<affine::AffineForOp> sourceLoops;
  task.walk([&](affine::AffineForOp loop) { sourceLoops.push_back(loop); });
  llvm::DenseSet<Operation *> represented;
  for (affine::AffineForOp loop : representedLoops)
    represented.insert(loop.getOperation());

  auto unsupported = [&](StringRef reason) {
    info.complete = false;
    info.reason = reason.str();
  };
  if (sourceLoops.empty()) {
    bool hasSequentialLoop = false;
    task.walk([&](scf::ForOp) { hasSequentialLoop = true; });
    task.walk([&](scf::WhileOp) { hasSequentialLoop = true; });
    if (hasSequentialLoop) {
      unsupported("task contains an SCF loop without source affine coverage");
      return info;
    }
    SmallVector<taskflow::TaskflowCounterOp> existingCounters;
    task.walk([&](taskflow::TaskflowCounterOp counter) {
      existingCounters.push_back(counter);
    });
    if (existingCounters.empty()) {
      info.complete = true;
      return info;
    }
    unsupported("Taskflow counters have no source affine-loop certificate");
    return info;
  }
  if (representedLoops.size() > sourceLoops.size()) {
    unsupported("represented counter axes exceed source affine loops");
    return info;
  }

  for (auto [ordinal, loop] : llvm::enumerate(representedLoops)) {
    SourceIterationAxis axis;
    axis.representedByTaskflow = true;
    axis.ordinal = ordinal;
    std::optional<int64_t> extent =
        sourceAffineTripCount(loop, task, axis.lower, axis.upper, axis.step);
    if (!extent) {
      unsupported("Taskflow source axis has a dynamic or invalid affine bound");
      return info;
    }
    axis.extent = *extent;
    int64_t product = 0;
    if (!checkedMultiply(info.representedMultiplicity, axis.extent, product)) {
      unsupported("source Taskflow-axis multiplicity exceeds int64");
      return info;
    }
    info.representedMultiplicity = product;
    info.axes.push_back(axis);
  }

  SmallVector<affine::AffineForOp> internalLoops;
  for (affine::AffineForOp loop : sourceLoops)
    if (!represented.contains(loop.getOperation()))
      internalLoops.push_back(loop);

  if (!internalLoops.empty()) {
    // The current certificate supports one bounded, loop-carried axis directly
    // nested under the last represented counter.  It records the state handoff
    // and requires every loop result to be consumed by parent-scope epilogue
    // work.  Other nest shapes remain executable but cannot produce a cost.
    if (internalLoops.size() != 1 || representedLoops.empty()) {
      unsupported("unrepresented source loops are not one direct nested axis");
      return info;
    }
    affine::AffineForOp inner = internalLoops.front();
    affine::AffineForOp parent = inner->getParentOfType<affine::AffineForOp>();
    if (!parent || parent != representedLoops.back() ||
        inner.getNumIterOperands() == 0 ||
        inner.getNumIterOperands() != inner.getNumResults()) {
      unsupported("unrepresented source loop lacks a certified carried-state parent");
      return info;
    }
    unsigned nestedAffineLoops = 0;
    inner.walk([&](affine::AffineForOp) { ++nestedAffineLoops; });
    if (nestedAffineLoops != 1) {
      unsupported("carried-state source loop contains a nested loop");
      return info;
    }
    bool hasNestedSCFLoop = false;
    inner.walk([&](scf::ForOp) { hasNestedSCFLoop = true; });
    inner.walk([&](scf::WhileOp) { hasNestedSCFLoop = true; });
    if (hasNestedSCFLoop) {
      unsupported("carried-state source loop contains an SCF loop");
      return info;
    }
    bool hasUnsupportedBranch = false;
    inner.walk([&](affine::AffineIfOp) { hasUnsupportedBranch = true; });
    inner.walk([&](scf::IfOp branch) {
      for (Region &region : branch->getRegions()) {
        if (!region.hasOneBlock()) {
          hasUnsupportedBranch = true;
          continue;
        }
        Block &block = region.front();
        auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
        if (!yield || yield.getNumOperands() != branch.getNumResults()) {
          hasUnsupportedBranch = true;
          continue;
        }
        for (Operation &operation : block.without_terminator())
          if (operation.getDialect()->getNamespace() != "arith")
            hasUnsupportedBranch = true;
      }
    });
    if (hasUnsupportedBranch) {
      unsupported("carried-state source loop contains a complex branch");
      return info;
    }

    // The initial carried values must be available in the parent's block
    // before the inner loop starts. This includes parent iter_args and values
    // produced by the per-outer-iteration prologue (the Ray Task_13 pattern).
    for (Value init : inner.getInits()) {
      if (auto argument = dyn_cast<BlockArgument>(init)) {
        if (argument.getOwner() != parent.getBody() &&
            argument.getOwner() != &task.getBody().front()) {
          unsupported("internal loop initial state is not parent-scoped");
          return info;
        }
        continue;
      }
      Operation *definition = init.getDefiningOp();
      if (!definition || definition->getBlock() != parent.getBody() ||
          !definition->isBeforeInBlock(inner)) {
        unsupported("internal loop initial state is not computed in its prologue");
        return info;
      }
    }

    SourceIterationAxis axis;
    axis.ordinal = 0;
    axis.parentCounterOrdinal = representedLoops.size() - 1;
    axis.carriedValues = inner.getNumIterOperands();
    std::optional<int64_t> extent =
        sourceAffineTripCount(inner, task, axis.lower, axis.upper, axis.step);
    if (!extent) {
      unsupported("internal source loop has a dynamic or invalid affine bound");
      return info;
    }
    axis.extent = *extent;
    for (Value result : inner.getResults()) {
      bool hasParentEpilogueUse = false;
      for (OpOperand &use : result.getUses()) {
        Operation *owner = use.getOwner();
        if (inner->isAncestor(owner))
          continue;
        if (owner->getBlock() != parent.getBody() ||
            !inner->isBeforeInBlock(owner)) {
          unsupported("internal reduction result escapes its parent epilogue");
          return info;
        }
        if (!isa<affine::AffineStoreOp, memref::StoreOp>(owner)) {
          unsupported("internal carried result is not consumed by a parent-scope store");
          return info;
        }
        hasParentEpilogueUse = true;
        ++axis.resultUses;
      }
      if (!hasParentEpilogueUse) {
        unsupported("internal reduction result has no parent-scope final use");
        return info;
      }
    }
    int64_t product = 0;
    if (!checkedMultiply(info.internalMultiplicity, axis.extent, product)) {
      unsupported("internal source-loop multiplicity exceeds int64");
      return info;
    }
    info.internalMultiplicity = product;
    info.axes.push_back(axis);
  }

  if (info.axes.size() != sourceLoops.size()) {
    unsupported("source affine-loop coverage is incomplete");
    return info;
  }

  // Affine-loop coverage alone is insufficient if another sequential loop or
  // a nested control loop is present in the task. Conditional selection in a
  // reduction body is allowed; loop constructs are not.
  bool hasUnsupportedLoop = false;
  task.walk([&](scf::ForOp) { hasUnsupportedLoop = true; });
  task.walk([&](scf::WhileOp) { hasUnsupportedLoop = true; });
  if (hasUnsupportedLoop) {
    unsupported("task contains an SCF loop outside the certified affine axes");
    return info;
  }
  if (!checkedMultiply(info.representedMultiplicity, info.internalMultiplicity,
                       info.sourceMultiplicity)) {
    unsupported("complete source iteration multiplicity exceeds int64");
    return info;
  }
  info.complete = true;
  return info;
}

inline std::string sourceIterationDomainCanonicalWitness(
    const SourceIterationDomainInfo &info) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << kSourceIterationDomainSchema << '\n'
         << "complete=" << (info.complete ? 1 : 0) << '\n'
         << "represented_multiplicity=" << info.representedMultiplicity
         << '\n'
         << "internal_multiplicity=" << info.internalMultiplicity << '\n'
         << "source_multiplicity=" << info.sourceMultiplicity << '\n'
         << "axis_count=" << info.axes.size() << '\n';
  for (const SourceIterationAxis &axis : info.axes)
    stream << "axis=" << (axis.representedByTaskflow ? 'T' : 'I') << ','
           << axis.ordinal << ',' << axis.lower << ',' << axis.upper << ','
           << axis.step << ',' << axis.extent << ','
           << (axis.expandedInsideMapperFiring ? 1 : 0) << ','
           << axis.parentCounterOrdinal << ',' << axis.carriedValues << ','
           << axis.resultUses << '\n';
  // Length-prefix the free-form reason so the witness remains unambiguous
  // even when diagnostic text contains punctuation or line breaks.
  stream << "reason_bytes=" << info.reason.size() << ':';
  stream.write(info.reason.data(), info.reason.size());
  stream << '\n';
  return text;
}

inline DictionaryAttr makeSourceIterationDomainAttr(
    Builder &builder, const SourceIterationDomainInfo &info) {
  SmallVector<Attribute> axes;
  axes.reserve(info.axes.size());
  for (const SourceIterationAxis &axis : info.axes) {
    axes.push_back(DictionaryAttr::get(
        builder.getContext(),
        {builder.getNamedAttr("kind", builder.getStringAttr(
                                            axis.representedByTaskflow
                                                ? "taskflow-counter"
                                                : "internal-carried-loop")),
         builder.getNamedAttr("expanded_inside_mapper_firing",
                              builder.getBoolAttr(
                                  axis.expandedInsideMapperFiring)),
         builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(axis.ordinal)),
         builder.getNamedAttr("lower", builder.getI64IntegerAttr(axis.lower)),
         builder.getNamedAttr("upper", builder.getI64IntegerAttr(axis.upper)),
         builder.getNamedAttr("step", builder.getI64IntegerAttr(axis.step)),
         builder.getNamedAttr("extent", builder.getI64IntegerAttr(axis.extent)),
         builder.getNamedAttr("parent_counter_ordinal",
                              builder.getI64IntegerAttr(
                                  axis.parentCounterOrdinal)),
         builder.getNamedAttr("carried_values",
                              builder.getI64IntegerAttr(axis.carriedValues)),
         builder.getNamedAttr("result_uses",
                              builder.getI64IntegerAttr(axis.resultUses))}));
  }
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("schema",
                           builder.getStringAttr(kSourceIterationDomainSchema)),
      builder.getNamedAttr("complete", builder.getBoolAttr(info.complete)),
      builder.getNamedAttr("represented_multiplicity",
                           builder.getI64IntegerAttr(info.representedMultiplicity)),
      builder.getNamedAttr("internal_multiplicity",
                           builder.getI64IntegerAttr(info.internalMultiplicity)),
      builder.getNamedAttr("source_multiplicity",
                           builder.getI64IntegerAttr(info.sourceMultiplicity)),
      builder.getNamedAttr("axes", builder.getArrayAttr(axes)),
      builder.getNamedAttr("canonical_witness",
                           builder.getStringAttr(
                               sourceIterationDomainCanonicalWitness(info)))};
  if (!info.reason.empty())
    fields.push_back(builder.getNamedAttr("reason",
                                         builder.getStringAttr(info.reason)));
  return DictionaryAttr::get(builder.getContext(), fields);
}

inline FailureOr<SourceIterationDomainInfo>
parseSourceIterationDomain(taskflow::TaskflowTaskOp task,
                           std::string &error) {
  auto attr = task->getAttrOfType<DictionaryAttr>(kSourceIterationDomainAttr);
  if (!attr) {
    error = "task " + task.getTaskName().str() +
            " has no source-owned iteration-domain certificate";
    return failure();
  }
  auto schema = attr.getAs<StringAttr>("schema");
  auto complete = attr.getAs<BoolAttr>("complete");
  auto represented = attr.getAs<IntegerAttr>("represented_multiplicity");
  auto internal = attr.getAs<IntegerAttr>("internal_multiplicity");
  auto source = attr.getAs<IntegerAttr>("source_multiplicity");
  auto axes = attr.getAs<ArrayAttr>("axes");
  auto canonicalWitness = attr.getAs<StringAttr>("canonical_witness");
  if (!schema || schema.getValue() != kSourceIterationDomainSchema ||
      !complete || !represented || !internal || !source || !axes ||
      !canonicalWitness ||
      represented.getInt() <= 0 || internal.getInt() <= 0 ||
      source.getInt() <= 0) {
    error = "task " + task.getTaskName().str() +
            " has a malformed source iteration-domain certificate";
    return failure();
  }
  SourceIterationDomainInfo info;
  info.complete = complete.getValue();
  info.representedMultiplicity = represented.getInt();
  info.internalMultiplicity = internal.getInt();
  info.sourceMultiplicity = source.getInt();
  if (auto reason = attr.getAs<StringAttr>("reason"))
    info.reason = reason.getValue().str();
  for (Attribute axisAttr : axes) {
    auto axisDict = dyn_cast<DictionaryAttr>(axisAttr);
    auto kind = axisDict ? axisDict.getAs<StringAttr>("kind") : StringAttr{};
    auto expanded = axisDict
                        ? axisDict.getAs<BoolAttr>(
                              "expanded_inside_mapper_firing")
                        : BoolAttr{};
    auto ordinal = axisDict ? axisDict.getAs<IntegerAttr>("ordinal") : IntegerAttr{};
    auto lower = axisDict ? axisDict.getAs<IntegerAttr>("lower") : IntegerAttr{};
    auto upper = axisDict ? axisDict.getAs<IntegerAttr>("upper") : IntegerAttr{};
    auto step = axisDict ? axisDict.getAs<IntegerAttr>("step") : IntegerAttr{};
    auto extent = axisDict ? axisDict.getAs<IntegerAttr>("extent") : IntegerAttr{};
    auto parent = axisDict
                      ? axisDict.getAs<IntegerAttr>("parent_counter_ordinal")
                      : IntegerAttr{};
    auto carried = axisDict ? axisDict.getAs<IntegerAttr>("carried_values")
                            : IntegerAttr{};
    auto uses = axisDict ? axisDict.getAs<IntegerAttr>("result_uses")
                         : IntegerAttr{};
    if (!kind || !expanded || !ordinal || !lower || !upper || !step ||
        !extent || !parent ||
        !carried || !uses || ordinal.getInt() < 0 || step.getInt() <= 0 ||
        extent.getInt() <= 0 || upper.getInt() <= lower.getInt() ||
        (kind.getValue() != "taskflow-counter" &&
         kind.getValue() != "internal-carried-loop")) {
      error = "task " + task.getTaskName().str() +
              " has a malformed source iteration-domain axis";
      return failure();
    }
    SourceIterationAxis parsed;
    parsed.representedByTaskflow = kind.getValue() == "taskflow-counter";
    parsed.expandedInsideMapperFiring = expanded.getValue();
    parsed.ordinal = ordinal.getInt();
    parsed.lower = lower.getInt();
    parsed.upper = upper.getInt();
    parsed.step = step.getInt();
    parsed.extent = extent.getInt();
    parsed.parentCounterOrdinal = parent.getInt();
    parsed.carriedValues = carried.getInt();
    parsed.resultUses = uses.getInt();
    if (parsed.lower < 0 &&
        parsed.upper > std::numeric_limits<int64_t>::max() + parsed.lower) {
      error = "task " + task.getTaskName().str() +
              " has an overflowing source iteration-domain extent";
      return failure();
    }
    int64_t distance = parsed.upper - parsed.lower;
    if (distance <= 0 || 1 + (distance - 1) / parsed.step != parsed.extent) {
      error = "task " + task.getTaskName().str() +
              " has an inconsistent source iteration-domain extent";
      return failure();
    }
    info.axes.push_back(parsed);
  }
  int64_t representedProduct = 1;
  int64_t internalProduct = 1;
  unsigned representedCount = 0;
  unsigned internalCount = 0;
  for (const SourceIterationAxis &axis : info.axes) {
    int64_t next = 0;
    if (axis.representedByTaskflow) {
      if (axis.ordinal != representedCount || axis.parentCounterOrdinal != -1 ||
          axis.expandedInsideMapperFiring || axis.carriedValues != 0 ||
          axis.resultUses != 0 ||
          !checkedMultiply(representedProduct, axis.extent, next)) {
        error = "task " + task.getTaskName().str() +
                " has inconsistent represented source-axis metadata";
        return failure();
      }
      representedProduct = next;
      ++representedCount;
    } else {
      if (axis.ordinal != internalCount || representedCount == 0 ||
          axis.parentCounterOrdinal !=
              static_cast<int64_t>(representedCount - 1) ||
          (info.complete && !axis.expandedInsideMapperFiring) ||
          axis.carriedValues <= 0 ||
          axis.resultUses <= 0 ||
          !checkedMultiply(internalProduct, axis.extent, next)) {
        error = "task " + task.getTaskName().str() +
                " has inconsistent internal source-axis metadata";
        return failure();
      }
      internalProduct = next;
      ++internalCount;
    }
  }
  int64_t sourceProduct = 0;
  if (representedProduct != info.representedMultiplicity ||
      internalProduct != info.internalMultiplicity ||
      !checkedMultiply(representedProduct, internalProduct, sourceProduct) ||
      sourceProduct != info.sourceMultiplicity || internalCount > 1) {
    error = "task " + task.getTaskName().str() +
            " has an inconsistent or unsupported source iteration product";
    return failure();
  }
  if (sourceIterationDomainCanonicalWitness(info) !=
      canonicalWitness.getValue()) {
    error = "task " + task.getTaskName().str() +
            " has a stale or inconsistent source iteration-domain witness";
    return failure();
  }
  if (!info.complete) {
    error = "task " + task.getTaskName().str() +
            " has incomplete source iteration-domain coverage: " +
            (info.reason.empty() ? "unknown control axis" : info.reason);
    return failure();
  }
  return info;
}

inline std::string
normalizeFalseDefaultMemoryHintsInBinding(StringRef binding) {
  std::string normalized = binding.str();
  constexpr llvm::StringLiteral operationNames[] = {"\"memref.load\"",
                                              "\"memref.store\""};
  constexpr llvm::StringLiteral defaultField = "nontemporal = false";
  for (StringRef operationName : operationNames) {
    size_t searchFrom = 0;
    while (true) {
      size_t operation = normalized.find(operationName.str(), searchFrom);
      if (operation == std::string::npos)
        break;
      // Do not interpret an escaped operation-looking string inside an
      // unrelated attribute as generic assembly syntax.
      size_t backslashes = 0;
      for (size_t index = operation; index > 0 && normalized[index - 1] == '\\';
           --index)
        ++backslashes;
      if (backslashes % 2 != 0) {
        searchFrom = operation + operationName.size();
        continue;
      }

      size_t typeSeparator =
          normalized.find(" : ", operation + operationName.size());
      if (typeSeparator == std::string::npos)
        break;
      size_t attributeOpen = normalized.rfind("<{", typeSeparator);
      if (attributeOpen == std::string::npos ||
          attributeOpen < operation + operationName.size()) {
        searchFrom = typeSeparator + 3;
        continue;
      }
      size_t attributeClose = normalized.find("}>", attributeOpen + 2);
      if (attributeClose == std::string::npos ||
          attributeClose + 2 > typeSeparator) {
        searchFrom = typeSeparator + 3;
        continue;
      }

      StringRef fields(normalized.data() + attributeOpen + 2,
                       attributeClose - attributeOpen - 2);
      size_t field = fields.find(defaultField);
      if (field == StringRef::npos ||
          fields.find("nontemporal", field + defaultField.size()) !=
              StringRef::npos) {
        searchFrom = typeSeparator + 3;
        continue;
      }
      const bool hasLeadingSeparator =
          field >= 2 && fields.substr(field - 2, 2) == ", ";
      size_t fieldEnd = field + defaultField.size();
      const bool hasTrailingSeparator =
          fields.substr(fieldEnd).starts_with(", ");
      if ((field != 0 && !hasLeadingSeparator) ||
          (fieldEnd != fields.size() && !hasTrailingSeparator)) {
        searchFrom = typeSeparator + 3;
        continue;
      }

      if (field == 0 && fieldEnd == fields.size()) {
        size_t eraseStart = attributeOpen;
        if (eraseStart > 0 && normalized[eraseStart - 1] == ' ')
          --eraseStart;
        normalized.erase(eraseStart, attributeClose + 2 - eraseStart);
        searchFrom = eraseStart + operationName.size();
        continue;
      }

      size_t fieldStart = hasLeadingSeparator ? field - 2 : field;
      size_t eraseEnd = !hasLeadingSeparator && hasTrailingSeparator
                            ? fieldEnd + 2
                            : fieldEnd;
      normalized.erase(attributeOpen + 2 + fieldStart, eraseEnd - fieldStart);
      searchFrom = attributeOpen + 2 + fieldStart;
    }
  }
  return normalized;
}

inline std::string currentSourceIterationControlBinding(
    taskflow::TaskflowTaskOp task, StringRef domainWitness) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << kSourceIterationDomainSchema << '\n'
         << "domain_witness_bytes=" << domainWitness.size() << ':'
         << domainWitness << '\n';
  // Bind all task control and body, including both Taskflow and Neura
  // counters. In particular, changing both mirrors' bounds cannot preserve
  // this exact witness. The certificate attributes are excluded to avoid a
  // recursive binding; all other task attributes and nested operations remain
  // part of the canonical generic-assembly witness.
  // Canonicalize external SSA references through a standalone argument list.
  // Printing a detached task with operands in another function can assign
  // different SSA names before/after a module round trip.
  SmallVector<Value> inputs;
  SmallVector<Type> inputTypes;
  llvm::DenseSet<Value> seenInputs;
  for (Value operand : task->getOperands()) {
    if (!seenInputs.insert(operand).second) continue;
    inputs.push_back(operand); inputTypes.push_back(operand.getType());
  }
  auto wrapper = func::FuncOp::create(task.getLoc(), "__source_domain_binding__",
      FunctionType::get(task.getContext(), inputTypes, {}));
  Block *entry = wrapper.addEntryBlock();
  IRMapping mapping;
  for (auto [source, argument] : llvm::zip(inputs, entry->getArguments()))
    mapping.map(source, argument);
  Operation *copy = task->clone(mapping);
  entry->push_back(copy);
  copy->removeAttr(kSourceIterationDomainAttr);
  copy->removeAttr(kSourceIterationControlBindingAttr);
  copy->removeAttr(kSourceIterationSourceControlBindingAttr);
  copy->removeAttr(kSourceIterationPartitionProofAttr);
  copy->removeAttr(kSourceIterationCapturePendingAttr);
  SmallVector<StringAttr> derived;
  for (NamedAttribute attr : copy->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name.starts_with("amoeba.mapper_") ||
        name.starts_with("amoeba.selected_") || name == "amoeba.exact_schedule" ||
        name == "amoeba.joint_shape_orientation_fixed" ||
        name == "compiled_ii" || name == "profile_info" ||
        name == "task_orchestration_info" || name == "cgra_count" ||
        name == "cgra_shape" || name == "replicas" || name == "tiling" ||
        name == "est_latency") derived.push_back(attr.getName());
  }
  for (StringAttr name : derived) copy->removeAttr(name);
  // Generic MLIR round-trips may materialize the default `nontemporal =
  // false` on loads or stores. Treat only that BoolAttr spelling as the same
  // canonical witness; true, malformed, and all other attributes remain
  // covered by the exact body binding.
  copy->walk([&](Operation *operation) {
    if (!isa<memref::LoadOp, memref::StoreOp>(operation))
      return;
    auto nontemporal = operation->getAttrOfType<BoolAttr>("nontemporal");
    if (nontemporal && !nontemporal.getValue())
      operation->removeAttr("nontemporal");
  });
  OpBuilder builder(task.getContext());
  builder.setInsertionPointToEnd(entry);
  builder.create<func::ReturnOp>(task.getLoc());
  wrapper.print(stream, OpPrintingFlags().printGenericOpForm().useLocalScope());
  wrapper->destroy();
  if (auto stored = task->getAttrOfType<StringAttr>(
          kSourceIterationControlBindingAttr)) {
    if (normalizeFalseDefaultMemoryHintsInBinding(stored.getValue()) ==
        normalizeFalseDefaultMemoryHintsInBinding(text))
      return stored.getValue().str();
  }
  return text;
}

} // namespace joint_scheduling
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_SOURCE_ITERATION_DOMAIN_H

//===- InheritReplicaAnalyticalTaskCostCatalogPass.cpp -------------------===//
//
// Carries an ML II/startup catalogue from a canonical task graph to a
// materialized replica graph after proving that every mapper-visible Neura
// body is structurally unchanged.  The catalogue remains per-iteration; the
// child trip count is used only for the derived ranker duration.
//
// This pass intentionally has no timeout or candidate restriction.  A
// missing proof, unsupported rewrite, stale identity, or incomplete shape
// coverage fails closed.
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "Backend/Neura/Orchestration/JointScheduling/TaskEdgeContract.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kInheritanceSchema =
    "orbit-replica-cost-inheritance-v1";
constexpr StringLiteral kFactoredSpaceSchema = "amoeba-analytical-task-space";
constexpr StringLiteral kNeighborhoodPartitionLineageAttr =
    "amoeba.neighborhood.partition_lineage.v1";

struct ShapeKey {
  std::string task;
  int64_t rows = 0;
  int64_t cols = 0;

  bool operator<(const ShapeKey &other) const {
    return std::tie(task, rows, cols) <
           std::tie(other.task, other.rows, other.cols);
  }
};

struct ChildFactor {
  std::string task;
  int64_t tripCount = 0;
  std::vector<ShapeKey> shapes;
};

struct ChildSpace {
  std::string function;
  std::string graphVariantId;
  std::string candidateCount;
  std::vector<ChildFactor> factors;
};

struct ReplicaRecord {
  TaskflowTaskOp child;
  std::string parent;
  // A source-owned M/N tiling may be followed by a generic replica.  Its
  // intermediate lineage is absent from the canonical parent module, so the
  // records are checked as one verified partition of the ultimate source
  // task rather than as a flat replica-id set.
  bool sourceOwnedTiling = false;
  int64_t id = -1;
  int64_t count = 0;
  int64_t shardAxis = -1;
  int64_t outputShardAxis = -1;
  int64_t childTripCount = 0;
  bool sequentialK = false;
  int64_t kLower = 0;
  int64_t kUpper = 0;
  int64_t mLower = 0;
  int64_t mUpper = 0;
  int64_t nLower = 0;
  int64_t nUpper = 0;
};

enum class CompositeFusionKind {
  Sibling,
  ProducerConsumerRetained,
  ProducerConsumerForwarded
};

struct CompositeFusionRecord {
  std::string childName;
  std::string firstParent;
  std::string secondParent;
  std::string materializerMode;
};

struct CounterBounds {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
};

struct BodyProof {
  bool equal = false;
  int64_t parentOperations = 0;
  int64_t childOperations = 0;
  int64_t parentKernels = 0;
  int64_t childKernels = 0;
  int64_t parentCounters = 0;
  int64_t childCounters = 0;
};

static std::optional<StringRef> requiredString(const json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value || value->empty())
    error = "missing or empty string field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> requiredInteger(const json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

static std::optional<bool> requiredBoolean(const json::Object &object,
                                           StringRef key,
                                           std::string &error) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value)
    error = "missing or invalid boolean field \"" + key.str() + "\"";
  return value;
}

static std::optional<std::string>
requiredDecimal(const json::Object &object, StringRef key, std::string &error) {
  if (std::optional<int64_t> numeric = object.getInteger(key)) {
    if (*numeric <= 0) {
      error = "invalid positive decimal field \"" + key.str() + "\"";
      return std::nullopt;
    }
    return std::to_string(*numeric);
  }
  std::optional<StringRef> text = object.getString(key);
  if (!text || text->empty() ||
      llvm::any_of(*text, [](char value) {
        return value < '0' || value > '9';
      })) {
    error = "missing or invalid positive decimal field \"" + key.str() +
            "\"";
    return std::nullopt;
  }
  StringRef normalized = text->ltrim('0');
  if (normalized.empty()) {
    error = "invalid positive decimal field \"" + key.str() + "\"";
    return std::nullopt;
  }
  return normalized.str();
}

static std::string multiplyDecimal(StringRef value, uint64_t factor) {
  if (factor == 0)
    return "0";
  std::string result;
  uint64_t carry = 0;
  for (auto it = value.rbegin(); it != value.rend(); ++it) {
    uint64_t digit = static_cast<uint64_t>(*it - '0');
    unsigned __int128 product = static_cast<unsigned __int128>(digit) * factor +
                                 carry;
    result.push_back(static_cast<char>('0' + product % 10));
    carry = static_cast<uint64_t>(product / 10);
  }
  while (carry) {
    result.push_back(static_cast<char>('0' + carry % 10));
    carry /= 10;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

static void putDecimal(json::Object &object, StringRef key, StringRef decimal) {
  uint64_t value = 0;
  if (!decimal.getAsInteger(10, value) &&
      value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    object[key] = static_cast<int64_t>(value);
  else
    object[key] = decimal.str();
}

// SHA-256 identities are deliberately outside the current contract.  Walk
// every object/array so a stale body_sha256 or graph_variant_sha256 cannot be
// silently carried into the inherited catalogue.
static bool rejectSha256(const json::Value &value, StringRef where,
                         std::string &error) {
  if (const json::Object *object = value.getAsObject()) {
    for (const auto &item : *object) {
      StringRef key = item.first;
      if (key.contains_insensitive("sha256")) {
        error = "SHA-256 identity is unsupported in " + where.str() +
                ": " + key.str();
        return false;
      }
      if (!rejectSha256(item.second, where, error))
        return false;
    }
    return true;
  }
  if (const json::Array *array = value.getAsArray()) {
    for (const json::Value &item : *array)
      if (!rejectSha256(item, where, error))
        return false;
  }
  return true;
}

static bool parseFactoredSpace(StringRef path, StringRef expectedFunction,
                               ChildSpace &space, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read child factored manifest " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  std::optional<json::Object> header;
  std::optional<json::Object> footer;
  for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
       !lines.is_at_end(); ++lines) {
    llvm::Expected<json::Value> parsed = json::parse(*lines);
    if (!parsed) {
      error = "invalid child factored manifest JSONL: " +
              llvm::toString(parsed.takeError());
      return false;
    }
    if (!rejectSha256(*parsed, "child factored manifest", error))
      return false;
    json::Object *object = parsed->getAsObject();
    if (!object) {
      error = "child factored manifest record is not an object";
      return false;
    }
    auto recordType = object->getString("record_type");
    if (!recordType) {
      error = "child factored manifest record has no record_type";
      return false;
    }
    if (*recordType == "space" && !header && !footer) {
      header = std::move(*object);
      continue;
    }
    if (*recordType == "footer" && header && !footer) {
      footer = std::move(*object);
      continue;
    }
    error = "child factored manifest must contain one space header and one "
            "footer";
    return false;
  }
  if (!header || !footer) {
    error = "child factored manifest is incomplete";
    return false;
  }
  auto schema = requiredString(*header, "schema", error);
  auto representation = requiredString(*header, "representation", error);
  auto exact = requiredBoolean(*header, "exact", error);
  auto function = requiredString(*header, "function", error);
  auto graph = requiredString(*header, "graph_variant_id", error);
  auto candidateCount = requiredDecimal(*header, "candidate_count", error);
  const json::Array *factors = header->getArray("factors");
  if (!schema || !representation || !exact || !function || !graph ||
      !candidateCount || !factors)
    return false;
  if (*schema != kFactoredSpaceSchema || *representation != "factored" ||
      !*exact || *function != expectedFunction || factors->empty()) {
    error = "child factored manifest schema/function/exactness mismatch";
    return false;
  }
  auto footerSchema = requiredString(*footer, "schema", error);
  auto footerRepresentation =
      requiredString(*footer, "representation", error);
  auto footerCount = requiredDecimal(*footer, "candidate_count", error);
  auto ranked = requiredBoolean(*footer, "ranked", error);
  if (!footerSchema || !footerRepresentation || !footerCount || !ranked)
    return false;
  if (*footerSchema != *schema || *footerRepresentation != *representation ||
      *footerCount != *candidateCount || *ranked) {
    error = "child factored manifest footer does not repeat its exact "
            "unranked candidate count";
    return false;
  }

  std::set<std::string> taskSet;
  std::set<ShapeKey> allShapes;
  for (const json::Value &factorValue : *factors) {
    const json::Object *factor = factorValue.getAsObject();
    if (!factor) {
      error = "child factored manifest factor is not an object";
      return false;
    }
    auto task = requiredString(*factor, "task", error);
    auto trip = requiredInteger(*factor, "trip_count", error);
    const json::Array *shapes = factor->getArray("shapes");
    if (!task || !trip || !shapes || *trip <= 0 || shapes->empty())
      return false;
    if (!taskSet.insert(task->str()).second) {
      error = "child factored manifest repeats task " + task->str();
      return false;
    }
    ChildFactor parsedFactor{task->str(), *trip, {}};
    std::set<std::pair<int64_t, int64_t>> shapeSet;
    for (const json::Value &shapeValue : *shapes) {
      const json::Object *shape = shapeValue.getAsObject();
      if (!shape) {
        error = "child factored manifest shape is not an object";
        return false;
      }
      auto rows = requiredInteger(*shape, "mapper_tile_rows", error);
      auto cols = requiredInteger(*shape, "mapper_tile_cols", error);
      auto kind = requiredString(*shape, "kind", error);
      if (!rows || !cols || !kind || *rows <= 0 || *cols <= 0 ||
          *kind != "rect")
        return false;
      if (!shapeSet.insert({*rows, *cols}).second) {
        error = "child factored manifest repeats task/mapper shape " +
                task->str();
        return false;
      }
      ShapeKey key{task->str(), *rows, *cols};
      parsedFactor.shapes.push_back(key);
      allShapes.insert(key);
    }
    space.factors.push_back(std::move(parsedFactor));
  }
  std::string computed = "1";
  for (const ChildFactor &factor : space.factors)
    computed = multiplyDecimal(computed, factor.shapes.size());
  if (computed != *candidateCount) {
    error = "child factored manifest candidate_count is not the exact "
            "Cartesian product of its factors";
    return false;
  }
  space.function = function->str();
  space.graphVariantId = graph->str();
  space.candidateCount = *candidateCount;
  return true;
}

static std::optional<unsigned> parseKernelInputReference(Attribute attribute) {
  auto text = dyn_cast_or_null<StringAttr>(attribute);
  if (!text)
    return std::nullopt;
  StringRef value = text.getValue();
  if (!value.consume_front("%input") || value.empty())
    return std::nullopt;
  unsigned index = 0;
  if (value.getAsInteger(10, index))
    return std::nullopt;
  return index;
}

// Resolve a compile-time index through the Taskflow value-input ABI.  A
// dynamic or opaque bound is deliberately rejected: K lineage is accepted
// only when both the Taskflow and Neura counters expose the same static range.
static std::optional<int64_t>
compileTimeIndex(Value value, TaskflowTaskOp task,
                 neura::KernelOp kernel = neura::KernelOp()) {
  if (!value)
    return std::nullopt;
  if (!kernel) {
    llvm::DenseSet<Value> active;
    if (auto folded = sourceStaticIndex(value, task, active)) return folded;
  }
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return compileTimeIndex(cast.getIn(), task, kernel);
  if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
    return compileTimeIndex(cast.getIn(), task, kernel);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (kernel && argument.getOwner() == &kernel.getBody().front()) {
      unsigned index = argument.getArgNumber();
      if (index >= kernel.getInputs().size())
        return std::nullopt;
      return compileTimeIndex(kernel.getInputs()[index], task);
    }
    if (argument.getOwner() == &task.getBody().front()) {
      unsigned firstValueInput = task.getWillReads().size() +
                                 task.getWillWrites().size();
      if (argument.getArgNumber() < firstValueInput)
        return std::nullopt;
      unsigned index = argument.getArgNumber() - firstValueInput;
      if (index >= task.getValueInputs().size())
        return std::nullopt;
      return compileTimeIndex(task.getValueInputs()[index], task);
    }
  }
  return std::nullopt;
}

static std::optional<int64_t>
compileTimeCounterAttribute(neura::CounterOp counter, StringRef attributeName,
                            TaskflowTaskOp task, neura::KernelOp kernel) {
  if (auto integer = counter->getAttrOfType<IntegerAttr>(attributeName))
    return integer.getInt();
  if (auto input = parseKernelInputReference(counter->getAttr(attributeName))) {
    if (*input >= kernel.getInputs().size())
      return std::nullopt;
    return compileTimeIndex(kernel.getInputs()[*input], task);
  }
  return std::nullopt;
}

static bool readRange(TaskflowTaskOp task, StringRef attributeName,
                      int64_t &lower, int64_t &upper, std::string &error) {
  auto range = task->getAttrOfType<DenseI64ArrayAttr>(attributeName);
  if (!range || range.size() != 2 || range[0] >= range[1]) {
    error = "sequential-K task " + task.getTaskName().str() +
            " has no valid " + attributeName.str();
    return false;
  }
  lower = range[0];
  upper = range[1];
  return true;
}

static bool readCounterBounds(TaskflowCounterOp counter,
                              TaskflowTaskOp task, CounterBounds &bounds,
                              std::string &error) {
  auto lower = compileTimeIndex(counter.getLowerBound(), task);
  auto upper = compileTimeIndex(counter.getUpperBound(), task);
  auto step = compileTimeIndex(counter.getStep(), task);
  if (!lower || !upper || !step || *upper <= *lower || *step <= 0) {
    error = "Taskflow counter bounds are dynamic or invalid in task " +
            task.getTaskName().str();
    return false;
  }
  bounds = {*lower, *upper, *step};
  return true;
}

static bool findTaskflowCounterBounds(TaskflowTaskOp task, int64_t id,
                                      CounterBounds &bounds,
                                      std::string &error) {
  TaskflowCounterOp found;
  bool duplicate = false;
  task.walk([&](TaskflowCounterOp counter) {
    auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!counterId || counterId.getInt() != id)
      return;
    if (found)
      duplicate = true;
    found = counter;
  });
  if (!found || duplicate) {
    error = "Taskflow counter_id " + std::to_string(id) +
            " is missing or duplicated in task " + task.getTaskName().str();
    return false;
  }
  return readCounterBounds(found, task, bounds, error);
}

static bool findSingleNeuraKernel(TaskflowTaskOp task, neura::KernelOp &kernel,
                                  std::string &error) {
  SmallVector<neura::KernelOp> kernels;
  task.walk([&](neura::KernelOp candidate) { kernels.push_back(candidate); });
  if (kernels.size() != 1) {
    error = "sequential-K body of task " + task.getTaskName().str() +
            " must contain exactly one mapper-visible Neura kernel";
    return false;
  }
  kernel = kernels.front();
  return true;
}

static bool findNeuraCounterBounds(TaskflowTaskOp task, int64_t id,
                                   CounterBounds &bounds,
                                   std::string &error) {
  neura::KernelOp kernel;
  if (!findSingleNeuraKernel(task, kernel, error))
    return false;
  neura::CounterOp found;
  bool duplicate = false;
  kernel.walk([&](neura::CounterOp counter) {
    auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!counterId || counterId.getInt() != id)
      return;
    if (found)
      duplicate = true;
    found = counter;
  });
  if (!found || duplicate) {
    error = "Neura counter_id " + std::to_string(id) +
            " is missing or duplicated in task " + task.getTaskName().str();
    return false;
  }
  auto readBound = [&](StringRef attributeName, Value operand,
                       std::optional<int64_t> &result) {
    std::optional<int64_t> fromOperand;
    if (operand) {
      fromOperand = compileTimeIndex(operand, task, kernel);
      if (!fromOperand) {
        error = "Neura counter " + std::to_string(id) + " has a dynamic " +
                attributeName.str() + " operand in task " +
                task.getTaskName().str();
        return false;
      }
    }
    std::optional<int64_t> fromAttribute = compileTimeCounterAttribute(
        found, attributeName, task, kernel);
    if (fromOperand && fromAttribute && *fromOperand != *fromAttribute) {
      error = "Neura counter " + std::to_string(id) + " " +
              attributeName.str() + " operand/attribute drift in task " +
              task.getTaskName().str();
      return false;
    }
    result = fromOperand ? fromOperand : fromAttribute;
    if (!result) {
      error = "Neura counter " + std::to_string(id) + " is missing " +
              attributeName.str() + " bounds in task " +
              task.getTaskName().str();
      return false;
    }
    return true;
  };
  std::optional<int64_t> lower;
  std::optional<int64_t> upper;
  std::optional<int64_t> step;
  if (!readBound("lower_bound_value", found.getLowerBound(), lower) ||
      !readBound("upper_bound_value", found.getUpperBound(), upper) ||
      !readBound("step_value", found.getStep(), step))
    return false;
  if (!lower || !upper || !step || *upper <= *lower || *step <= 0) {
    error = "Neura counter bounds are dynamic or invalid in task " +
            task.getTaskName().str();
    return false;
  }
  bounds = {*lower, *upper, *step};
  return true;
}

static bool equalBounds(const CounterBounds &lhs, const CounterBounds &rhs) {
  return lhs.lower == rhs.lower && lhs.upper == rhs.upper &&
         lhs.step == rhs.step;
}

static bool checkedTripProduct(const CounterBounds &m, const CounterBounds &n,
                               const CounterBounds &k, int64_t &tripCount) {
  if (m.step != 1 || n.step != 1 || k.step != 1 || m.upper <= m.lower ||
      n.upper <= n.lower || k.upper <= k.lower)
    return false;
  __int128 product = static_cast<__int128>(m.upper - m.lower) *
                     static_cast<__int128>(n.upper - n.lower) *
                     static_cast<__int128>(k.upper - k.lower);
  if (product <= 0 ||
      product > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
    return false;
  tripCount = static_cast<int64_t>(product);
  return true;
}

static bool isSequentialKTask(TaskflowTaskOp task) {
  auto source = task->getAttrOfType<StringAttr>("amoeba.semantic.k_source");
  auto policy = task->getAttrOfType<StringAttr>("amoeba.semantic.k_policy");
  auto parallel = task->getAttrOfType<BoolAttr>(
      "amoeba.semantic.k_parallel_legal");
  auto topology = task->getAttrOfType<StringAttr>(
      "amoeba.semantic.reduction_topology");
  return source && source.getValue() == "memory_feedback" && policy &&
         policy.getValue() == "sequential" && parallel && !parallel.getValue() &&
         topology && topology.getValue() == "none";
}

static bool isSequentialKDerivedTaskAttr(StringRef name) {
  return name == "amoeba.semantic.K_range" ||
         name == "amoeba.semantic.M_range" ||
         name == "amoeba.semantic.N_range" ||
         name == "amoeba.semantic.k_source" ||
         name == "amoeba.semantic.k_policy" ||
         name == "amoeba.semantic.k_parallel_legal" ||
         name == "amoeba.semantic.k_block_index" ||
         name == "amoeba.semantic.k_block_count" ||
         name == "amoeba.semantic.k_tiled" ||
         name == "amoeba.semantic.initial_C_ownership" ||
         name == "amoeba.semantic.final_C_ownership" ||
         name == "amoeba.semantic.output_visibility" ||
         name == "amoeba.semantic.reduction_topology" ||
         name == "amoeba.semantic.task_kind" ||
         name == "amoeba.semantic.task_role" ||
         name == "amoeba.semantic.schema" ||
         name == "amoeba.semantic.rewrite_schema";
}

static bool isIgnoredTaskAttr(StringRef name) {
  return name == "task_name" || name == "trip_count" ||
         name == "amoeba.selected_trip_count" ||
         name == kSourceIterationDomainAttr ||
         name == kSourceIterationControlBindingAttr ||
         name == kSourceIterationSourceControlBindingAttr ||
         name == kSourceIterationPartitionProofAttr ||
         name == kSourceIterationCapturePendingAttr ||
         name == kNeighborhoodPartitionLineageAttr ||
         name.starts_with("amoeba.replica.") ||
         name == "amoeba.semantic.incoming_edges" ||
         name == "amoeba.tiling.parent_task" ||
         name == "amoeba.tiling.output_region_lowers" ||
         name == "amoeba.tiling.output_region_uppers" ||
         name.starts_with("amoeba.tiling.input_region_");
}

static constexpr StringLiteral kOriginalAmoebaDecisionModeAttr =
    "amoeba.original_amoeba.fixed_decision_materialization";

// The nested composition path keeps the source-owned M/N tiling contract on
// the sibling tile and keeps the output rectangle on both that tile and its
// child replicas.  These metadata checks are deliberately local and strict;
// a task name that merely contains ".tile." is never sufficient lineage.
static bool hasOutputRegionProof(TaskflowTaskOp task, std::string &error) {
  auto lowers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_lowers");
  auto uppers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_uppers");
  if (!lowers || !uppers || lowers.empty() || lowers.size() != uppers.size()) {
    error = "source-owned nested partition is missing output-region proof";
    return false;
  }
  for (auto pair : llvm::zip(lowers, uppers)) {
    auto lower = dyn_cast<DenseI64ArrayAttr>(std::get<0>(pair));
    auto upper = dyn_cast<DenseI64ArrayAttr>(std::get<1>(pair));
    if (!lower || !upper || lower.size() == 0 ||
        lower.size() != upper.size()) {
      error = "source-owned nested partition has malformed output region";
      return false;
    }
    for (auto bounds : llvm::zip(lower.asArrayRef(), upper.asArrayRef()))
      if (std::get<0>(bounds) < 0 ||
          std::get<1>(bounds) <= std::get<0>(bounds)) {
        error = "source-owned nested partition has non-positive output "
                "region";
        return false;
      }
  }
  return true;
}

static bool collectCounterIds(TaskflowTaskOp task,
                              SmallVector<int64_t> &ids,
                              std::string &error) {
  std::set<int64_t> unique;
  bool malformed = false;
  task.walk([&](TaskflowCounterOp counter) {
    auto id = counter->getAttrOfType<IntegerAttr>("counter_id");
    if (!id || id.getInt() < 0 || !unique.insert(id.getInt()).second)
      malformed = true;
  });
  if (malformed || unique.empty()) {
    error = "ordinary replica task " + task.getTaskName().str() +
            " has missing or duplicate Taskflow counter IDs";
    return false;
  }
  for (int64_t id : unique)
    ids.push_back(id);
  for (auto [index, id] : llvm::enumerate(ids))
    if (id != static_cast<int64_t>(index)) {
      error = "ordinary replica task " + task.getTaskName().str() +
              " has non-contiguous Taskflow counter IDs";
      return false;
    }
  return true;
}

static bool checkedCounterVolume(ArrayRef<CounterBounds> bounds,
                                 int64_t &volume) {
  if (bounds.empty())
    return false;
  __int128 product = 1;
  for (const CounterBounds &bound : bounds) {
    if (bound.step != 1 || bound.upper <= bound.lower)
      return false;
    product *= static_cast<__int128>(bound.upper - bound.lower);
    if (product <= 0 ||
        product > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
      return false;
  }
  volume = static_cast<int64_t>(product);
  return true;
}

static bool validateOutputRegionAgainstProof(
    TaskflowTaskOp task, unsigned writeIndex,
    const ReplicaOutputCoordinateProof &proof, bool requireMetadata,
    std::string &error) {
  auto lowers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_lowers");
  auto uppers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_uppers");
  if (!lowers && !uppers) {
    if (requireMetadata) {
      error = "ordinary dual-output replica " + task.getTaskName().str() +
              " is missing output-region metadata";
      return false;
    }
    return true;
  }
  if (!lowers || !uppers || lowers.size() != task.getWillWrites().size() ||
      uppers.size() != lowers.size() || writeIndex >= lowers.size()) {
    error = "ordinary dual-output replica " + task.getTaskName().str() +
            " has incomplete output-region metadata";
    return false;
  }
  auto lower = dyn_cast<DenseI64ArrayAttr>(lowers[writeIndex]);
  auto upper = dyn_cast<DenseI64ArrayAttr>(uppers[writeIndex]);
  if (!lower || !upper ||
      static_cast<size_t>(lower.size()) != proof.outputCounterAxes.size() ||
      upper.size() != lower.size() ||
      proof.producedLowers.size() != proof.outputCounterAxes.size() ||
      proof.producedShape.size() != proof.outputCounterAxes.size()) {
    error = "ordinary dual-output replica " + task.getTaskName().str() +
            " has malformed output-region rank";
    return false;
  }
  SmallVector<int64_t> expectedUpper(proof.producedLowers.begin(),
                                     proof.producedLowers.end());
  for (auto [dimension, extent] : llvm::enumerate(proof.producedShape)) {
    int64_t start = proof.producedLowers[dimension];
    if (start < 0 || extent <= 0 ||
        start > std::numeric_limits<int64_t>::max() - extent) {
      error = "ordinary dual-output replica " + task.getTaskName().str() +
              " has an invalid proven output region";
      return false;
    }
    expectedUpper[dimension] = start + extent;
  }
  if (!llvm::equal(lower.asArrayRef(), proof.producedLowers) ||
      !llvm::equal(upper.asArrayRef(), expectedUpper)) {
    error = "ordinary dual-output replica " + task.getTaskName().str() +
            " output-region metadata disagrees with actual indexed "
            "coordinates";
    return false;
  }
  return true;
}

static bool validateReplicaOutputShardMap(TaskflowTaskOp parentTask,
                                          TaskflowTaskOp childTask,
                                          int64_t outputAxis,
                                          int64_t shardAxis,
                                          std::string &error) {
  if (outputAxis < 0 || shardAxis < 0 ||
      parentTask.getWillWrites().size() != childTask.getWillWrites().size() ||
      parentTask.getOriginalWriteMemrefs().size() !=
          parentTask.getWillWrites().size() ||
      childTask.getOriginalWriteMemrefs().size() !=
          childTask.getWillWrites().size()) {
    error = "ordinary replica " + childTask.getTaskName().str() +
            " has incompatible output-axis or write metadata";
    return false;
  }
  for (unsigned writeIndex = 0;
       writeIndex < childTask.getWillWrites().size(); ++writeIndex) {
    ReplicaOutputCoordinateProof parentProof =
        analyzeReplicaOutputCoordinates(parentTask, -1, writeIndex);
    ReplicaOutputCoordinateProof childProof =
        analyzeReplicaOutputCoordinates(childTask, outputAxis, writeIndex);
    if (!parentProof.proven || !childProof.proven) {
      error = "ordinary replica " + childTask.getTaskName().str() +
              " lacks an actual Taskflow/Neura output proof: " +
              (childProof.proven ? parentProof.reason : childProof.reason);
      return false;
    }
    if (parentProof.outputCounterAxes != childProof.outputCounterAxes ||
        outputAxis >= static_cast<int64_t>(
                          childProof.outputCounterAxes.size()) ||
        childProof.outputCounterAxes[outputAxis] ==
            ReplicaOutputCoordinateProof::kConstantAxis ||
        childProof.outputCounterAxes[outputAxis] !=
            static_cast<unsigned>(shardAxis)) {
      error = "ordinary replica " + childTask.getTaskName().str() +
              " output shard axis does not map to its actual counter axis";
      return false;
    }
  }
  return true;
}

static bool validateOrdinaryReplicaOutputs(const TaskMetadata &parent,
                                           const ReplicaRecord &record,
                                           int64_t outputAxis,
                                           std::string &error) {
  TaskflowTaskOp parentTask = parent.op;
  TaskflowTaskOp childTask = record.child;
  if (childTask.getWillWrites().size() <= 1)
    return true;
  if (parentTask.getWillWrites().size() != childTask.getWillWrites().size() ||
      parentTask.getOriginalWriteMemrefs().size() !=
          parentTask.getWillWrites().size() ||
      childTask.getOriginalWriteMemrefs().size() !=
          childTask.getWillWrites().size() ||
      outputAxis < 0) {
    error = "ordinary dual-output replica " + childTask.getTaskName().str() +
            " has incompatible output arity or shard axis";
    return false;
  }
  if (!childTask->hasAttr("amoeba.replica.output_counter_axes")) {
    error = "ordinary dual-output replica " + childTask.getTaskName().str() +
            " is missing the authenticated output counter map";
    return false;
  }
  for (unsigned writeIndex = 0;
       writeIndex < childTask.getWillWrites().size(); ++writeIndex) {
    ReplicaOutputCoordinateProof parentProof =
        analyzeReplicaOutputCoordinates(parentTask, -1, writeIndex);
    ReplicaOutputCoordinateProof childProof =
        analyzeReplicaOutputCoordinates(childTask, outputAxis, writeIndex);
    if (!parentProof.proven || !childProof.proven) {
      error = "ordinary dual-output replica " + childTask.getTaskName().str() +
              " lacks an actual Taskflow/Neura output proof for slot " +
              std::to_string(writeIndex) + ": " +
              (childProof.proven ? parentProof.reason : childProof.reason);
      return false;
    }
    if (parentProof.outputCounterAxes != childProof.outputCounterAxes) {
      error = "ordinary dual-output replica " + childTask.getTaskName().str() +
              " changed the actual output counter map for slot " +
              std::to_string(writeIndex);
      return false;
    }
    if (outputAxis >= static_cast<int64_t>(
                          childProof.outputCounterAxes.size()) ||
        childProof.outputCounterAxes[outputAxis] ==
            ReplicaOutputCoordinateProof::kConstantAxis ||
        childProof.outputCounterAxes[outputAxis] !=
            static_cast<unsigned>(record.shardAxis)) {
      error = "ordinary dual-output replica " + childTask.getTaskName().str() +
              " output shard axis does not map to its actual counter axis";
      return false;
    }
    if (!validateOutputRegionAgainstProof(
            parentTask, writeIndex, parentProof, /*requireMetadata=*/false,
            error) ||
        !validateOutputRegionAgainstProof(
            childTask, writeIndex, childProof, /*requireMetadata=*/true,
            error))
      return false;
  }
  return true;
}

static bool validateOrdinaryReplicaGroup(
    const TaskMetadata &parent, ArrayRef<ReplicaRecord> records,
    std::string &error) {
  if (records.empty())
    return false;
  SmallVector<int64_t> parentIds;
  if (!collectCounterIds(parent.op, parentIds, error))
    return false;
  SmallVector<CounterBounds> parentBounds;
  parentBounds.reserve(parentIds.size());
  for (int64_t id : parentIds) {
    CounterBounds taskBounds, neuraBounds;
    if (!findTaskflowCounterBounds(parent.op, id, taskBounds, error) ||
        !findNeuraCounterBounds(parent.op, id, neuraBounds, error) ||
        !equalBounds(taskBounds, neuraBounds)) {
      error = "canonical ordinary replica parent " + parent.name +
              " has Taskflow/Neura counter bound drift";
      return false;
    }
    parentBounds.push_back(taskBounds);
  }
  int64_t parentVolume = 0;
  if (!checkedCounterVolume(parentBounds, parentVolume) ||
      parent.tripCount != parentVolume) {
    error = "canonical ordinary replica parent " + parent.name +
            " trip count does not match actual counter volume";
    return false;
  }

  int64_t shardAxis = -1;
  int64_t outputAxis = -1;
  SmallVector<std::pair<int64_t, int64_t>> intervals;
  intervals.reserve(records.size());
  for (const ReplicaRecord &record : records) {
    TaskflowTaskOp child = record.child;
    SmallVector<int64_t> childIds;
    if (!collectCounterIds(child, childIds, error) ||
        childIds.size() != parentIds.size()) {
      error = "ordinary replica " + child.getTaskName().str() +
              " changed counter rank";
      return false;
    }
    SmallVector<CounterBounds> childBounds;
    childBounds.reserve(childIds.size());
    for (int64_t id : childIds) {
      CounterBounds taskBounds, neuraBounds;
      if (!findTaskflowCounterBounds(child, id, taskBounds, error) ||
          !findNeuraCounterBounds(child, id, neuraBounds, error) ||
          !equalBounds(taskBounds, neuraBounds)) {
        error = "ordinary replica " + child.getTaskName().str() +
                " has Taskflow/Neura counter bound drift";
        return false;
      }
      childBounds.push_back(taskBounds);
    }
    auto axis = child->getAttrOfType<IntegerAttr>("amoeba.replica.shard_axis");
    if (!axis || axis.getInt() < 0 ||
        axis.getInt() >= static_cast<int64_t>(parentBounds.size())) {
      error = "ordinary replica " + child.getTaskName().str() +
              " has no valid shard axis";
      return false;
    }
    if (shardAxis < 0)
      shardAxis = axis.getInt();
    if (shardAxis != axis.getInt()) {
      error = "ordinary replica group changes shard axis";
      return false;
    }
    for (int64_t id = 0; id < static_cast<int64_t>(parentBounds.size());
         ++id) {
      if (id == shardAxis) {
        if (childBounds[id].lower < parentBounds[id].lower ||
            childBounds[id].upper > parentBounds[id].upper) {
          error = "ordinary replica " + child.getTaskName().str() +
                  " shard bounds are outside the canonical parent";
          return false;
        }
      } else if (!equalBounds(childBounds[id], parentBounds[id])) {
        error = "ordinary replica " + child.getTaskName().str() +
                " changed an unsharded counter bound";
        return false;
      }
    }
    int64_t childVolume = 0;
    if (!checkedCounterVolume(childBounds, childVolume) ||
        record.childTripCount != childVolume) {
      error = "ordinary replica " + child.getTaskName().str() +
              " trip count does not match actual counter volume";
      return false;
    }
    auto shardLower = child->getAttrOfType<IntegerAttr>(
        "amoeba.replica.shard_lower");
    auto shardUpper = child->getAttrOfType<IntegerAttr>(
        "amoeba.replica.shard_upper");
    if (!shardLower || shardLower.getInt() != childBounds[shardAxis].lower ||
        !shardUpper || shardUpper.getInt() != childBounds[shardAxis].upper) {
      error = "ordinary replica " + child.getTaskName().str() +
              " shard metadata disagrees with actual counter bounds";
      return false;
    }
    if (auto declared = child->getAttrOfType<IntegerAttr>(
            "amoeba.replica.shard_trip_count");
        !declared || declared.getInt() != record.childTripCount) {
      error = "ordinary replica " + child.getTaskName().str() +
              " shard trip count disagrees with actual counter volume";
      return false;
    }
    if (auto declared = child->getAttrOfType<IntegerAttr>(
            "amoeba.replica.total_trip_count");
        declared && declared.getInt() != parent.tripCount) {
      error = "ordinary replica " + child.getTaskName().str() +
              " total trip count disagrees with canonical parent";
      return false;
    }
    intervals.push_back(
        {childBounds[shardAxis].lower, childBounds[shardAxis].upper});
    auto childOutputAxis = child->getAttrOfType<IntegerAttr>(
        "amoeba.replica.output_shard_axis");
    if (child->hasAttr("amoeba.replica.output_shard_axis") &&
        !childOutputAxis) {
      error = "ordinary replica " + child.getTaskName().str() +
              " has malformed output shard axis";
      return false;
    }
    if (childOutputAxis) {
      if (outputAxis < 0)
        outputAxis = childOutputAxis.getInt();
      if (outputAxis != childOutputAxis.getInt()) {
        error = "ordinary replica group changes output shard axis";
        return false;
      }
      if (child.getWillWrites().size() == 1 &&
          !validateReplicaOutputShardMap(parent.op, child,
                                         childOutputAxis.getInt(), shardAxis,
                                         error))
        return false;
    }
    if (child.getWillWrites().size() > 1) {
      if (!childOutputAxis || childOutputAxis.getInt() < 0) {
        error = "ordinary dual-output replica " + child.getTaskName().str() +
                " has no valid output shard axis";
        return false;
      }
      if (outputAxis < 0)
        outputAxis = childOutputAxis.getInt();
      if (outputAxis != childOutputAxis.getInt() ||
          !validateOrdinaryReplicaOutputs(parent, record, outputAxis, error))
        return false;
    }
  }
  llvm::sort(intervals);
  int64_t cursor = parentBounds[shardAxis].lower;
  for (auto [lower, upper] : intervals) {
    if (lower != cursor || upper <= lower) {
      error = "ordinary replica shard intervals overlap or leave a gap";
      return false;
    }
    cursor = upper;
  }
  if (cursor != parentBounds[shardAxis].upper) {
    error = "ordinary replica shard intervals do not cover the parent";
    return false;
  }
  return true;
}

static bool validateSourceOwnedTilingTask(TaskflowTaskOp task,
                                          std::string &error) {
  auto parent = task->getAttrOfType<StringAttr>(
      "amoeba.neura.tiling.parent_task");
  auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
  auto part =
      task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.part_index");
  auto factor =
      task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
  auto original = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.original_range");
  auto derived = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.derived_range");
  auto rewrite = task->getAttrOfType<StringAttr>(
      "amoeba.neura.joint_rewrite");
  if (!parent || parent.getValue().empty() || !axis || !part || !factor ||
      !original || !derived || !rewrite ||
      rewrite.getValue() != "post-neura-mn-tiling" || axis.getInt() < 0 ||
      part.getInt() < 0 || factor.getInt() < 2 ||
      part.getInt() >= factor.getInt() || original.size() != 2 ||
      derived.size() != 2 || derived[0] >= derived[1] ||
      original[0] >= original[1] || derived[0] < original[0] ||
      derived[1] > original[1]) {
    error = "source-owned nested partition has incomplete M/N tiling "
            "metadata";
    return false;
  }
  int64_t extent = original[1] - original[0];
  int64_t baseWidth = extent / factor.getInt();
  int64_t remainder = extent % factor.getInt();
  int64_t expectedLower =
      original[0] + baseWidth * part.getInt() +
      std::min<int64_t>(part.getInt(), remainder);
  int64_t expectedWidth =
      baseWidth + (part.getInt() < remainder ? int64_t(1) : int64_t(0));
  if (axis.getInt() > 1 || extent <= 0 || baseWidth <= 0 ||
      derived[0] != expectedLower ||
      derived[1] != expectedLower + expectedWidth) {
    error = "source-owned nested partition has inconsistent M/N tile range";
    return false;
  }
  return hasOutputRegionProof(task, error);
}

// A source-owned M/N tile is accepted by inheritance only after its actual
// counter boxes have been checked.  Metadata and the child trip count are
// witnesses, not a partition proof: a forged pair of overlapping rectangles
// can preserve both while dropping work.  Keep this check local to the
// source-owned nested path; ordinary replica and sequential-K paths retain
// their established validators below.
struct SourceOwnedPartitionBox {
  SmallVector<CounterBounds, 3> bounds;
  const ReplicaRecord *record = nullptr;
};

static bool decodeLineageCounterBounds(Attribute attribute,
                                       MutableArrayRef<CounterBounds> bounds) {
  auto encoded = dyn_cast_or_null<DenseI64ArrayAttr>(attribute);
  if (!encoded || bounds.empty() || bounds.size() > 3 ||
      encoded.size() != static_cast<int64_t>(bounds.size() * 3))
    return false;
  for (size_t axis = 0; axis < bounds.size(); ++axis) {
    bounds[axis] = {encoded[axis * 3], encoded[axis * 3 + 1],
                    encoded[axis * 3 + 2]};
    if (bounds[axis].lower < 0 || bounds[axis].upper <= bounds[axis].lower ||
        bounds[axis].step != 1)
      return false;
  }
  return true;
}

// The serialized root is only a lookup hint for a candidate whose immediate
// parent has already been replaced.  Callers must still prove the complete
// interval chain and bind its final step to the task's typed counters and
// lineage attributes before using this name as a canonical parent.
static std::optional<std::string>
sourceOwnedPartitionRootHint(TaskflowTaskOp task) {
  auto roots = dyn_cast_or_null<ArrayAttr>(
      task->getAttr(kNeighborhoodPartitionLineageAttr));
  if (!roots || roots.size() != 1)
    return std::nullopt;
  auto root = dyn_cast<DictionaryAttr>(roots[0]);
  auto name = root ? root.getAs<StringAttr>("root") : StringAttr{};
  if (!name || name.getValue().empty())
    return std::nullopt;
  return name.getValue().str();
}

static bool lineagePartitionMatches(StringRef family, int64_t axis,
                                    int64_t factor, int64_t part,
                                    ArrayRef<CounterBounds> before,
                                    ArrayRef<CounterBounds> after) {
  if ((family != "tiling" && family != "replica") || before.size() < 2 ||
      before.size() > 3 || before.size() != after.size() || axis < 0 ||
      axis > 1 || factor < 2 || factor > 8 || part < 0 || part >= factor)
    return false;
  for (size_t dimension = 0; dimension < before.size(); ++dimension) {
    const CounterBounds &source = before[dimension];
    if (source.lower < 0 || source.upper <= source.lower || source.step != 1)
      return false;
    if (static_cast<int64_t>(dimension) != axis) {
      if (!equalBounds(source, after[dimension]))
        return false;
      continue;
    }
    int64_t extent = source.upper - source.lower;
    if (extent < factor)
      return false;
    int64_t quotient = extent / factor;
    int64_t remainder = extent % factor;
    int64_t lower = source.lower + quotient * part + std::min(part, remainder);
    int64_t upper = lower + quotient + (part < remainder ? 1 : 0);
    if (after[dimension].lower != lower || after[dimension].upper != upper ||
        after[dimension].step != 1)
      return false;
  }
  return true;
}

// Authenticate a nested tile's immediate parent interval from the replayed
// source-owned partition chain. The chain is checked from the canonical root
// through every actual interval split and is finally tied to the task's
// current Taskflow/Neura bounds, task name, and tiling attributes.
static bool sourceOwnedImmediateTilingBounds(
    TaskflowTaskOp task, StringRef canonicalName,
    ArrayRef<CounterBounds> canonicalBounds,
    ArrayRef<CounterBounds> current,
    SmallVectorImpl<CounterBounds> &immediateBounds,
    bool &finalStepIsTiling, std::string &error) {
  finalStepIsTiling = false;
  const size_t rank = canonicalBounds.size();
  if (rank < 2 || rank > 3 || current.size() != rank) {
    error = "source-owned tiling has an unsupported counter rank";
    return false;
  }
  Attribute ledgerAttribute = task->getAttr(kNeighborhoodPartitionLineageAttr);
  if (!ledgerAttribute) {
    if (rank == 3) {
      error = "source-owned rank-three partition requires an authenticated "
              "full-rank lineage ledger";
      return false;
    }
    auto declaredParent =
        task->getAttrOfType<StringAttr>("amoeba.neura.tiling.parent_task");
    auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
    auto factor = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
    auto part = task->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.part_index");
    if (!declaredParent || declaredParent.getValue() != canonicalName) {
      error = "nested source-owned tiling requires authenticated partition "
              "lineage";
      return false;
    }
    auto originalRange = task->getAttrOfType<DenseI64ArrayAttr>(
        "amoeba.neura.tiling.original_range");
    auto derivedRange = task->getAttrOfType<DenseI64ArrayAttr>(
        "amoeba.neura.tiling.derived_range");
    if (!axis || !factor || !part || axis.getInt() < 0 || axis.getInt() > 1 ||
        !originalRange ||
        !derivedRange || originalRange.size() != 2 ||
        derivedRange.size() != 2 ||
        originalRange[0] != canonicalBounds[axis.getInt()].lower ||
        originalRange[1] != canonicalBounds[axis.getInt()].upper ||
        derivedRange[0] != current[axis.getInt()].lower ||
        derivedRange[1] != current[axis.getInt()].upper ||
        (rank == 3 && !equalBounds(current[2], canonicalBounds[2])) ||
        !lineagePartitionMatches("tiling", axis.getInt(),
                                 factor.getInt(), part.getInt(),
                                 canonicalBounds, current)) {
      error = "source-owned tiling range attributes disagree with canonical "
              "or current counter bounds";
      return false;
    }
    immediateBounds.assign(canonicalBounds.begin(), canonicalBounds.end());
    finalStepIsTiling = true;
    return true;
  }

  auto roots = dyn_cast<ArrayAttr>(ledgerAttribute);
  if (!roots || roots.size() != 1) {
    error = "source-owned tiling has malformed or multi-root partition lineage";
    return false;
  }
  auto root = dyn_cast<DictionaryAttr>(roots[0]);
  auto rootName = root ? root.getAs<StringAttr>("root") : StringAttr{};
  auto steps = root ? root.getAs<ArrayAttr>("steps") : ArrayAttr{};
  SmallVector<CounterBounds, 3> original(rank);
  if (!rootName || rootName.getValue() != canonicalName || !steps ||
      steps.empty() ||
      !decodeLineageCounterBounds(root.get("original"), original) ||
      !llvm::equal(original, canonicalBounds, equalBounds)) {
    error = "source-owned tiling partition lineage does not match its "
            "canonical root";
    return false;
  }

  SmallVector<CounterBounds, 3> domain(original.begin(), original.end());
  SmallVector<CounterBounds, 3> lastBefore;
  struct StepContract {
    bool present = false;
    std::string parent;
    int64_t axis = -1;
    int64_t factor = -1;
    int64_t part = -1;
    SmallVector<CounterBounds, 3> before;
    SmallVector<CounterBounds, 3> after;
  } latestTiling, latestReplica;
  std::string derivedName = canonicalName.str();
  std::string immediateName;
  std::string lastFamily;
  for (Attribute stepAttribute : steps) {
    auto step = dyn_cast<DictionaryAttr>(stepAttribute);
    auto family = step ? step.getAs<StringAttr>("family") : StringAttr{};
    auto axis = step ? step.getAs<IntegerAttr>("axis") : IntegerAttr{};
    auto factor = step ? step.getAs<IntegerAttr>("factor") : IntegerAttr{};
    auto part = step ? step.getAs<IntegerAttr>("part") : IntegerAttr{};
    SmallVector<CounterBounds, 3> before(rank);
    SmallVector<CounterBounds, 3> after(rank);
    if (!family || !axis || !factor || !part || !axis.getType().isInteger(64) ||
        !factor.getType().isInteger(64) || !part.getType().isInteger(64) ||
        !decodeLineageCounterBounds(step.get("before"), before) ||
        !decodeLineageCounterBounds(step.get("after"), after) ||
        !llvm::equal(domain, before, equalBounds) ||
        !lineagePartitionMatches(family.getValue(), axis.getInt(),
                                 factor.getInt(), part.getInt(), before,
                                 after)) {
      error = "source-owned tiling partition lineage has a broken interval "
              "chain";
      return false;
    }
    immediateName = derivedName;
    StepContract &familyContract = family.getValue() == "tiling"
                                       ? latestTiling
                                       : latestReplica;
    familyContract.present = true;
    familyContract.parent = derivedName;
    familyContract.axis = axis.getInt();
    familyContract.factor = factor.getInt();
    familyContract.part = part.getInt();
    familyContract.before.assign(before.begin(), before.end());
    familyContract.after.assign(after.begin(), after.end());
    if (family.getValue() == "tiling") {
      derivedName +=
          (Twine(".tile.") + Twine(axis.getInt()) + "." + Twine(part.getInt()))
              .str();
    } else {
      derivedName += (Twine(".replica.") + Twine(part.getInt())).str();
    }
    lastBefore.assign(before.begin(), before.end());
    domain.assign(after.begin(), after.end());
    lastFamily = family.getValue().str();
  }
  finalStepIsTiling = lastFamily == "tiling";
  auto finalParent = task->getAttrOfType<StringAttr>(
      finalStepIsTiling ? "amoeba.neura.tiling.parent_task"
                        : "amoeba.replica.parent_task");
  if (derivedName != task.getTaskName() || !finalParent ||
      immediateName != finalParent.getValue() ||
      !llvm::equal(domain, current, equalBounds)) {
    error = "source-owned tiling partition lineage disagrees with its "
            "immediate parent or current bounds";
    return false;
  }

  auto hasTilingMetadata = task->hasAttr("amoeba.neura.tiling.parent_task") ||
                           task->hasAttr("amoeba.neura.tiling.axis") ||
                           task->hasAttr("amoeba.neura.tiling.factor") ||
                           task->hasAttr("amoeba.neura.tiling.part_index") ||
                           task->hasAttr("amoeba.neura.tiling.original_range") ||
                           task->hasAttr("amoeba.neura.tiling.derived_range");
  // Replica materialization strips every amoeba.neura.tiling.* attribute
  // from its output tasks. When the authenticated ledger ends in a replica,
  // that complete absence is valid because the ledger and replica counters
  // still prove the tile boundary. Partial metadata is never accepted.
  bool replicaMaterializerStrippedTileMetadata =
      latestTiling.present && lastFamily == "replica" && !hasTilingMetadata;
  if ((!latestTiling.present && hasTilingMetadata) ||
      (latestTiling.present && !hasTilingMetadata &&
       !replicaMaterializerStrippedTileMetadata)) {
    error = "source-owned tiling attributes do not match their latest "
            "authenticated lineage step";
    return false;
  }
  if (latestTiling.present && hasTilingMetadata) {
    auto parent = task->getAttrOfType<StringAttr>(
        "amoeba.neura.tiling.parent_task");
    auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
    auto factor = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
    auto part = task->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.part_index");
    auto originalRange = task->getAttrOfType<DenseI64ArrayAttr>(
        "amoeba.neura.tiling.original_range");
    auto derivedRange = task->getAttrOfType<DenseI64ArrayAttr>(
        "amoeba.neura.tiling.derived_range");
    int64_t axisIndex = latestTiling.axis;
    if (!parent || parent.getValue() != latestTiling.parent || !axis ||
        !factor || !part || !originalRange || !derivedRange ||
        axis.getInt() != axisIndex || factor.getInt() != latestTiling.factor ||
        part.getInt() != latestTiling.part || originalRange.size() != 2 ||
        derivedRange.size() != 2 ||
        originalRange[0] != latestTiling.before[axisIndex].lower ||
        originalRange[1] != latestTiling.before[axisIndex].upper ||
        derivedRange[0] != latestTiling.after[axisIndex].lower ||
        derivedRange[1] != latestTiling.after[axisIndex].upper) {
      error = "source-owned tiling attributes do not match their latest "
              "authenticated lineage step";
      return false;
    }
    immediateBounds.assign(latestTiling.before.begin(),
                           latestTiling.before.end());
  } else {
    immediateBounds.assign(lastBefore.begin(), lastBefore.end());
  }

  auto hasReplicaMetadata = task->hasAttr("amoeba.replica.parent_task") ||
                            task->hasAttr("amoeba.replica.id") ||
                            task->hasAttr("amoeba.replica.count") ||
                            task->hasAttr("amoeba.replica.shard_axis") ||
                            task->hasAttr("amoeba.replica.shard_lower") ||
                            task->hasAttr("amoeba.replica.shard_upper");
  if (latestReplica.present != hasReplicaMetadata) {
    error = "source-owned replica attributes do not match their latest "
            "authenticated lineage step";
    return false;
  }
  if (latestReplica.present) {
    auto parent = task->getAttrOfType<StringAttr>(
        "amoeba.replica.parent_task");
    auto id = task->getAttrOfType<IntegerAttr>("amoeba.replica.id");
    auto count = task->getAttrOfType<IntegerAttr>("amoeba.replica.count");
    auto axis = task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_axis");
    auto lower = task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_lower");
    auto upper = task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_upper");
    auto shardTripCount = task->getAttrOfType<IntegerAttr>(
        "amoeba.replica.shard_trip_count");
    int64_t axisIndex = latestReplica.axis;
    if (!parent || parent.getValue() != latestReplica.parent || !id || !count ||
        !axis || !lower || !upper || id.getInt() != latestReplica.part ||
        count.getInt() != latestReplica.factor ||
        axis.getInt() != axisIndex ||
        lower.getInt() != latestReplica.after[axisIndex].lower ||
        upper.getInt() != latestReplica.after[axisIndex].upper) {
      error = "source-owned replica attributes do not match their latest "
              "authenticated lineage step";
      return false;
    }
    int64_t historicalReplicaTripCount = 0;
    if (!shardTripCount || !shardTripCount.getType().isInteger(64) ||
        !checkedCounterVolume(latestReplica.after,
                              historicalReplicaTripCount) ||
        shardTripCount.getInt() != historicalReplicaTripCount) {
      error = "source-owned replica shard trip count disagrees with "
              "authenticated replica bounds";
      return false;
    }
    // If no later tile narrowed this replica, the historical replica volume
    // is also the current task volume.  A replica followed by a tile keeps
    // the replica's original shard_trip_count while its task trip count
    // reflects only the smaller current box.
    if (lastFamily == "replica") {
      int64_t currentTripCount = 0;
      if (!checkedCounterVolume(current, currentTripCount) ||
          shardTripCount.getInt() != currentTripCount) {
        error = "source-owned final replica shard trip count disagrees with "
                "current counter volume";
        return false;
      }
    }
  }
  return true;
}

// Indexed output coordinates are authenticated from the actual Neura DFG.
// A copied region attribute is only a witness: forwarding through predication
// or a phi is accepted when every incoming value names the same counter, while
// arithmetic or an opaque value is rejected.
static std::optional<unsigned>
sourceOwnedOutputCounterAxis(Value value, ArrayRef<neura::CounterOp> counters,
                             unsigned depth = 0) {
  for (auto [axis, counterRef] : llvm::enumerate(counters)) {
    neura::CounterOp counter = counterRef;
    if (value == counter.getCurrentIndex())
      return axis;
  }
  if (!value || depth >= 64)
    return std::nullopt;
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return std::nullopt;
  StringRef name = operation->getName().getStringRef();
  if ((name == "neura.data_mov" || name == "neura.grant_predicate") &&
      operation->getNumOperands() >= 1)
    return sourceOwnedOutputCounterAxis(operation->getOperand(0), counters,
                                        depth + 1);
  if (name == "neura.phi" && operation->getNumOperands() == 2) {
    std::optional<unsigned> first = sourceOwnedOutputCounterAxis(
        operation->getOperand(0), counters, depth + 1);
    std::optional<unsigned> second = sourceOwnedOutputCounterAxis(
        operation->getOperand(1), counters, depth + 1);
    if (first && second && *first == *second)
      return first;
  }
  return std::nullopt;
}

static bool validateSourceOwnedTaskBox(
    const TaskMetadata &parent, const ReplicaRecord &record,
    ArrayRef<CounterBounds> parentBounds, SourceOwnedPartitionBox &box,
    std::string &error) {
  TaskflowTaskOp task = record.child;
  const size_t rank = parentBounds.size();
  if (rank < 2 || rank > 3) {
    error = "source-owned nested partition has an unsupported M/N/K rank";
    return false;
  }
  box.bounds.assign(rank, CounterBounds{});
  SmallVector<TaskflowCounterOp> taskCounters;
  task.walk([&](TaskflowCounterOp counter) { taskCounters.push_back(counter); });
  if (taskCounters.size() != rank) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " must retain exactly its canonical M/N[/K] counters";
    return false;
  }
  for (size_t id = 0; id < rank; ++id) {
    unsigned matches = 0;
    for (TaskflowCounterOp counter : taskCounters)
      if (auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
          counterId && counterId.getInt() == static_cast<int64_t>(id))
        ++matches;
    if (matches != 1) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has a missing or duplicate Taskflow counter id " +
              std::to_string(id);
      return false;
    }
    if (!findTaskflowCounterBounds(task, id, box.bounds[id], error))
      return false;
    if (box.bounds[id].step != 1 ||
        box.bounds[id].lower < parentBounds[id].lower ||
        box.bounds[id].upper > parentBounds[id].upper) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has an out-of-range or non-unit Taskflow counter box";
      return false;
    }
  }

  SmallVector<neura::KernelOp> kernels;
  task.walk([&](neura::KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.size() != 1) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " must have one Neura kernel for counter-box proof";
    return false;
  }
  SmallVector<neura::CounterOp> neuraCounters;
  kernels.front().walk(
      [&](neura::CounterOp counter) { neuraCounters.push_back(counter); });
  if (neuraCounters.size() != rank) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " must retain exactly its canonical Neura M/N[/K] counters";
    return false;
  }
  for (size_t id = 0; id < rank; ++id) {
    unsigned matches = 0;
    for (neura::CounterOp counter : neuraCounters)
      if (auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
          counterId && counterId.getInt() == static_cast<int64_t>(id))
        ++matches;
    if (matches != 1) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has a missing or duplicate Neura counter id " +
              std::to_string(id);
      return false;
    }
    CounterBounds neuraBounds;
    if (!findNeuraCounterBounds(task, id, neuraBounds, error) ||
        !equalBounds(box.bounds[id], neuraBounds)) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has Taskflow/Neura counter bound drift";
      return false;
    }
  }

  if (rank == 3 && !equalBounds(box.bounds[2], parentBounds[2])) {
    error = "source-owned rank-three child changed its canonical K "
            "reduction bounds";
    return false;
  }

  int64_t volume = 0;
  if (!checkedCounterVolume(box.bounds, volume) ||
      record.childTripCount != volume) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " trip count does not match its actual M/N[/K] counter volume";
    return false;
  }

  auto lowers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_lowers");
  auto uppers = task->getAttrOfType<ArrayAttr>(
      "amoeba.tiling.output_region_uppers");
  if (!lowers || !uppers || lowers.size() != task.getWillWrites().size() ||
      uppers.size() != lowers.size() || lowers.empty()) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " has incomplete output-region metadata";
    return false;
  }

  // Each indexed store must name one of the task's actual output kernel
  // inputs, and every index must be a direct coordinate route from the same
  // two counters.  This catches a changed/folded output base and a metadata
  // region that no longer describes the coordinates actually written.
  Block &body = task.getBody().front();
  unsigned firstWrite = task.getWillReads().size();
  llvm::SmallDenseSet<unsigned, 4> outputKernelInputs;
  SmallVector<unsigned> outputKernelInputByWrite;
  for (unsigned writeIndex = 0; writeIndex < task.getWillWrites().size();
       ++writeIndex) {
    if (firstWrite + writeIndex >= body.getNumArguments()) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has no body argument for an output";
      return false;
    }
    Value outputArgument = body.getArgument(firstWrite + writeIndex);
    for (auto [inputIndex, input] : llvm::enumerate(kernels.front().getInputs()))
      if (input == outputArgument) {
        outputKernelInputs.insert(inputIndex);
        outputKernelInputByWrite.push_back(inputIndex);
      }
  }
  if (outputKernelInputs.size() != task.getWillWrites().size() ||
      outputKernelInputByWrite.size() != task.getWillWrites().size()) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " has an output kernel-input mapping gap";
    return false;
  }
  SmallVector<unsigned> outputCounterAxes;
  SmallVector<unsigned> outputStoreCounts(task.getWillWrites().size(), 0);
  kernels.front().walk([&](neura::StoreIndexedOp store) {
    if (!error.empty())
      return;
    auto outputInput =
        mlir::amoeba::neura::joint_scheduling::detail::accessInput(
            store, store.getBase(), kernels.front());
    auto outputIt = outputInput
                       ? llvm::find(outputKernelInputByWrite, *outputInput)
                       : outputKernelInputByWrite.end();
    if (!outputInput || !outputKernelInputs.contains(*outputInput)) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has an indexed store outside its authenticated task outputs";
      return;
    }
    if (outputIt == outputKernelInputByWrite.end() ||
        store.getIndices().size() != 2) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has an indexed output store with a non-rectangular "
              "coordinate route";
      return;
    }
    unsigned outputIndex = outputIt - outputKernelInputByWrite.begin();
    SmallVector<unsigned> route;
    for (Value index : store.getIndices()) {
      std::optional<unsigned> axis = sourceOwnedOutputCounterAxis(
          index, neuraCounters);
      if (!axis || llvm::is_contained(route, *axis)) {
        error = "source-owned nested task " + task.getTaskName().str() +
                " has an indexed output coordinate that is not a unique "
                "counter route";
        return;
      }
      route.push_back(*axis);
    }
    if (rank == 3) {
      SmallVector<unsigned> sortedRoute(route);
      llvm::sort(sortedRoute);
      if (sortedRoute != SmallVector<unsigned>{0, 1}) {
        error = "source-owned rank-three output must be indexed by M/N "
                "counters while retaining the full K reduction";
        return;
      }
    }
    if (outputCounterAxes.empty())
      outputCounterAxes = route;
    else if (!llvm::equal(outputCounterAxes, route)) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " output stores disagree on their counter-coordinate map";
      return;
    }
    ++outputStoreCounts[outputIndex];
  });
  if (!error.empty())
    return false;
  if (outputCounterAxes.size() != 2 ||
      llvm::any_of(outputStoreCounts, [](unsigned count) { return count == 0; })) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " does not cover every output with an authenticated indexed "
            "store";
    return false;
  }
  for (auto [regionIndex, pair] : llvm::enumerate(llvm::zip(lowers, uppers))) {
    auto lower = dyn_cast<DenseI64ArrayAttr>(std::get<0>(pair));
    auto upper = dyn_cast<DenseI64ArrayAttr>(std::get<1>(pair));
    if (!lower || !upper ||
        static_cast<size_t>(lower.size()) != outputCounterAxes.size() ||
        static_cast<size_t>(upper.size()) != outputCounterAxes.size()) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " has malformed output-region rank";
      return false;
    }
    for (auto [dimension, axis] : llvm::enumerate(outputCounterAxes))
      if (lower[dimension] != box.bounds[axis].lower ||
          upper[dimension] != box.bounds[axis].upper) {
        error = "source-owned nested task " + task.getTaskName().str() +
                " output region disagrees with its actual indexed output "
                "coordinates";
        return false;
      }
    (void)regionIndex;
  }

  // Authenticate all source-owned partition steps, including a replica after
  // a tile or a tile after a replica.  The latest tile's range attributes
  // describe that intermediate tile even when the final task is a later
  // replica, so they are checked against their own ledger step rather than
  // against the final task box.
  bool hasPartitionLineage =
      task->hasAttr(kNeighborhoodPartitionLineageAttr) ||
      task->hasAttr("amoeba.neura.tiling.axis") ||
      task->hasAttr("amoeba.replica.parent_task");
  if (hasPartitionLineage) {
    SmallVector<CounterBounds, 3> latestTilingParentBounds;
    bool finalStepIsTiling = false;
    if (!sourceOwnedImmediateTilingBounds(
            task, parent.name, parentBounds, box.bounds,
            latestTilingParentBounds, finalStepIsTiling, error))
      return false;
    if (finalStepIsTiling) {
      auto axis = task->getAttrOfType<IntegerAttr>(
          "amoeba.neura.tiling.axis");
      if (!axis || axis.getInt() < 0 || axis.getInt() > 1) {
        error = "source-owned nested task " + task.getTaskName().str() +
                " has no valid final tiling axis";
        return false;
      }
      for (size_t dimension = 0; dimension < rank; ++dimension)
        if (static_cast<int64_t>(dimension) != axis.getInt() &&
            !equalBounds(box.bounds[dimension],
                         latestTilingParentBounds[dimension])) {
          error = "source-owned nested task " + task.getTaskName().str() +
                  " changed an unsharded M/N axis of its immediate tile "
                  "parent";
          return false;
        }
    }
  }
  box.record = &record;
  return true;
}

static Value stripSourceOwnedMemrefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

static TaskflowTaskOp sourceOwnedStateTask(Value state) {
  state = stripSourceOwnedMemrefCasts(state);
  return state.getDefiningOp<TaskflowTaskOp>();
}

struct SourceOwnedLineageStep {
  std::string family;
  int64_t axis = -1;
  int64_t factor = 0;
  int64_t part = -1;
  SmallVector<CounterBounds, 3> before;
  SmallVector<CounterBounds, 3> after;
};

static bool decodeSourceOwnedLineageSteps(
    TaskflowTaskOp task, StringRef canonicalName,
    ArrayRef<CounterBounds> canonicalBounds,
    SmallVectorImpl<SourceOwnedLineageStep> &steps, std::string &error) {
  auto roots = dyn_cast_or_null<ArrayAttr>(
      task->getAttr(kNeighborhoodPartitionLineageAttr));
  auto root = roots && roots.size() == 1
                  ? dyn_cast<DictionaryAttr>(roots[0])
                  : DictionaryAttr{};
  auto rootName = root ? root.getAs<StringAttr>("root") : StringAttr{};
  auto original = root ? root.getAs<DenseI64ArrayAttr>("original")
                       : DenseI64ArrayAttr{};
  auto encodedSteps = root ? root.getAs<ArrayAttr>("steps") : ArrayAttr{};
  SmallVector<CounterBounds, 3> originalBounds(canonicalBounds.size());
  if (!rootName || rootName.getValue() != canonicalName || !original ||
      !decodeLineageCounterBounds(original, originalBounds) ||
      !llvm::equal(originalBounds, canonicalBounds, equalBounds) ||
      !encodedSteps || encodedSteps.empty()) {
    error = "source-owned mixed partition has no complete canonical typed "
            "lineage";
    return false;
  }

  std::string derivedName = canonicalName.str();
  SmallVector<CounterBounds, 3> current(canonicalBounds.begin(),
                                        canonicalBounds.end());
  for (Attribute attribute : encodedSteps) {
    auto encoded = dyn_cast<DictionaryAttr>(attribute);
    auto family = encoded ? encoded.getAs<StringAttr>("family") : StringAttr{};
    auto axis = encoded ? encoded.getAs<IntegerAttr>("axis") : IntegerAttr{};
    auto factor = encoded ? encoded.getAs<IntegerAttr>("factor")
                          : IntegerAttr{};
    auto part = encoded ? encoded.getAs<IntegerAttr>("part") : IntegerAttr{};
    SourceOwnedLineageStep step;
    step.before.resize(canonicalBounds.size());
    step.after.resize(canonicalBounds.size());
    if (!family || !axis || !factor || !part ||
        !decodeLineageCounterBounds(encoded.get("before"), step.before) ||
        !decodeLineageCounterBounds(encoded.get("after"), step.after) ||
        !llvm::equal(step.before, current, equalBounds) ||
        !lineagePartitionMatches(family.getValue(), axis.getInt(),
                                 factor.getInt(), part.getInt(), step.before,
                                 step.after)) {
      error = "source-owned mixed partition has a broken typed lineage "
              "step";
      return false;
    }
    step.family = family.getValue().str();
    step.axis = axis.getInt();
    step.factor = factor.getInt();
    step.part = part.getInt();
    if (step.family == "tiling")
      derivedName += (Twine(".tile.") + Twine(step.axis) + "." +
                      Twine(step.part))
                         .str();
    else
      derivedName += (Twine(".replica.") + Twine(step.part)).str();
    current.assign(step.after.begin(), step.after.end());
    steps.push_back(std::move(step));
  }
  if (derivedName != task.getTaskName()) {
    error = "source-owned mixed partition lineage does not derive the "
            "current task name";
    return false;
  }
  return true;
}

static bool collectSourceOwnedJoinLeaves(
    Value state, SmallVectorImpl<TaskflowTaskOp> &leaves,
    DenseSet<Operation *> &activeJoins) {
  state = stripSourceOwnedMemrefCasts(state);
  if (TaskflowTaskOp task = state.getDefiningOp<TaskflowTaskOp>()) {
    leaves.push_back(task);
    return true;
  }
  TaskflowJoinOp join = state.getDefiningOp<TaskflowJoinOp>();
  if (!join || !activeJoins.insert(join.getOperation()).second)
    return false;
  if (join.getTileStates().empty())
    return false;
  for (Value childState : join.getTileStates())
    if (!collectSourceOwnedJoinLeaves(childState, leaves, activeJoins))
      return false;
  activeJoins.erase(join.getOperation());
  return true;
}

static bool sameSourceOwnedLineageStepKey(const SourceOwnedLineageStep &lhs,
                                          const SourceOwnedLineageStep &rhs) {
  return lhs.family == rhs.family && lhs.axis == rhs.axis &&
         lhs.factor == rhs.factor &&
         llvm::equal(lhs.before, rhs.before, equalBounds);
}

static bool sameSourceOwnedLineageStep(const SourceOwnedLineageStep &lhs,
                                       const SourceOwnedLineageStep &rhs) {
  return sameSourceOwnedLineageStepKey(lhs, rhs) && lhs.part == rhs.part &&
         llvm::equal(lhs.after, rhs.after, equalBounds);
}

static bool sourceOwnedBoxesCoverRegion(
    ArrayRef<unsigned> indices, ArrayRef<SourceOwnedPartitionBox> boxes,
    ArrayRef<CounterBounds> region, std::string &error) {
  if (region.size() < 2 || region.size() > 3) {
    error = "source-owned completion-join branch has an unsupported typed "
            "region rank";
    return false;
  }
  __int128 volume = 0;
  for (unsigned index : indices) {
    ArrayRef<CounterBounds> box = boxes[index].bounds;
    if (box[0].lower < region[0].lower || box[0].upper > region[0].upper ||
        box[1].lower < region[1].lower || box[1].upper > region[1].upper) {
      error = "source-owned completion-join branch has a leaf outside its "
              "typed output region";
      return false;
    }
    if (region.size() == 3 &&
        !equalBounds(box[2], region[2])) {
      error = "source-owned completion-join branch changes its retained K "
              "reduction bounds";
      return false;
    }
    volume += static_cast<__int128>(box[0].upper - box[0].lower) *
              static_cast<__int128>(box[1].upper - box[1].lower);
    for (unsigned other : indices) {
      if (other <= index)
        continue;
      bool overlap = box[0].lower < boxes[other].bounds[0].upper &&
                     boxes[other].bounds[0].lower < box[0].upper &&
                     box[1].lower < boxes[other].bounds[1].upper &&
                     boxes[other].bounds[1].lower < box[1].upper;
      if (overlap) {
        error = "source-owned completion-join branch contains overlapping "
                "actual leaf boxes";
        return false;
      }
    }
  }
  __int128 expected = static_cast<__int128>(region[0].upper - region[0].lower) *
                      static_cast<__int128>(region[1].upper - region[1].lower);
  if (expected <= 0 || volume != expected) {
    error = "source-owned completion-join branch leaves do not cover their "
            "typed output region";
    return false;
  }
  return true;
}

// A mixed replica/tile state is an intermediate tree: one replica branch
// may already be replaced by a nested tile join while sibling branches remain
// leaf tasks. Authenticate the entire result tree against each leaf's typed
// partition ledger, and link every join state to the exact child completion
// result. The task-box checks performed by validateSourceOwnedPartition
// independently prove current Taskflow/Neura bounds and indexed output
// coordinates before this structural check runs.
static bool validateSourceOwnedMixedJoinTree(
    StringRef canonicalName, ArrayRef<CounterBounds> canonicalBounds,
    ArrayRef<ReplicaRecord> records,
    ArrayRef<SourceOwnedPartitionBox> boxes, size_t canonicalDoneReadCount,
    std::string &error) {
  if (canonicalDoneReadCount != 0) {
    error = "source-owned mixed partition with sparse done-read outputs is "
            "unsupported";
    return false;
  }
  func::FuncOp function =
      records.front().child->getParentOfType<func::FuncOp>();
  if (!function) {
    error = "source-owned mixed partition has no enclosing function";
    return false;
  }

  SmallVector<SmallVector<SourceOwnedLineageStep>> lineages(records.size());
  std::map<Operation *, unsigned> recordByTask;
  for (auto [index, record] : llvm::enumerate(records)) {
    TaskflowTaskOp child = record.child;
    if (!recordByTask.emplace(child.getOperation(), index).second ||
        !decodeSourceOwnedLineageSteps(child, canonicalName,
                                       canonicalBounds, lineages[index],
                                       error))
      return false;
    if (child.getWillWrites().size() != 1 ||
        child.getDoneWrites().size() != 1 ||
        child.getOriginalWriteMemrefs().size() != 1) {
      error = "source-owned mixed partition requires one exact output and "
              "completion result per leaf";
      return false;
    }
  }

  auto indicesForState = [&](Value state, SmallVectorImpl<unsigned> &indices) {
    SmallVector<TaskflowTaskOp> leaves;
    DenseSet<Operation *> activeJoins;
    if (!collectSourceOwnedJoinLeaves(state, leaves, activeJoins))
      return false;
    DenseSet<unsigned> seen;
    for (TaskflowTaskOp leaf : leaves) {
      auto found = recordByTask.find(leaf.getOperation());
      if (found == recordByTask.end() || !seen.insert(found->second).second)
        return false;
      indices.push_back(found->second);
    }
    return !indices.empty();
  };

  std::function<bool(TaskflowJoinOp, ArrayRef<unsigned>,
                     DenseSet<Operation *> &)> validateNode;
  validateNode = [&](TaskflowJoinOp join, ArrayRef<unsigned> indices,
                     DenseSet<Operation *> &active) {
    if (!join || indices.size() < 2 ||
        !active.insert(join.getOperation()).second ||
        failed(verify(join.getOperation()))) {
      error = "source-owned mixed partition has an invalid or cyclic "
              "completion join";
      return false;
    }
    auto leaveActive = [&]() { active.erase(join.getOperation()); };

    size_t split = 0;
    for (;; ++split) {
      if (llvm::any_of(indices, [&](unsigned index) {
            return lineages[index].size() <= split;
          })) {
        error = "source-owned mixed completion tree ends before its leaf "
                "lineages diverge";
        leaveActive();
        return false;
      }
      bool common = llvm::all_of(indices.drop_front(), [&](unsigned index) {
        return sameSourceOwnedLineageStep(
            lineages[indices.front()][split], lineages[index][split]);
      });
      if (!common)
        break;
    }
    const SourceOwnedLineageStep &splitStep =
        lineages[indices.front()][split];
    for (unsigned index : indices)
      if (!sameSourceOwnedLineageStepKey(splitStep,
                                         lineages[index][split])) {
        error = "source-owned mixed completion join does not correspond to "
                "one typed partition step";
        leaveActive();
        return false;
      }
    if (splitStep.factor != static_cast<int64_t>(join.getTileStates().size()) ||
        join.getAxis() != static_cast<uint64_t>(splitStep.axis) ||
        join.getRegionLower().size() != 2 ||
        join.getRegionUpper().size() != 2) {
      error = "source-owned mixed completion join has a mismatched typed "
              "factor, axis, or region rank";
      leaveActive();
      return false;
    }
    for (int64_t dimension = 0; dimension < 2; ++dimension)
      if (join.getRegionLower()[dimension] !=
              splitStep.before[dimension].lower ||
          join.getRegionUpper()[dimension] !=
              splitStep.before[dimension].upper) {
        error = "source-owned mixed completion join region disagrees with "
                "its typed before bounds";
        leaveActive();
        return false;
      }
    if (!join->hasAttr("amoeba.semantic.completion_only")) {
      error = "source-owned mixed completion join lacks semantic "
              "completion-only marking";
      leaveActive();
      return false;
    }
    if (splitStep.family == "replica") {
      if (!join->hasAttr("amoeba.replica.completion_only") ||
          join->hasAttr("amoeba.neura.joint_rewrite")) {
        error = "source-owned mixed replica split has the wrong completion "
                "join identity";
        leaveActive();
        return false;
      }
    } else {
      auto rewrite = join->getAttrOfType<StringAttr>(
          "amoeba.neura.joint_rewrite");
      if (!rewrite || rewrite.getValue() != "post-neura-mn-tiling" ||
          join->hasAttr("amoeba.replica.completion_only")) {
        error = "source-owned mixed tile split has the wrong completion "
                "join identity";
        leaveActive();
        return false;
      }
    }

    Value joinBase = stripSourceOwnedMemrefCasts(join.getBase());
    if (!joinBase) {
      error = "source-owned mixed completion join has no storage base";
      leaveActive();
      return false;
    }
    for (unsigned index : indices) {
      TaskflowTaskOp child = records[index].child;
      if (stripSourceOwnedMemrefCasts(child.getWillWrites().front()) !=
              joinBase ||
          stripSourceOwnedMemrefCasts(
              child.getOriginalWriteMemrefs().front()) != joinBase) {
        error = "source-owned mixed completion join changes a leaf output "
                "storage root";
        leaveActive();
        return false;
      }
    }

    DenseSet<unsigned> joinedIndices;
    DenseSet<int64_t> joinedParts;
    for (Value state : join.getTileStates()) {
      SmallVector<unsigned> branchIndices;
      if (!indicesForState(state, branchIndices)) {
        error = "source-owned mixed completion join has a state outside its "
                "authenticated leaf set";
        leaveActive();
        return false;
      }
      for (unsigned index : branchIndices)
        if (!llvm::is_contained(indices, index) ||
            !joinedIndices.insert(index).second) {
          error = "source-owned mixed completion join duplicates or escapes "
                  "a leaf";
          leaveActive();
          return false;
        }
      const SourceOwnedLineageStep &branchStep =
          lineages[branchIndices.front()][split];
      if (!sameSourceOwnedLineageStepKey(splitStep, branchStep) ||
          !joinedParts.insert(branchStep.part).second) {
        error = "source-owned mixed completion join has duplicate or "
                "inconsistent typed partition parts";
        leaveActive();
        return false;
      }
      for (unsigned index : branchIndices)
        if (!sameSourceOwnedLineageStep(branchStep,
                                        lineages[index][split]) ||
            lineages[index][split].part != branchStep.part) {
          error = "source-owned mixed join state combines different typed "
                  "partition parts";
          leaveActive();
          return false;
        }
      if (branchStep.part < 0 || branchStep.part >= splitStep.factor ||
          !sourceOwnedBoxesCoverRegion(branchIndices, boxes,
                                       branchStep.after, error)) {
        if (error.empty())
          error = "source-owned mixed completion join has an invalid typed "
                  "branch interval";
        leaveActive();
        return false;
      }

      Value stateRoot = stripSourceOwnedMemrefCasts(state);
      if (TaskflowTaskOp leaf = stateRoot.getDefiningOp<TaskflowTaskOp>()) {
        if (branchIndices.size() != 1 || leaf != records[branchIndices.front()].child ||
            leaf.getDoneWrites().size() != 1 ||
            stripSourceOwnedMemrefCasts(leaf.getDoneWrites().front()) !=
                stateRoot) {
          error = "source-owned mixed join state is not the exact leaf "
                  "completion result";
          leaveActive();
          return false;
        }
      } else if (TaskflowJoinOp nested =
                     stateRoot.getDefiningOp<TaskflowJoinOp>()) {
        if (stripSourceOwnedMemrefCasts(nested.getJoined()) != stateRoot ||
            !validateNode(nested, branchIndices, active)) {
          if (error.empty())
            error = "source-owned mixed join state is not the exact nested "
                    "completion result";
          leaveActive();
          return false;
        }
      } else {
        error = "source-owned mixed join state is neither a leaf task nor a "
                "nested completion join";
        leaveActive();
        return false;
      }
    }
    if (joinedIndices.size() != indices.size() ||
        static_cast<int64_t>(joinedParts.size()) != splitStep.factor) {
      error = "source-owned mixed completion join omits a typed leaf or "
              "partition part";
      leaveActive();
      return false;
    }
    for (int64_t part = 0; part < splitStep.factor; ++part)
      if (!joinedParts.contains(part)) {
        error = "source-owned mixed completion join is missing typed part " +
                std::to_string(part);
        leaveActive();
        return false;
      }
    leaveActive();
    return true;
  };

  SmallVector<unsigned> expectedIndices;
  for (unsigned index = 0; index < records.size(); ++index)
    expectedIndices.push_back(index);
  SmallVector<TaskflowJoinOp> rootJoins;
  function.walk([&](TaskflowJoinOp candidate) {
    SmallVector<unsigned> indices;
    if (!indicesForState(candidate.getJoined(), indices) ||
        indices.size() != records.size())
      return;
    llvm::sort(indices);
    if (llvm::equal(indices, expectedIndices))
      rootJoins.push_back(candidate);
  });
  if (rootJoins.size() != 1) {
    error = "source-owned mixed partition does not have one unique root "
            "completion join for its complete leaf set";
    return false;
  }
  DenseSet<Operation *> active;
  return validateNode(rootJoins.front(), expectedIndices, active);
}

// The final partition check proves global coverage.  This second check proves
// each replaced intermediate tile's own completion join: the children must
// cover that join's region along exactly its selected axis and must agree on
// every unsharded axis.  It also authenticates the join states to the child
// done-write roots, so a forged parent name cannot borrow an unrelated join.
static bool validateSourceOwnedNestedJoins(
    ArrayRef<ReplicaRecord> records,
    ArrayRef<SourceOwnedPartitionBox> boxes, size_t canonicalDoneReadCount,
    std::string &error) {
  if (records.size() != boxes.size() || records.empty())
    return false;
  func::FuncOp function =
      records.front().child->getParentOfType<func::FuncOp>();
  if (!function) {
    error = "source-owned nested partition has no enclosing function";
    return false;
  }

  std::map<std::string, SmallVector<unsigned>> groups;
  for (auto [index, record] : llvm::enumerate(records)) {
    auto parent = record.child->getAttrOfType<StringAttr>(
        "amoeba.replica.parent_task");
    if (!parent)
      continue;
    groups[parent.getValue().str()].push_back(index);
  }
  for (const auto &group : groups) {
    StringRef parentName = group.first;
    ArrayRef<unsigned> indices = group.second;
    if (indices.size() < 2)
      return false;
    int64_t selectedAxis = -1;
    int64_t replicaCount = -1;
    DenseSet<Operation *> expectedTasks;
    DenseSet<int64_t> replicaIds;
    for (unsigned index : indices) {
      const ReplicaRecord &record = records[index];
      TaskflowTaskOp child = record.child;
      auto parent = child->getAttrOfType<StringAttr>(
          "amoeba.replica.parent_task");
      auto axis = child->getAttrOfType<IntegerAttr>(
          "amoeba.replica.shard_axis");
      auto count = child->getAttrOfType<IntegerAttr>(
          "amoeba.replica.count");
      auto id = child->getAttrOfType<IntegerAttr>("amoeba.replica.id");
      auto lower = child->getAttrOfType<IntegerAttr>(
          "amoeba.replica.shard_lower");
      auto upper = child->getAttrOfType<IntegerAttr>(
          "amoeba.replica.shard_upper");
      if (!parent || parent.getValue() != parentName || !axis || !count ||
          !id || !lower || !upper || axis.getInt() < 0 || axis.getInt() > 1 ||
          count.getInt() < 2 || id.getInt() < 0 || id.getInt() >= count.getInt() ||
          lower.getInt() >= upper.getInt() ||
          !child.getTaskName().starts_with((Twine(parentName) + ".replica.").str())) {
        error = "source-owned nested replica has incomplete parent or shard "
                "identity";
        return false;
      }
      if (selectedAxis < 0)
        selectedAxis = axis.getInt();
      if (replicaCount < 0)
        replicaCount = count.getInt();
      if (selectedAxis != axis.getInt() || replicaCount != count.getInt() ||
          !replicaIds.insert(id.getInt()).second || record.id != id.getInt() ||
          record.count != count.getInt() ||
          boxes[index].bounds[axis.getInt()].lower != lower.getInt() ||
          boxes[index].bounds[axis.getInt()].upper != upper.getInt()) {
        error = "source-owned nested replica has inconsistent shard identity "
                "or bounds";
        return false;
      }
      expectedTasks.insert(child.getOperation());
    }
    if (replicaCount != static_cast<int64_t>(indices.size()) ||
        static_cast<int64_t>(replicaIds.size()) != replicaCount)
      return false;

    TaskflowJoinOp completionJoin;
    function.walk([&](TaskflowJoinOp candidate) {
      if (completionJoin ||
          candidate.getTileStates().size() != indices.size())
        return;
      DenseSet<Operation *> actualTasks;
      for (Value state : candidate.getTileStates()) {
        TaskflowTaskOp task = sourceOwnedStateTask(state);
        if (!task || !expectedTasks.contains(task.getOperation()) ||
            !actualTasks.insert(task.getOperation()).second) {
          actualTasks.clear();
          break;
        }
      }
      if (actualTasks.size() == expectedTasks.size())
        completionJoin = candidate;
    });
    if (!completionJoin || !completionJoin->hasAttr("amoeba.replica.completion_only") ||
        !completionJoin->hasAttr("amoeba.semantic.completion_only")) {
      error = "source-owned nested replica parent has no authenticated "
              "completion join";
      return false;
    }
    if (completionJoin.getAxis() != static_cast<uint64_t>(selectedAxis) ||
        completionJoin.getRegionLower().size() != 2 ||
        completionJoin.getRegionUpper().size() != 2) {
      error = "source-owned nested completion join has an invalid shard axis "
              "or region rank";
      return false;
    }

    Value joinBase = stripSourceOwnedMemrefCasts(completionJoin.getBase());
    SmallVector<std::pair<int64_t, int64_t>> intervals;
    intervals.reserve(indices.size());
    for (unsigned index : indices) {
      const ReplicaRecord &record = records[index];
      const SourceOwnedPartitionBox &box = boxes[index];
      TaskflowTaskOp child = record.child;
      if (box.bounds[0].lower < completionJoin.getRegionLower()[0] ||
          box.bounds[0].upper > completionJoin.getRegionUpper()[0] ||
          box.bounds[1].lower < completionJoin.getRegionLower()[1] ||
          box.bounds[1].upper > completionJoin.getRegionUpper()[1]) {
        error = "source-owned nested replica box is outside its parent join "
                "region";
        return false;
      }
      for (int64_t dimension = 0; dimension < 2; ++dimension)
        if (dimension != selectedAxis &&
            (box.bounds[dimension].lower !=
                 completionJoin.getRegionLower()[dimension] ||
             box.bounds[dimension].upper !=
                 completionJoin.getRegionUpper()[dimension])) {
          error = "source-owned nested replica changed an unsharded axis "
                  "outside its parent join";
          return false;
        }
      if (child.getWillWrites().size() != 1 ||
          child.getDoneWrites().size() != child.getWillWrites().size() ||
          child.getOriginalWriteMemrefs().size() !=
              child.getWillWrites().size() ||
          stripSourceOwnedMemrefCasts(child.getWillWrites().front()) !=
              joinBase ||
          stripSourceOwnedMemrefCasts(child.getOriginalWriteMemrefs().front()) !=
              joinBase) {
        error = "source-owned nested completion join has a child storage-root "
                "or write-arity mismatch";
        return false;
      }
      intervals.push_back({box.bounds[selectedAxis].lower,
                           box.bounds[selectedAxis].upper});
    }
    llvm::sort(intervals);
    int64_t cursor = completionJoin.getRegionLower()[selectedAxis];
    for (auto [lower, upper] : intervals) {
      if (lower != cursor || upper <= lower) {
        error = "source-owned nested completion join has overlapping or "
                "gapped shard intervals";
        return false;
      }
      cursor = upper;
    }
    if (cursor != completionJoin.getRegionUpper()[selectedAxis]) {
      error = "source-owned nested completion join does not cover its parent "
              "axis";
      return false;
    }

    // Sparse done-read results carry WAR completion just as output joins
    // carry write completion. Every source read result must be preserved by
    // a distinct typed fan-in linked to this exact output join; forwarding a
    // single replica's read token would let a later writer race sibling reads.
    TaskflowTaskOp firstChild = records.front().child;
    const size_t doneReadCount = firstChild.getDoneReads().size();
    if (doneReadCount != canonicalDoneReadCount || doneReadCount > 1 ||
        !llvm::all_of(records, [&](const ReplicaRecord &record) {
          TaskflowTaskOp child = record.child;
          return child.getDoneReads().size() == doneReadCount;
        })) {
      error = "source-owned replica has inconsistent or unsupported sparse "
              "done-read outputs";
      return false;
    }
    SmallVector<TaskflowReadCompletionJoinOp> readJoins;
    function.walk([&](TaskflowReadCompletionJoinOp candidate) {
      if (candidate.getWriteCompletion() == completionJoin.getJoined())
        readJoins.push_back(candidate);
    });
    if (readJoins.size() != doneReadCount) {
      error = "source-owned replica read completion join count does not "
              "preserve its sparse done-read outputs";
      return false;
    }
    for (unsigned readResult = 0; readResult < doneReadCount; ++readResult) {
      DenseSet<Value> expectedReadStates;
      for (unsigned index : indices) {
        TaskflowTaskOp child = records[index].child;
        expectedReadStates.insert(child.getDoneReads()[readResult]);
      }
      TaskflowReadCompletionJoinOp readJoin;
      for (TaskflowReadCompletionJoinOp candidate : readJoins) {
        DenseSet<Value> actualReadStates;
        for (Value state : candidate.getTileStates())
          actualReadStates.insert(state);
        if (actualReadStates.size() == expectedReadStates.size() &&
            llvm::all_of(expectedReadStates, [&](Value state) {
              return actualReadStates.contains(state);
            }))
          readJoin = candidate;
      }
      if (!readJoin || failed(verify(readJoin.getOperation()))) {
        error = "source-owned replica done-read fan-in does not match every "
                "child and its output completion join";
        return false;
      }
      for (Value state : expectedReadStates) {
        if (!llvm::hasSingleElement(state.getUses()) ||
            state.use_begin()->getOwner() != readJoin.getOperation()) {
          error = "source-owned replica done-read state has an escaped or "
                  "missing join use";
          return false;
        }
      }
    }
  }
  return true;
}

static bool validateSourceOwnedRankOneTilingMetadata(
    TaskflowTaskOp task, StringRef canonicalName,
    const CounterBounds &canonicalBounds, const CounterBounds &currentBounds,
    std::string &error) {
  auto parent =
      task->getAttrOfType<StringAttr>("amoeba.neura.tiling.parent_task");
  auto axis = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.axis");
  auto part =
      task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.part_index");
  auto factor = task->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
  auto original = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.original_range");
  auto derived = task->getAttrOfType<DenseI64ArrayAttr>(
      "amoeba.neura.tiling.derived_range");
  auto rewrite = task->getAttrOfType<StringAttr>("amoeba.neura.joint_rewrite");
  std::string expectedName =
      (Twine(canonicalName) + ".tile.0." + Twine(part ? part.getInt() : -1))
          .str();
  if (!parent || parent.getValue() != canonicalName || !axis ||
      axis.getInt() != 0 || !part || part.getInt() < 0 || !factor ||
      (factor.getInt() != 2 && factor.getInt() != 4 && factor.getInt() != 8) ||
      part.getInt() >= factor.getInt() || !original || original.size() != 2 ||
      !derived || derived.size() != 2 || original[0] != canonicalBounds.lower ||
      original[1] != canonicalBounds.upper ||
      derived[0] != currentBounds.lower || derived[1] != currentBounds.upper ||
      !rewrite || rewrite.getValue() != "post-neura-mn-tiling" ||
      task.getTaskName() != StringRef(expectedName) ||
      task->hasAttr("amoeba.replica.parent_task")) {
    error = "source-owned rank-1 tile attributes disagree with its canonical "
            "parent, part, or current counter bounds";
    return false;
  }
  int64_t extent = canonicalBounds.upper - canonicalBounds.lower;
  int64_t width = extent / factor.getInt();
  int64_t remainder = extent % factor.getInt();
  int64_t expectedLower = canonicalBounds.lower + part.getInt() * width +
                          std::min(part.getInt(), remainder);
  int64_t expectedUpper = expectedLower + width +
                          (part.getInt() < remainder ? 1 : 0);
  if (currentBounds.lower != expectedLower ||
      currentBounds.upper != expectedUpper) {
    error = "source-owned rank-1 tile interval does not match its ordered part";
    return false;
  }
  Attribute lineageAttribute = task->getAttr(kNeighborhoodPartitionLineageAttr);
  if (!lineageAttribute)
    return true;
  auto roots = dyn_cast<ArrayAttr>(lineageAttribute);
  auto root = roots && roots.size() == 1 ? dyn_cast<DictionaryAttr>(roots[0])
                                         : DictionaryAttr{};
  auto rootName = root ? root.getAs<StringAttr>("root") : StringAttr{};
  auto originalDomain =
      root ? root.getAs<DenseI64ArrayAttr>("original") : DenseI64ArrayAttr{};
  auto steps = root ? root.getAs<ArrayAttr>("steps") : ArrayAttr{};
  auto tripleMatches = [](DenseI64ArrayAttr encoded,
                          const CounterBounds &bounds) {
    return encoded && encoded.size() == 3 && encoded[0] == bounds.lower &&
           encoded[1] == bounds.upper && encoded[2] == bounds.step;
  };
  auto step = steps && steps.size() == 1 ? dyn_cast<DictionaryAttr>(steps[0])
                                         : DictionaryAttr{};
  auto family = step ? step.getAs<StringAttr>("family") : StringAttr{};
  auto lineageAxis = step ? step.getAs<IntegerAttr>("axis") : IntegerAttr{};
  auto lineageFactor = step ? step.getAs<IntegerAttr>("factor") : IntegerAttr{};
  auto lineagePart = step ? step.getAs<IntegerAttr>("part") : IntegerAttr{};
  auto before =
      step ? step.getAs<DenseI64ArrayAttr>("before") : DenseI64ArrayAttr{};
  auto after =
      step ? step.getAs<DenseI64ArrayAttr>("after") : DenseI64ArrayAttr{};
  if (!rootName || rootName.getValue() != canonicalName || !originalDomain ||
      !tripleMatches(originalDomain, canonicalBounds) || !step || !family ||
      family.getValue() != "tiling" || !lineageAxis ||
      lineageAxis.getInt() != 0 || !lineageFactor ||
      lineageFactor.getInt() != factor.getInt() || !lineagePart ||
      lineagePart.getInt() != part.getInt() ||
      !tripleMatches(before, canonicalBounds) ||
      !tripleMatches(after, currentBounds)) {
    error = "source-owned rank-1 typed partition lineage does not match its "
            "single tile step";
    return false;
  }
  return true;
}

static std::optional<unsigned> sourceOwnedNoAliasArgumentIndex(Value value) {
  FailureOr<Value> root = resolveTaskflowMemoryRoot(value);
  if (failed(root))
    return std::nullopt;
  value = *root;
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return std::nullopt;
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front() ||
      !function.getArgAttr(argument.getArgNumber(), "amoeba.noalias"))
    return std::nullopt;
  return argument.getArgNumber();
}

static bool validateSourceOwnedRankOneOutput(
    TaskflowTaskOp task, unsigned expectedArgument,
    ArrayRef<int64_t> expectedShape, bool requireMetadata,
    ReplicaOutputCoordinateProof &proof, std::string &error) {
  if (task.getWillWrites().size() != 1 ||
      task.getOriginalWriteMemrefs().size() != 1) {
    error = "source-owned rank-1 task must have one output and write root";
    return false;
  }
  std::optional<unsigned> argument =
      sourceOwnedNoAliasArgumentIndex(task.getOriginalWriteMemrefs().front());
  if (!argument || *argument != expectedArgument) {
    error = "source-owned rank-1 output no longer names the canonical "
            "noalias argument";
    return false;
  }
  proof = analyzeReplicaOutputCoordinates(task, /*requestedAxis=*/0,
                                          /*outputWriteIndex=*/0);
  if (!proof.proven || !proof.sawOutputLoad || !proof.sawOutputStore ||
      proof.outputCounterAxes.size() != 1 ||
      proof.outputCounterAxes.front() != 0 || proof.callerShape.size() != 1 ||
      proof.callerShape.front() <= 0 ||
      (!expectedShape.empty() &&
       !llvm::equal(proof.callerShape, expectedShape)) ||
      proof.taskLowers.size() != 1 || proof.taskUppers.size() != 1 ||
      proof.kernelLowers.size() != 1 || proof.kernelUppers.size() != 1 ||
      proof.producedLowers.size() != 1 || proof.producedShape.size() != 1 ||
      proof.taskLowers.front() < 0 ||
      proof.taskUppers.front() <= proof.taskLowers.front() ||
      proof.producedLowers.front() != proof.taskLowers.front() ||
      proof.producedShape.front() !=
          proof.taskUppers.front() - proof.taskLowers.front()) {
    error = "source-owned rank-1 output coordinates do not prove the actual "
            "counter interval: " +
            proof.reason;
    return false;
  }
  return validateOutputRegionAgainstProof(task, /*writeIndex=*/0, proof,
                                          requireMetadata, error);
}

static bool validateSourceOwnedRankOneJoin(const TaskMetadata &parent,
                                           ArrayRef<ReplicaRecord> records,
                                           const CounterBounds &parentBounds,
                                           unsigned outputArgument,
                                           std::string &error) {
  if (records.empty())
    return false;
  func::FuncOp function =
      records.front().child->getParentOfType<func::FuncOp>();
  if (!function) {
    error = "source-owned rank-1 partition has no enclosing function";
    return false;
  }
  DenseSet<Value> expectedStates;
  for (const ReplicaRecord &record : records) {
    TaskflowTaskOp child = record.child;
    if (child.getDoneWrites().size() != 1 ||
        !expectedStates.insert(child.getDoneWrites().front()).second) {
      error = "source-owned rank-1 partition has missing or duplicate tile "
              "completion states";
      return false;
    }
  }
  TaskflowJoinOp completionJoin;
  unsigned matchingJoins = 0;
  function.walk([&](TaskflowJoinOp candidate) {
    if (candidate.getTileStates().size() != expectedStates.size())
      return;
    DenseSet<Value> actualStates;
    for (Value state : candidate.getTileStates())
      if (!actualStates.insert(state).second || !expectedStates.contains(state))
        return;
    if (actualStates.size() == expectedStates.size()) {
      ++matchingJoins;
      completionJoin = candidate;
    }
  });
  if (matchingJoins != 1 || !completionJoin ||
      !completionJoin->hasAttr("amoeba.semantic.completion_only") ||
      completionJoin->hasAttr("amoeba.replica.completion_only") ||
      failed(verify(completionJoin.getOperation()))) {
    error = "source-owned rank-1 tiles lack one verified completion join "
            "linked to their exact done-write leaf set";
    return false;
  }
  auto rewrite =
      completionJoin->getAttrOfType<StringAttr>("amoeba.neura.joint_rewrite");
  if (!rewrite || rewrite.getValue() != "post-neura-mn-tiling" ||
      completionJoin.getAxis() != 0 ||
      completionJoin.getRegionLower().size() != 1 ||
      completionJoin.getRegionUpper().size() != 1 ||
      completionJoin.getRegionLower().front() != parentBounds.lower ||
      completionJoin.getRegionUpper().front() != parentBounds.upper) {
    error = "source-owned rank-1 completion join has a mismatched region or "
            "rewrite identity";
    return false;
  }
  std::optional<unsigned> joinArgument =
      sourceOwnedNoAliasArgumentIndex(completionJoin.getBase());
  std::optional<unsigned> parentArgument;
  TaskflowTaskOp parentTask = parent.op;
  if (parentTask.getOriginalWriteMemrefs().size() == 1)
    parentArgument = sourceOwnedNoAliasArgumentIndex(
        parentTask.getOriginalWriteMemrefs().front());
  if (!joinArgument || *joinArgument != outputArgument || !parentArgument ||
      *parentArgument != outputArgument) {
    error = "source-owned rank-1 completion join is not based on the "
            "canonical noalias output";
    return false;
  }
  for (Value state : expectedStates)
    if (!llvm::hasSingleElement(state.getUses()) ||
        state.use_begin()->getOwner() != completionJoin.getOperation()) {
      error = "source-owned rank-1 tile completion state escapes its exact "
              "join";
      return false;
    }
  return true;
}

static bool validateSourceOwnedRankOnePartition(const TaskMetadata &parent,
                                                ArrayRef<ReplicaRecord> records,
                                                std::string &error) {
  TaskflowTaskOp parentTask = parent.op;
  if (records.size() < 2 || parentTask.getWillWrites().size() != 1 ||
      parentTask.getOriginalWriteMemrefs().size() != 1) {
    error = "source-owned rank-1 partition requires at least two tiles and "
            "one canonical output";
    return false;
  }
  std::optional<unsigned> outputArgument = sourceOwnedNoAliasArgumentIndex(
      parentTask.getOriginalWriteMemrefs().front());
  if (!outputArgument) {
    error = "canonical source rank-1 output is not a noalias function "
            "argument";
    return false;
  }
  ReplicaOutputCoordinateProof parentProof;
  if (!validateSourceOwnedRankOneOutput(
          parentTask, *outputArgument, /*expectedShape=*/{},
          /*requireMetadata=*/false, parentProof, error))
    return false;
  CounterBounds parentBounds{parentProof.taskLowers.front(),
                             parentProof.taskUppers.front(), 1};
  int64_t parentExtent = parentBounds.upper - parentBounds.lower;
  if (parent.tripCount != parentExtent ||
      parent.taskflowTripCount != parentExtent) {
    error = "canonical source task " + parent.name +
            " trip count does not match its actual rank-1 counter extent";
    return false;
  }
  int64_t factor = -1;
  SmallVector<std::pair<int64_t, int64_t>> intervals;
  std::set<int64_t> seenParts;
  for (const ReplicaRecord &record : records) {
    TaskflowTaskOp child = record.child;
    if (!record.sourceOwnedTiling || record.sequentialK ||
        child.getWillWrites().size() != 1) {
      error = "source-owned rank-1 partition is mixed with a non-tile child";
      return false;
    }
    for (NamedAttribute attribute : child->getAttrs())
      if (attribute.getName().strref().starts_with("amoeba.replica.")) {
        error = "source-owned rank-1 pure tiling rejects replica metadata";
        return false;
      }
    ReplicaOutputCoordinateProof childProof;
    if (!validateSourceOwnedRankOneOutput(
            child, *outputArgument, parentProof.callerShape,
            /*requireMetadata=*/true, childProof, error)) {
      error = "source-owned rank-1 tile output proof failed: " + error;
      return false;
    }
    CounterBounds childBounds{childProof.taskLowers.front(),
                              childProof.taskUppers.front(), 1};
    if (childBounds.lower < parentBounds.lower ||
        childBounds.upper > parentBounds.upper) {
      error = "source-owned rank-1 tile has invalid actual counter bounds: " +
              error;
      return false;
    }
    auto childFactor =
        child->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.factor");
    auto childPart =
        child->getAttrOfType<IntegerAttr>("amoeba.neura.tiling.part_index");
    if (!childFactor || !childPart ||
        (factor >= 0 && factor != childFactor.getInt()) ||
        !seenParts.insert(childPart.getInt()).second) {
      error = "source-owned rank-1 tiles have inconsistent factors or "
              "duplicate part indices";
      return false;
    }
    factor = childFactor.getInt();
    if (!validateSourceOwnedRankOneTilingMetadata(
            child, parent.name, parentBounds, childBounds, error))
      return false;
    int64_t childExtent = childBounds.upper - childBounds.lower;
    if (record.childTripCount != childExtent) {
      error = "source-owned rank-1 tile trip count does not match its actual "
              "counter extent";
      return false;
    }
    intervals.push_back({childBounds.lower, childBounds.upper});
  }
  if (factor != static_cast<int64_t>(records.size()) ||
      static_cast<int64_t>(seenParts.size()) != factor) {
    error = "source-owned rank-1 tile family is incomplete";
    return false;
  }
  for (int64_t part = 0; part < factor; ++part)
    if (seenParts.find(part) == seenParts.end()) {
      error = "source-owned rank-1 tile family is missing a part";
      return false;
    }
  llvm::sort(intervals);
  int64_t cursor = parentBounds.lower;
  for (auto [lower, upper] : intervals) {
    if (lower != cursor || upper <= lower) {
      error = "source-owned rank-1 tiles overlap or leave a gap in actual "
              "counter bounds";
      return false;
    }
    cursor = upper;
  }
  if (cursor != parentBounds.upper) {
    error = "source-owned rank-1 tiles do not cover the canonical counter "
            "interval";
    return false;
  }
  return validateSourceOwnedRankOneJoin(parent, records, parentBounds,
                                        *outputArgument, error);
}

static bool validateSourceOwnedPartition(const TaskMetadata &parent,
                                         ArrayRef<ReplicaRecord> records,
                                         std::string &error) {
  if (records.empty()) {
    error = "source-owned nested partition is empty";
    return false;
  }
  TaskflowTaskOp parentTask = parent.op;
  if (parentTask.getBody().hasOneBlock()) {
    unsigned taskflowCounterCount = 0;
    for (Operation &operation : parentTask.getBody().front())
      taskflowCounterCount += isa<TaskflowCounterOp>(operation);
    if (taskflowCounterCount == 1)
      return validateSourceOwnedRankOnePartition(parent, records, error);
  }
  SmallVector<int64_t> parentCounterIds;
  if (!collectCounterIds(parentTask, parentCounterIds, error) ||
      parentCounterIds.size() < 2 || parentCounterIds.size() > 3) {
    if (error.empty())
      error = "canonical source task " + parent.name +
              " must have two M/N counters or three M/N/K counters";
    return false;
  }
  const size_t rank = parentCounterIds.size();
  SmallVector<CounterBounds, 3> parentBounds(rank);
  SmallVector<neura::CounterOp> parentNeuraCounters;
  parentTask.walk([&](neura::CounterOp counter) {
    parentNeuraCounters.push_back(counter);
  });
  if (parentNeuraCounters.size() != rank) {
    error = "canonical source task " + parent.name +
            " must retain exactly its M/N[/K] Neura counters";
    return false;
  }
  for (size_t axis = 0; axis < rank; ++axis) {
    CounterBounds neuraBounds;
    if (!findTaskflowCounterBounds(parentTask, axis, parentBounds[axis], error) ||
        !findNeuraCounterBounds(parentTask, axis, neuraBounds, error) ||
        !equalBounds(parentBounds[axis], neuraBounds) ||
        parentBounds[axis].step != 1) {
      error = "canonical source task " + parent.name +
              " has invalid or drifting M/N[/K] counter bounds";
      return false;
    }
  }
  int64_t parentVolume = 0;
  if (!checkedCounterVolume(parentBounds, parentVolume) ||
      parent.tripCount != parentVolume) {
    error = "canonical source task " + parent.name +
            " trip count does not match its full M/N[/K] counter volume";
    return false;
  }

  SmallVector<SourceOwnedPartitionBox> boxes;
  boxes.reserve(records.size());
  __int128 volume = 0;
  for (const ReplicaRecord &record : records) {
    SourceOwnedPartitionBox box;
    if (!validateSourceOwnedTaskBox(parent, record, parentBounds, box, error))
      return false;
    int64_t childVolume = 0;
    if (!checkedCounterVolume(box.bounds, childVolume)) {
      error = "source-owned nested child has an invalid or overflowing full "
              "M/N[/K] counter volume";
      return false;
    }
    volume += static_cast<__int128>(childVolume);
    if (volume > static_cast<__int128>(parentVolume)) {
      error = "source-owned nested child volumes exceed the canonical parent";
      return false;
    }
    boxes.push_back(std::move(box));
  }
  for (size_t lhs = 0; lhs < boxes.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < boxes.size(); ++rhs) {
      bool overlap = true;
      for (size_t axis = 0; axis < rank; ++axis)
        if (boxes[lhs].bounds[axis].lower >= boxes[rhs].bounds[axis].upper ||
            boxes[rhs].bounds[axis].lower >= boxes[lhs].bounds[axis].upper) {
          overlap = false;
          break;
        }
      if (overlap) {
        error = "source-owned nested partition has overlapping actual "
                "M/N[/K] counter boxes";
        return false;
      }
    }
  if (volume != static_cast<__int128>(parentVolume)) {
    error = "source-owned nested partition actual M/N[/K] boxes do not "
            "cover the canonical parent";
    return false;
  }
  TaskflowTaskOp canonicalTask = parent.op;
  // A replica followed by tiling leaves the historical replica attributes on
  // each narrower tile. Classify the proof from the authenticated ledger
  // families, not from the final task's attributes: mixed join validation
  // checks the historical shard step and the current leaf boxes separately.
  bool hasLineageTiling = false;
  bool hasLineageReplica = false;
  for (const ReplicaRecord &record : records) {
    if (!record.child->hasAttr(kNeighborhoodPartitionLineageAttr))
      continue;
    SmallVector<SourceOwnedLineageStep> steps;
    if (!decodeSourceOwnedLineageSteps(record.child, parent.name,
                                       parentBounds, steps, error))
      return false;
    for (const SourceOwnedLineageStep &step : steps) {
      hasLineageTiling |= step.family == "tiling";
      hasLineageReplica |= step.family == "replica";
    }
  }
  if (hasLineageTiling && hasLineageReplica)
    return validateSourceOwnedMixedJoinTree(
        parent.name, parentBounds, records, boxes,
        canonicalTask.getDoneReads().size(), error);
  if (!validateSourceOwnedNestedJoins(
          records, boxes, canonicalTask.getDoneReads().size(), error))
    return false;
  return true;
}

static bool isIgnoredCounterAttr(StringRef name) {
  return name == "counter_dynamism" || name == "lower_bound_value" ||
         name == "upper_bound_value" || name == "step_value" ||
         name == "operandSegmentSizes" || name == "operand_segment_sizes";
}

static bool isSourceOwnedTilingAttr(StringRef name) {
  return name == "amoeba.neura.joint_rewrite" ||
         name.starts_with("amoeba.neura.tiling.") ||
         name == kNeighborhoodPartitionLineageAttr;
}

static bool
attrsEquivalent(Operation *lhs, Operation *rhs, bool taskAttrs,
                bool counterAttrs, bool sequentialK = false,
                bool sourceOwnedTiling = false,
                bool verifiedOriginalAmoebaRealization = false,
                SmallVectorImpl<std::string> *mismatchedAttrs = nullptr) {
  auto recordMismatch = [&](StringRef name) {
    if (mismatchedAttrs && !llvm::is_contained(*mismatchedAttrs, name.str()))
      mismatchedAttrs->push_back(name.str());
  };
  auto isVerifiedDerivedMetadata = [&](StringRef name) {
    if (!verifiedOriginalAmoebaRealization)
      return false;
    if (name == kOriginalAmoebaDecisionModeAttr ||
        name == "amoeba.original_amoeba.replica_decision" ||
        name == "amoeba.original_amoeba.replica_placements" ||
        name == "amoeba.original_amoeba.source_partition_realization")
      return true;
    // The native materializer removes these parent-firing summaries from
    // verified children so they cannot masquerade as per-shard profile data.
    // Permit only that validated parent-present/child-absent relationship;
    // changed or child-invented summaries still compare exactly.
    return (name == "profile_info" || name == "task_orchestration_info") &&
           lhs->hasAttr(name) && !rhs->hasAttr(name);
  };
  for (NamedAttribute left : lhs->getAttrs()) {
    StringRef name = left.getName().getValue();
    if ((taskAttrs && isIgnoredTaskAttr(name)) ||
        (counterAttrs && isIgnoredCounterAttr(name)) ||
        (sequentialK && isSequentialKDerivedTaskAttr(name)) ||
        (sourceOwnedTiling && isSourceOwnedTilingAttr(name)) ||
        isVerifiedDerivedMetadata(name))
      continue;
    Attribute right = rhs->getAttr(name);
    if (!right || right != left.getValue()) {
      recordMismatch(name);
      return false;
    }
  }
  for (NamedAttribute right : rhs->getAttrs()) {
    StringRef name = right.getName().getValue();
    if ((taskAttrs && isIgnoredTaskAttr(name)) ||
        (counterAttrs && isIgnoredCounterAttr(name)) ||
        (sequentialK && isSequentialKDerivedTaskAttr(name)) ||
        (sourceOwnedTiling && isSourceOwnedTilingAttr(name)) ||
        isVerifiedDerivedMetadata(name))
      continue;
    Attribute left = lhs->getAttr(name);
    if (!left || left != right.getValue()) {
      recordMismatch(name);
      return false;
    }
  }
  return true;
}

static std::string conciseAttributeText(Attribute attribute) {
  if (!attribute)
    return "<absent>";
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  stream.flush();
  constexpr size_t kMaxAttributeTextLength = 160;
  if (text.size() > kMaxAttributeTextLength) {
    text.resize(kMaxAttributeTextLength);
    text += "...";
  }
  return text;
}

static bool sourceOwnedTypesCompatible(Type lhs, Type rhs,
                                       bool sourceOwnedTiling) {
  if (lhs == rhs)
    return true;
  if (!sourceOwnedTiling)
    return false;
  if (auto left = dyn_cast<neura::PredicatedValue>(lhs)) {
    auto right = dyn_cast<neura::PredicatedValue>(rhs);
    return right && left.getPredicateType() == right.getPredicateType() &&
           sourceOwnedTypesCompatible(left.getValueType(), right.getValueType(),
                                       /*sourceOwnedTiling=*/true);
  }
  auto leftMemref = dyn_cast<MemRefType>(lhs);
  auto rightMemref = dyn_cast<MemRefType>(rhs);
  if (!leftMemref || !rightMemref)
    return false;
  return memref::CastOp::areCastCompatible({lhs}, {rhs});
}

static bool isBoundProducer(Operation *operation,
                            const DenseSet<Operation *> &boundUsers) {
  if (!operation || operation->getNumResults() != 1 ||
      !operation->getRegions().empty())
    return false;
  if (!isa<arith::ConstantOp, arith::ConstantIndexOp, neura::ConstantOp>(
          operation))
    return false;
  for (Value result : operation->getResults())
    for (OpOperand &use : result.getUses())
      if (!boundUsers.contains(use.getOwner()))
        return false;
  return true;
}

static DenseSet<Operation *> collectBoundProducers(Operation *kernel,
                                                   bool sourceOwnedTiling) {
  DenseSet<Operation *> counterUsers;
  kernel->walk([&](Operation *operation) {
    if (isa<neura::CounterOp>(operation) ||
        (sourceOwnedTiling && isa<TaskflowCounterOp>(operation)))
      counterUsers.insert(operation);
  });
  DenseSet<Operation *> ignored;
  kernel->walk([&](Operation *operation) {
    if (!isa<neura::CounterOp>(operation) &&
        !(sourceOwnedTiling && isa<TaskflowCounterOp>(operation)))
      return;
    for (Value operand : operation->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (isBoundProducer(def, counterUsers))
          ignored.insert(def);
  });
  if (sourceOwnedTiling)
    kernel->walk([&](Operation *operation) {
      // The source-owned post-Neura cloner can leave dead literal bounds in
      // the kernel block after replacing a counter's interval.  They carry
      // no DFG meaning; ignore them only when every remaining use is a
      // counter (or there is no use at all), preserving semantic literals.
      if (isa<arith::ConstantOp, arith::ConstantIndexOp, neura::ConstantOp>(
              operation) &&
          isBoundProducer(operation, counterUsers))
        ignored.insert(operation);
    });
  return ignored;
}

static int64_t countOperations(Operation *root) {
  int64_t count = 0;
  root->walk([&](Operation *) { ++count; });
  return count;
}

static int64_t countCounters(Operation *root) {
  int64_t count = 0;
  root->walk([&](neura::CounterOp) { ++count; });
  return count;
}

static bool compareRegion(Region &lhs, Region &rhs,
                          DenseMap<Value, Value> &mapping,
                          const DenseSet<Operation *> &ignoredLhs,
                          const DenseSet<Operation *> &ignoredRhs,
                          std::string &reason,
                          bool sourceOwnedTiling = false);

static bool compareOperation(Operation *lhs, Operation *rhs,
                             DenseMap<Value, Value> &mapping,
                             const DenseSet<Operation *> &ignoredLhs,
                             const DenseSet<Operation *> &ignoredRhs,
                             bool rootKernel, std::string &reason,
                             bool sourceOwnedTiling = false) {
  // Counter operands are optional index bounds. The mapper-visible counter
  // data result is identical when bounds move from attributes to literal SSA
  // operands during sharding; neither spelling is a DFG data input.
  bool counter = isa<neura::CounterOp>(lhs) ||
                 (sourceOwnedTiling && isa<TaskflowCounterOp>(lhs));
  if (lhs->getName() != rhs->getName() ||
      (!counter && lhs->getNumOperands() != rhs->getNumOperands()) ||
      lhs->getNumResults() != rhs->getNumResults() ||
      lhs->getNumRegions() != rhs->getNumRegions() ||
      lhs->getNumSuccessors() != rhs->getNumSuccessors()) {
    reason = "Neura body operation shape changed";
    return false;
  }
  bool rhsCounter = isa<neura::CounterOp>(rhs) ||
                    (sourceOwnedTiling && isa<TaskflowCounterOp>(rhs));
  if (counter != rhsCounter) {
    reason = "Neura counter operation changed";
    return false;
  }
  SmallVector<std::string, 4> mismatchedAttrs;
  if (!attrsEquivalent(lhs, rhs, /*taskAttrs=*/false, counter,
                       /*sequentialK=*/false,
                       /*sourceOwnedTiling=*/false,
                       /*verifiedOriginalAmoebaRealization=*/false,
                       &mismatchedAttrs)) {
    reason = "Neura body semantic attributes changed on " +
             lhs->getName().getStringRef().str();
    constexpr size_t kMaxReportedAttributeDifferences = 8;
    for (auto [index, name] : llvm::enumerate(mismatchedAttrs)) {
      if (index == kMaxReportedAttributeDifferences) {
        reason += "; additional attribute differences omitted";
        break;
      }
      reason += "; ";
      reason += name;
      reason += " source=";
      reason += conciseAttributeText(lhs->getAttr(name));
      reason += " child=";
      reason += conciseAttributeText(rhs->getAttr(name));
    }
    return false;
  }
  if (counter) {
    for (Operation *operation : {lhs, rhs})
      for (Value operand : operation->getOperands())
        if (!operand.getType().isIndex()) {
          reason = "Neura counter has a non-bound data operand";
          return false;
        }
  }
  for (auto [index, pair] : llvm::enumerate(llvm::zip(lhs->getOperands(),
                                                        rhs->getOperands()))) {
    Value left = std::get<0>(pair);
    Value right = std::get<1>(pair);
    if (!sourceOwnedTypesCompatible(left.getType(), right.getType(),
                                    sourceOwnedTiling)) {
      reason = "Neura body operand type changed";
      return false;
    }
    if (counter)
      continue; // Bounds may change only for a proven counter operation.
    auto mapped = mapping.find(left);
    if (mapped == mapping.end() || mapped->second != right) {
      reason = "Neura body SSA dataflow changed";
      return false;
    }
    (void)index;
  }
  for (auto pair : llvm::zip(lhs->getResults(), rhs->getResults())) {
    Value left = std::get<0>(pair);
    Value right = std::get<1>(pair);
    if (!sourceOwnedTypesCompatible(left.getType(), right.getType(),
                                    sourceOwnedTiling)) {
      reason = "Neura body result type changed";
      return false;
    }
    mapping[left] = right;
  }
  for (auto pair : llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!compareRegion(std::get<0>(pair), std::get<1>(pair), mapping,
                       ignoredLhs, ignoredRhs, reason, sourceOwnedTiling))
      return false;
  (void)rootKernel;
  return true;
}

static bool compareRegion(Region &lhs, Region &rhs,
                          DenseMap<Value, Value> &mapping,
                          const DenseSet<Operation *> &ignoredLhs,
                          const DenseSet<Operation *> &ignoredRhs,
                          std::string &reason, bool sourceOwnedTiling) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    reason = "Neura body block structure changed";
    return false;
  }
  for (auto pair : llvm::zip(lhs, rhs)) {
    Block &leftBlock = std::get<0>(pair);
    Block &rightBlock = std::get<1>(pair);
    if (leftBlock.getNumArguments() != rightBlock.getNumArguments()) {
      reason = "Neura body block arguments changed";
      return false;
    }
    for (auto args : llvm::zip(leftBlock.getArguments(),
                               rightBlock.getArguments())) {
      Value left = std::get<0>(args);
      Value right = std::get<1>(args);
      if (!sourceOwnedTypesCompatible(left.getType(), right.getType(),
                                      sourceOwnedTiling)) {
        reason = "Neura body block argument type changed";
        return false;
      }
      mapping[left] = right;
    }
    auto leftIt = leftBlock.begin();
    auto rightIt = rightBlock.begin();
    while (true) {
      while (leftIt != leftBlock.end() && ignoredLhs.contains(&*leftIt))
        ++leftIt;
      while (rightIt != rightBlock.end() && ignoredRhs.contains(&*rightIt))
        ++rightIt;
      if (leftIt == leftBlock.end() || rightIt == rightBlock.end())
        break;
      if (!compareOperation(&*leftIt, &*rightIt, mapping, ignoredLhs,
                            ignoredRhs, /*rootKernel=*/false, reason,
                            sourceOwnedTiling))
        return false;
      ++leftIt;
      ++rightIt;
    }
    if (leftIt != leftBlock.end() || rightIt != rightBlock.end()) {
      reason = "Neura body semantic operation was added or removed";
      return false;
    }
  }
  return true;
}

static bool compareKernel(neura::KernelOp lhs, neura::KernelOp rhs,
                          BodyProof &proof, std::string &reason,
                          bool sourceOwnedTiling = false) {
  DenseSet<Operation *> ignoredLhs =
      collectBoundProducers(lhs, sourceOwnedTiling);
  DenseSet<Operation *> ignoredRhs =
      collectBoundProducers(rhs, sourceOwnedTiling);
  DenseMap<Value, Value> mapping;
  if (lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      !attrsEquivalent(lhs, rhs, /*taskAttrs=*/false, /*counterAttrs=*/false)) {
    reason = "Neura kernel inputs, outputs, or attributes changed";
    return false;
  }
  for (auto pair : llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    Value left = std::get<0>(pair);
    Value right = std::get<1>(pair);
    if (!sourceOwnedTypesCompatible(left.getType(), right.getType(),
                                    sourceOwnedTiling)) {
      reason = "Neura kernel input type changed";
      return false;
    }
    mapping[left] = right;
  }
  for (auto pair : llvm::zip(lhs->getResults(), rhs->getResults())) {
    Value left = std::get<0>(pair);
    Value right = std::get<1>(pair);
    if (!sourceOwnedTypesCompatible(left.getType(), right.getType(),
                                    sourceOwnedTiling)) {
      reason = "Neura kernel output type changed";
      return false;
    }
    mapping[left] = right;
  }
  // The kernel operation itself owns the first region. Its block arguments
  // and all nested semantic operations are checked by the same mapping.
  if (!compareRegion(lhs.getBody(), rhs.getBody(), mapping, ignoredLhs,
                     ignoredRhs, reason, sourceOwnedTiling))
    return false;
  proof.parentOperations += countOperations(lhs);
  proof.childOperations += countOperations(rhs);
  proof.parentCounters += countCounters(lhs);
  proof.childCounters += countCounters(rhs);
  return true;
}

static bool proveTaskBody(TaskflowTaskOp parent, TaskflowTaskOp child,
                          BodyProof &proof, std::string &error,
                          bool sequentialK = false,
                          bool sourceOwnedTiling = false,
                          bool verifiedOriginalAmoebaRealization = false) {
  if (parent->getNumOperands() != child->getNumOperands() ||
      parent->getNumResults() != child->getNumResults() ||
      !attrsEquivalent(parent, child, /*taskAttrs=*/true,
                       /*counterAttrs=*/false, sequentialK,
                       sourceOwnedTiling,
                       verifiedOriginalAmoebaRealization)) {
    error = "task shell inputs, outputs, or semantic attributes changed";
    return false;
  }
  for (auto pair : llvm::zip(parent->getOperands(), child->getOperands()))
    if (!sourceOwnedTypesCompatible(std::get<0>(pair).getType(),
                                    std::get<1>(pair).getType(),
                                    sourceOwnedTiling)) {
      error = "task shell operand type changed";
      return false;
    }
  for (auto pair : llvm::zip(parent->getResults(), child->getResults()))
    if (!sourceOwnedTypesCompatible(std::get<0>(pair).getType(),
                                    std::get<1>(pair).getType(),
                                    sourceOwnedTiling)) {
      error = "task shell result type changed";
      return false;
    }
  SmallVector<neura::KernelOp> parentKernels;
  SmallVector<neura::KernelOp> childKernels;
  parent.walk([&](neura::KernelOp kernel) { parentKernels.push_back(kernel); });
  child.walk([&](neura::KernelOp kernel) { childKernels.push_back(kernel); });
  if (parentKernels.empty() || childKernels.empty()) {
    error = "mapper-visible Neura body proof is unavailable";
    return false;
  }
  proof.parentKernels = parentKernels.size();
  proof.childKernels = childKernels.size();
  for (auto pair : llvm::zip(parentKernels, childKernels))
    if (!compareKernel(std::get<0>(pair), std::get<1>(pair), proof, error,
                       sourceOwnedTiling))
      return false;
  if (parentKernels.size() != childKernels.size() ||
      proof.parentCounters != proof.childCounters) {
    error = "mapper-visible Neura kernel or counter structure changed";
    return false;
  }
  proof.equal = true;
  return true;
}

static bool isGitCommit(StringRef value) {
  return value.size() >= 7 && value.size() <= 64 &&
         llvm::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f') ||
                  (character >= 'A' && character <= 'F');
         });
}

static bool rejectMeasuredFields(const json::Object &object, StringRef where,
                                 std::string &error) {
  for (const auto &item : object) {
    StringRef key = item.first;
    if (key.contains_insensitive("measured") ||
        key.contains_insensitive("mapper_cost") || key == "native_cycles" ||
        key == "mapper_cycles") {
      error = "measured mapper cost field is forbidden in " + where.str() +
              ": " + key.str();
      return false;
    }
  }
  return true;
}

static bool copyJsonObject(const json::Object &source, json::Object &target) {
  for (const auto &field : source)
    target[field.first] = field.second;
  return true;
}

static std::optional<double> predictedDuration(const json::Object &entry,
                                               int64_t tripCount,
                                               std::string &error) {
  if (entry.getString("support_status") != std::optional<StringRef>("supported"))
    return std::nullopt;
  auto ii = entry.getNumber("predicted_ii");
  auto startup = entry.getNumber("startup_cycles");
  if (!ii || !startup || !std::isfinite(*ii) || !std::isfinite(*startup) ||
      *ii <= 0.0 || *startup <= 0.0 || tripCount <= 0) {
    error = "supported parent cost entry has invalid II/startup or child "
            "trip count";
    return std::nullopt;
  }
  long double duration = static_cast<long double>(*startup) +
                         static_cast<long double>(*ii) *
                             static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(std::numeric_limits<double>::max())) {
    error = "child replica duration is outside the finite ranker range";
    return std::nullopt;
  }
  return static_cast<double>(duration);
}

static bool validateModelMetadata(const json::Object &metadata,
                                  StringRef namespaceValue,
                                  StringRef expectedNamespace, StringRef model,
                                  StringRef modelSchema,
                                  StringRef featureContract,
                                  StringRef featureExtractor,
                                  std::string &error) {
  auto provenance = requiredString(metadata, "provenance_schema", error);
  auto actualModel = requiredString(metadata, "model", error);
  auto actualSchema = requiredString(metadata, "model_schema", error);
  auto actualFeature = requiredString(metadata, "feature_contract_id", error);
  auto actualExtractor = requiredString(metadata, "feature_extractor", error);
  auto modelStatus = requiredString(metadata, "model_status", error);
  auto qualityStatus = requiredString(metadata, "quality_status", error);
  auto productionReady = requiredBoolean(metadata, "production_ready", error);
  if (!provenance || !actualModel || !actualSchema || !actualFeature ||
      !actualExtractor || !modelStatus || !qualityStatus || !productionReady)
    return false;
  if (*provenance != kCostProvenanceSchema || namespaceValue.empty() ||
      namespaceValue != expectedNamespace || *actualModel != model ||
      *actualSchema != modelSchema || *actualFeature != featureContract ||
      *actualExtractor != featureExtractor || *productionReady) {
    error = "replica inheritance requires the exact exploratory ML model "
            "contract and production_ready=false";
    return false;
  }
  return true;
}

static bool parseParentCatalogue(
    StringRef path, StringRef expectedFunction,
    ArrayRef<TaskMetadata> parentTasks, StringRef sourceRepository,
    StringRef sourceCommit, StringRef architecturePath, StringRef parentGraph,
    StringRef expectedNamespace, StringRef expectedModel,
    StringRef expectedModelSchema, StringRef expectedFeatureContract,
    StringRef expectedExtractor, json::Object &root,
    std::map<ShapeKey, const json::Object *> &entries, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read parent cost catalogue " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "invalid parent cost catalogue JSON: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  if (!rejectSha256(*parsed, "parent cost catalogue", error))
    return false;
  json::Object *parsedRoot = parsed->getAsObject();
  if (!parsedRoot) {
    error = "parent cost catalogue must be a JSON object";
    return false;
  }
  auto schema = requiredString(*parsedRoot, "schema", error);
  auto function = requiredString(*parsedRoot, "function", error);
  auto namespaceValue = requiredString(*parsedRoot, "namespace", error);
  json::Object *metadata = parsedRoot->getObject("predictor_metadata");
  json::Array *rawEntries = parsedRoot->getArray("entries");
  if (!schema || !function || !namespaceValue || !metadata || !rawEntries)
    return false;
  if (*schema != kCostSchema || *function != expectedFunction ||
      !rejectMeasuredFields(*parsedRoot, "parent catalogue", error) ||
      !validateModelMetadata(*metadata, *namespaceValue, expectedNamespace,
                             expectedModel, expectedModelSchema,
                             expectedFeatureContract, expectedExtractor,
                             error))
    return false;
  auto sourceGraph = requiredString(*metadata, "source_graph_id", error);
  auto sourceRepositoryValue =
      requiredString(*metadata, "source_repository", error);
  auto sourceCommitValue = requiredString(*metadata, "source_commit", error);
  auto architecture = requiredString(*metadata, "architecture_path", error);
  if (!sourceGraph || !sourceRepositoryValue || !sourceCommitValue ||
      !architecture || *sourceGraph != parentGraph ||
      *sourceRepositoryValue != sourceRepository ||
      *sourceCommitValue != sourceCommit ||
      !samePath(*architecture, architecturePath)) {
    error = "parent cost catalogue source identity does not match the "
            "requested canonical graph contract";
    return false;
  }
  TaskShapeCostCache cache;
  if (!cache.load(path, expectedFunction, parentTasks, sourceRepository,
                  sourceCommit, architecturePath, parentGraph, error))
    return false;
  std::set<std::string> parentIds;
  for (const TaskMetadata &task : parentTasks)
    parentIds.insert(task.name);
  for (json::Value &rawEntry : *rawEntries) {
    json::Object *entry = rawEntry.getAsObject();
    if (!entry || !rejectMeasuredFields(*entry, "parent cost entry", error))
      return false;
    auto task = requiredString(*entry, "task", error);
    auto rows = requiredInteger(*entry, "mapper_tile_rows", error);
    auto cols = requiredInteger(*entry, "mapper_tile_cols", error);
    auto status = requiredString(*entry, "support_status", error);
    if (std::optional<bool> productionReady =
            entry->getBoolean("production_ready");
        productionReady && *productionReady) {
      error = "parent cost entry is marked production_ready=true";
      return false;
    }
    if (!task || !rows || !cols || !status || *rows <= 0 || *cols <= 0 ||
        parentIds.find(task->str()) == parentIds.end())
      return false;
    ShapeKey key{task->str(), *rows, *cols};
    if (!entries.emplace(key, entry).second) {
      error = "parent cost catalogue contains duplicate task/shape entries";
      return false;
    }
  }
  if (entries.empty()) {
    error = "parent cost catalogue contains no entries";
    return false;
  }
  root = std::move(*parsedRoot);
  return true;
}

static bool parseReplicaInteger(TaskflowTaskOp task, StringRef attr,
                                int64_t &value, std::string &error) {
  auto integer = task->getAttrOfType<IntegerAttr>(attr);
  if (!integer) {
    error = "replica task " + task.getTaskName().str() + " is missing " +
            attr.str();
    return false;
  }
  value = integer.getInt();
  return true;
}

static bool parseIntegerAttr(TaskflowTaskOp task, StringRef attr,
                             int64_t &value, std::string &error) {
  auto integer = task->getAttrOfType<IntegerAttr>(attr);
  if (!integer) {
    error = "sequential-K task " + task.getTaskName().str() +
            " is missing " + attr.str();
    return false;
  }
  value = integer.getInt();
  return true;
}

static bool validateSequentialKShard(const TaskMetadata &parent,
                                     const TaskMetadata &child,
                                     ReplicaRecord &record,
                                     std::string &error) {
  TaskflowTaskOp task = child.op;
  if (!isSequentialKTask(task)) {
    error = "task " + child.name +
            " is not a proven sequential memory-feedback K shard";
    return false;
  }
  if (!task->hasAttr("amoeba.semantic.k_tiled")) {
    error = "sequential-K task " + child.name +
            " is missing amoeba.semantic.k_tiled";
    return false;
  }
  auto kind = task->getAttrOfType<StringAttr>("amoeba.semantic.task_kind");
  auto role = task->getAttrOfType<StringAttr>("amoeba.semantic.task_role");
  auto schema = task->getAttrOfType<StringAttr>("amoeba.semantic.schema");
  auto rewrite =
      task->getAttrOfType<StringAttr>("amoeba.semantic.rewrite_schema");
  auto visibility =
      task->getAttrOfType<StringAttr>("amoeba.semantic.output_visibility");
  if (!kind || kind.getValue() != "memory_feedback_producer" || !role ||
      role.getValue() != "producer" || !schema ||
      schema.getValue() != "orbit-joint-taskflow-materialization-v1" ||
      !rewrite || rewrite.getValue() != "orbit-joint-semantic-rewrite-v1" ||
      !visibility || visibility.getValue() != "global_materialized") {
    error = "sequential-K task " + child.name +
            " has an unsupported semantic materialization contract";
    return false;
  }
  if (!parseIntegerAttr(task, "amoeba.semantic.k_block_index", record.id,
                        error) ||
      !parseIntegerAttr(task, "amoeba.semantic.k_block_count", record.count,
                        error) ||
      record.count < 2 || record.id < 0 || record.id >= record.count) {
    if (error.empty())
      error = "sequential-K task " + child.name +
              " has an invalid block index/count";
    return false;
  }
  if (!readRange(task, "amoeba.semantic.K_range", record.kLower,
                 record.kUpper, error) ||
      !readRange(task, "amoeba.semantic.M_range", record.mLower,
                 record.mUpper, error) ||
      !readRange(task, "amoeba.semantic.N_range", record.nLower,
                 record.nUpper, error))
    return false;

  CounterBounds taskM, taskN, taskK;
  CounterBounds neuraM, neuraN, neuraK;
  if (!findTaskflowCounterBounds(task, 0, taskM, error) ||
      !findTaskflowCounterBounds(task, 1, taskN, error) ||
      !findTaskflowCounterBounds(task, 2, taskK, error) ||
      !findNeuraCounterBounds(task, 0, neuraM, error) ||
      !findNeuraCounterBounds(task, 1, neuraN, error) ||
      !findNeuraCounterBounds(task, 2, neuraK, error))
    return false;
  if (!equalBounds(taskM, neuraM) || !equalBounds(taskN, neuraN) ||
      !equalBounds(taskK, neuraK)) {
    error = "sequential-K task " + child.name +
            " has Taskflow/Neura counter bound drift";
    return false;
  }
  if (taskM.lower != record.mLower || taskM.upper != record.mUpper ||
      taskN.lower != record.nLower || taskN.upper != record.nUpper ||
      taskK.lower != record.kLower || taskK.upper != record.kUpper) {
    error = "sequential-K task " + child.name +
            " semantic ranges do not match actual counter bounds";
    return false;
  }

  CounterBounds parentM, parentN, parentK;
  CounterBounds parentNeuraM, parentNeuraN, parentNeuraK;
  if (!findTaskflowCounterBounds(parent.op, 0, parentM, error) ||
      !findTaskflowCounterBounds(parent.op, 1, parentN, error) ||
      !findTaskflowCounterBounds(parent.op, 2, parentK, error) ||
      !findNeuraCounterBounds(parent.op, 0, parentNeuraM, error) ||
      !findNeuraCounterBounds(parent.op, 1, parentNeuraN, error) ||
      !findNeuraCounterBounds(parent.op, 2, parentNeuraK, error))
    return false;
  if (!equalBounds(parentM, parentNeuraM) ||
      !equalBounds(parentN, parentNeuraN) ||
      !equalBounds(parentK, parentNeuraK)) {
    error = "canonical parent " + parent.name +
            " has Taskflow/Neura counter bound drift";
    return false;
  }
  if (parentM.lower != record.mLower || parentM.upper != record.mUpper ||
      parentN.lower != record.nLower || parentN.upper != record.nUpper) {
    error = "sequential-K task " + child.name +
            " changed canonical M/N bounds";
    return false;
  }
  int64_t expectedTrip = 0;
  if (!checkedTripProduct(taskM, taskN, taskK, expectedTrip) ||
      child.tripCount != expectedTrip) {
    error = "sequential-K task " + child.name +
            " trip count does not match its actual M/N/K counter ranges";
    return false;
  }
  int64_t parentTrip = 0;
  if (!checkedTripProduct(parentM, parentN, parentK, parentTrip) ||
      parent.tripCount != parentTrip) {
    error = "canonical parent " + parent.name +
            " trip count does not match its actual M/N/K counter ranges";
    return false;
  }
  if (record.id == 0) {
    auto initial = task->getAttrOfType<StringAttr>(
        "amoeba.semantic.initial_C_ownership");
    if (!initial || initial.getValue() != "initial_C_once") {
      error = "first sequential-K block does not own initial C exactly once";
      return false;
    }
  } else {
    auto initial = task->getAttrOfType<StringAttr>(
        "amoeba.semantic.initial_C_ownership");
    if (!initial || initial.getValue() != "state_carry") {
      error = "non-first sequential-K block does not declare state carry";
      return false;
    }
  }
  auto final = task->getAttrOfType<StringAttr>(
      "amoeba.semantic.final_C_ownership");
  if (!final || (final.getValue() != "state_carry" &&
                 final.getValue() != "external_final_C")) {
    error = "sequential-K block has invalid final C ownership";
    return false;
  }

  record.child = task;
  record.parent = parent.name;
  record.childTripCount = child.tripCount;
  record.sequentialK = true;
  return true;
}

static bool hasOnlyIncomingStateEdge(TaskflowTaskOp task, StringRef source) {
  auto edges = task->getAttrOfType<ArrayAttr>(
      "amoeba.semantic.incoming_edges");
  if (!edges)
    // Some prepared canonical modules do not carry the optional edge
    // annotation.  The required direct SSA state read is checked by the
    // caller; accept the absent annotation rather than inventing one.
    return true;
  std::string expected = (Twine(source) + "|state_carry|tile_local").str();
  unsigned stateEdges = 0;
  bool expectedFound = false;
  for (Attribute edge : edges) {
    auto text = dyn_cast<StringAttr>(edge);
    if (!text || !text.getValue().contains("|state_carry|"))
      continue;
    ++stateEdges;
    expectedFound |= text.getValue() == expected;
  }
  return stateEdges == 1 && expectedFound;
}

static bool hasAnyIncomingStateEdge(TaskflowTaskOp task) {
  auto edges = task->getAttrOfType<ArrayAttr>(
      "amoeba.semantic.incoming_edges");
  if (!edges)
    return false;
  return llvm::any_of(edges, [](Attribute edge) {
    auto text = dyn_cast<StringAttr>(edge);
    return text && text.getValue().contains("|state_carry|");
  });
}

static bool validateSequentialKStateGroup(
    const TaskMetadata &parent, ArrayRef<ReplicaRecord> records,
    std::string &error) {
  if (records.empty())
    return false;
  std::map<int64_t, const ReplicaRecord *> byIndex;
  int64_t count = records.front().count;
  for (const ReplicaRecord &record : records) {
    if (!record.sequentialK || record.count != count ||
        !byIndex.emplace(record.id, &record).second) {
      error = "sequential-K group has inconsistent block identity";
      return false;
    }
  }
  if (static_cast<int64_t>(byIndex.size()) != count) {
    error = "sequential-K group is missing one or more block indices";
    return false;
  }
  CounterBounds parentM, parentN, parentK;
  if (!findTaskflowCounterBounds(parent.op, 0, parentM, error) ||
      !findTaskflowCounterBounds(parent.op, 1, parentN, error) ||
      !findTaskflowCounterBounds(parent.op, 2, parentK, error))
    return false;
  if (auto range = parent.op->getAttrOfType<DenseI64ArrayAttr>(
          "amoeba.semantic.K_range");
      range && (range.size() != 2 || range[0] != parentK.lower ||
                range[1] != parentK.upper)) {
    error = "canonical parent semantic K range differs from actual bounds";
    return false;
  }
  if (auto range = parent.op->getAttrOfType<DenseI64ArrayAttr>(
          "amoeba.semantic.M_range");
      range && (range.size() != 2 || range[0] != parentM.lower ||
                range[1] != parentM.upper)) {
    error = "canonical parent semantic M range differs from actual bounds";
    return false;
  }
  if (auto range = parent.op->getAttrOfType<DenseI64ArrayAttr>(
          "amoeba.semantic.N_range");
      range && (range.size() != 2 || range[0] != parentN.lower ||
                range[1] != parentN.upper)) {
    error = "canonical parent semantic N range differs from actual bounds";
    return false;
  }
  int64_t kCursor = parentK.lower;
  for (int64_t index = 0; index < count; ++index) {
    auto found = byIndex.find(index);
    if (found == byIndex.end()) {
      error = "sequential-K group is missing block index " +
              std::to_string(index);
      return false;
    }
    const ReplicaRecord &record = *found->second;
    TaskflowTaskOp current = record.child;
    if (record.kLower != kCursor || record.kUpper <= record.kLower ||
        record.kUpper > parentK.upper) {
      error = "sequential-K group has a gap, overlap, or out-of-range K block";
      return false;
    }
    kCursor = record.kUpper;
    auto final = current->getAttrOfType<StringAttr>(
        "amoeba.semantic.final_C_ownership");
    if (!final || (index + 1 == count
                       ? final.getValue() != "external_final_C"
                       : final.getValue() != "state_carry")) {
      error = "sequential-K group has invalid final C ownership order";
      return false;
    }
    if (index == 0) {
      if (hasAnyIncomingStateEdge(current)) {
        error = "first sequential-K block unexpectedly consumes state carry";
        return false;
      }
      for (Value read : current.getWillReads()) {
        auto producer = read.getDefiningOp<TaskflowTaskOp>();
        if (!producer)
          continue;
        for (const auto &entry : byIndex)
          if (producer == entry.second->child) {
            error = "first sequential-K block consumes a K state producer";
            return false;
          }
      }
    } else {
      const ReplicaRecord &previous = *byIndex[index - 1];
      TaskflowTaskOp previousTask = previous.child;
      if (current.getWillReads().size() == 0 ||
          previousTask.getDoneWrites().size() != 1 ||
          current.getWillWrites().size() != 1 ||
          previousTask.getWillWrites().size() != 1 ||
          previousTask.getWillWrites().front() !=
              current.getWillWrites().front()) {
        error = "sequential-K state-carry storage dependency is invalid";
        return false;
      }
      unsigned stateReads = 0;
      for (Value read : current.getWillReads())
        if (read == previousTask.getDoneWrites().front()) {
          ++stateReads;
          if (read.getType() != previousTask.getDoneWrites().front().getType()) {
            error = "sequential-K state-carry type changed";
            return false;
          }
        }
      if (stateReads != 1 ||
          !hasOnlyIncomingStateEdge(current, previousTask.getTaskName())) {
        error = "sequential-K state-carry SSA or incoming-edge proof is missing";
        return false;
      }
      if (current.getOriginalWriteMemrefs().size() != 1 ||
          !llvm::is_contained(current.getOriginalReadMemrefs(),
                              current.getOriginalWriteMemrefs().front())) {
        error = "sequential-K state-carry storage alias is not explicit";
        return false;
      }
    }
  }
  if (kCursor != parentK.upper) {
    error = "sequential-K blocks do not cover the canonical K range";
    return false;
  }
  int64_t parentTrip = 0;
  if (!checkedTripProduct(parentM, parentN, parentK, parentTrip) ||
      parent.tripCount != parentTrip) {
    error = "canonical parent trip count is inconsistent with K group";
    return false;
  }
  int64_t tripSum = 0;
  for (const ReplicaRecord &record : records) {
    if (record.childTripCount > std::numeric_limits<int64_t>::max() - tripSum) {
      error = "sequential-K trip count overflows";
      return false;
    }
    tripSum += record.childTripCount;
  }
  if (tripSum != parent.tripCount) {
    error = "sequential-K shard trip counts do not partition canonical trip "
            "count";
    return false;
  }
  return true;
}

static bool collectReplicaLayout(
    ArrayRef<TaskMetadata> parentTasks, ArrayRef<TaskMetadata> childTasks,
    std::map<std::string, TaskMetadata> &parents,
    std::map<std::string, TaskMetadata> &children,
    std::map<std::string, std::vector<ReplicaRecord>> &replicas,
    std::map<std::string, bool> &sourceOwnedLineage,
    std::map<std::string, CompositeFusionRecord> &compositeFusions,
    std::string &error) {
  for (const TaskMetadata &task : parentTasks)
    if (!parents.emplace(task.name, task).second) {
      error = "parent canonical module repeats task " + task.name;
      return false;
    }
  for (const TaskMetadata &task : childTasks)
    if (!children.emplace(task.name, task).second) {
      error = "child module repeats task " + task.name;
      return false;
    }

  for (const TaskMetadata &task : childTasks) {
    bool hasFusionMetadata = false;
    for (NamedAttribute attribute : task.op->getAttrs())
      hasFusionMetadata |= attribute.getName().strref().starts_with(
          "amoeba.neura.fusion.");
    if (!hasFusionMetadata)
      continue;
    auto mode = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.fusion.mode");
    auto loads = task.op->getAttrOfType<IntegerAttr>(
        "amoeba.neura.fusion.eliminated_loads");
    auto stores = task.op->getAttrOfType<IntegerAttr>(
        "amoeba.neura.fusion.eliminated_stores");
    auto first = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.fusion.sibling_first");
    auto second = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.fusion.sibling_second");
    auto rewrite = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.joint_rewrite");
    if (!mode || !loads || !stores || !rewrite) {
      error = "composite source-domain proof requires exact post-Neura "
              "fusion metadata";
      return false;
    }
    CompositeFusionKind kind;
    std::string materializerMode;
    std::string firstName;
    std::string secondName;
    if (mode.getValue() == "sibling" &&
        rewrite.getValue() == "post-neura-sibling-fusion" &&
        loads.getInt() >= 0 && stores.getInt() >= 0 && first && second &&
        !first.getValue().empty() && !second.getValue().empty() &&
        first != second) {
      kind = CompositeFusionKind::Sibling;
      materializerMode = "sibling";
      firstName = first.getValue().str();
      secondName = second.getValue().str();
    } else if (mode.getValue() == "retained" &&
               rewrite.getValue() ==
                   "post-neura-producer-consumer-retained" &&
               ((loads.getInt() == 0 && stores.getInt() == 0) ||
                (loads.getInt() == 1 && stores.getInt() == 0)) &&
               !first && !second) {
      // Unlike sibling fusion, retained producer-consumer materialization
      // keeps the producer output store/completion; it may also share the
      // consumer's matched load. Its derived task name encodes the ordered
      // producer and consumer; canonical replay below proves the exact
      // load/store elimination counts and body.
      StringRef fusedName = task.name;
      size_t separator = fusedName.find(".fuse.");
      if (separator == StringRef::npos || separator == 0 ||
          separator + StringRef(".fuse.").size() >= fusedName.size() ||
          fusedName.find(".fuse.", separator + 1) != StringRef::npos) {
        error = "retained producer-consumer fusion name does not encode one "
                "ordered canonical parent pair";
        return false;
      }
      kind = CompositeFusionKind::ProducerConsumerRetained;
      materializerMode = "producer-consumer-retained";
      firstName = fusedName.take_front(separator).str();
      secondName = fusedName.drop_front(separator + StringRef(".fuse.").size())
                       .str();
    } else if (mode.getValue() == "forwarded" &&
               rewrite.getValue() ==
                   "post-neura-producer-consumer-forwarded" &&
               loads.getInt() == 1 && stores.getInt() == 1 && !first &&
               !second) {
      StringRef fusedName = task.name;
      size_t separator = fusedName.find(".fuse.");
      if (separator == StringRef::npos || separator == 0 ||
          separator + StringRef(".fuse.").size() >= fusedName.size() ||
          fusedName.find(".fuse.", separator + 1) != StringRef::npos) {
        error = "forwarded producer-consumer fusion name does not encode one "
                "ordered canonical parent pair";
        return false;
      }
      kind = CompositeFusionKind::ProducerConsumerForwarded;
      materializerMode = "producer-consumer-forwarded";
      firstName = fusedName.take_front(separator).str();
      secondName = fusedName.drop_front(separator + StringRef(".fuse.").size())
                       .str();
    } else {
      error = "composite source-domain proof supports only exact sibling or "
              "retained/forwarded producer-consumer fusion metadata";
      return false;
    }
    std::string expectedName =
        (Twine(firstName) + ".fuse." + secondName).str();
    if (task.name != expectedName) {
      error = "composite fusion task name disagrees with its ordered parent "
              "pair";
      return false;
    }
    for (NamedAttribute attribute : task.op->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if (name.starts_with("amoeba.neura.fusion.") &&
          name != "amoeba.neura.fusion.mode" &&
          name != "amoeba.neura.fusion.eliminated_loads" &&
          name != "amoeba.neura.fusion.eliminated_stores" &&
          name != "amoeba.neura.fusion.sibling_first" &&
          name != "amoeba.neura.fusion.sibling_second") {
        error = "composite source-domain proof rejects unknown fusion "
                "metadata";
        return false;
      }
    }
    if (kind != CompositeFusionKind::Sibling &&
        (task.op->hasAttr("amoeba.neura.fusion.sibling_first") ||
         task.op->hasAttr("amoeba.neura.fusion.sibling_second"))) {
      error = "producer-consumer fusion rejects sibling-only "
              "parent metadata";
      return false;
    }
    CompositeFusionRecord record{task.name, std::move(firstName),
                                 std::move(secondName),
                                 std::move(materializerMode)};
    if (!compositeFusions.emplace(task.name, std::move(record)).second) {
      error = "source-domain proof supports only one composite "
              "fusion";
      return false;
    }
  }

  if (!compositeFusions.empty()) {
    if (compositeFusions.size() != 1 ||
        childTasks.size() + 1 != parentTasks.size()) {
      error = "source-domain composite fusion supports one fused pair with "
              "all other canonical tasks unchanged";
      return false;
    }
    const CompositeFusionRecord &fusion = compositeFusions.begin()->second;
    auto first = parents.find(fusion.firstParent);
    auto second = parents.find(fusion.secondParent);
    if (first == parents.end() || second == parents.end() ||
        first == second || children.count(fusion.firstParent) ||
        children.count(fusion.secondParent) ||
        fusion.childName !=
            (fusion.firstParent + ".fuse." + fusion.secondParent)) {
      error = "composite fusion does not replace exactly two canonical parent "
              "tasks";
      return false;
    }
    auto fusedChild = children.find(fusion.childName);
    if (first->second.tripCount != second->second.tripCount ||
        fusedChild == children.end() ||
        fusedChild->second.tripCount != first->second.tripCount) {
      error = "composite fusion source tasks do not have one shared firing "
              "count";
      return false;
    }
    for (const auto &parent : parents) {
      if (parent.first == fusion.firstParent ||
          parent.first == fusion.secondParent)
        continue;
      auto child = children.find(parent.first);
      if (child == children.end() ||
          child->second.tripCount != parent.second.tripCount) {
        error = "composite fusion changed or dropped an unrelated canonical "
                "task";
        return false;
      }
      for (NamedAttribute attribute : child->second.op->getAttrs()) {
        StringRef name = attribute.getName().strref();
        if (name.starts_with("amoeba.replica.") ||
            name.starts_with("amoeba.tiling.") ||
            name.starts_with("amoeba.neura.tiling.") ||
            name.starts_with("amoeba.neura.fusion.")) {
          error = "composite fusion composition rejects transformed unrelated "
                  "tasks";
          return false;
        }
      }
    }
    sourceOwnedLineage.clear();
    return true;
  }

  struct TilingFamily {
    std::string root;
    int64_t factor = 0;
  };
  // Build lineage only from source-owned tiling attributes.  A nested
  // replica's parent name can refer to the replaced tile, so the surviving
  // sibling tile is the witness that maps that intermediate name back to the
  // canonical source task.
  std::map<std::string, TilingFamily> tilingFamilies;
  for (const TaskMetadata &task : childTasks) {
    if (!task.op->hasAttr("amoeba.neura.tiling.parent_task"))
      continue;
    if (!validateSourceOwnedTilingTask(task.op, error))
      return false;
    auto parent = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.tiling.parent_task");
    auto axis = task.op->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.axis");
    auto factor = task.op->getAttrOfType<IntegerAttr>(
        "amoeba.neura.tiling.factor");
    std::string family =
        (Twine(parent.getValue()) + ".tile." + Twine(axis.getInt()) + ".")
            .str();
    auto found = tilingFamilies.find(family);
    if (found == tilingFamilies.end()) {
      tilingFamilies.emplace(family,
                             TilingFamily{parent.getValue().str(),
                                          factor.getInt()});
    } else if (found->second.root != parent.getValue() ||
               found->second.factor != factor.getInt()) {
      error = "source-owned tiling family has inconsistent parent/factor";
      return false;
    }
  }
  auto resolveSourceOwnedFamily = [&](StringRef candidate,
                                      std::string &root) {
    bool matched = false;
    for (const auto &family : tilingFamilies) {
      if (!candidate.starts_with(family.first))
        continue;
      StringRef partText = candidate.drop_front(family.first.size());
      int64_t part = -1;
      if (partText.empty() || partText.getAsInteger(10, part) || part < 0 ||
          part >= family.second.factor)
        continue;
      if (matched && root != family.second.root)
        return false;
      root = family.second.root;
      matched = true;
    }
    return matched;
  };
  for (const TaskMetadata &task : childTasks) {
    auto replicaParent = task.op->getAttrOfType<StringAttr>(
        "amoeba.replica.parent_task");
    auto sequentialKParent = task.op->getAttrOfType<StringAttr>(
        "amoeba.tiling.parent_task");
    auto sourceTilingParent = task.op->getAttrOfType<StringAttr>(
        "amoeba.neura.tiling.parent_task");
    if (replicaParent && sequentialKParent &&
        replicaParent.getValue() != sequentialKParent.getValue()) {
      error = "task " + task.name +
              " has conflicting replica and sequential-K parent lineage";
      return false;
    }
    if (!replicaParent && !sequentialKParent && !sourceTilingParent) {
      auto parent = parents.find(task.name);
      if (parent == parents.end()) {
        error = "child task " + task.name +
                " is neither an unchanged parent task nor a proved replica";
        return false;
      }
      if (task.tripCount != parent->second.tripCount) {
        error = "unchanged task " + task.name + " changed trip count";
        return false;
      }
      sourceOwnedLineage[task.name] = false;
      continue;
    }
    StringRef declaredParent = sourceTilingParent
                                   ? sourceTilingParent.getValue()
                                   : (sequentialKParent
                                          ? sequentialKParent.getValue()
                                          : replicaParent.getValue());
    if (declaredParent.empty()) {
      error = "derived task " + task.name + " has an empty parent task";
      return false;
    }
    std::string parentName = declaredParent.str();
    bool sourceOwnedTiling = sourceTilingParent != nullptr;
    if (parents.find(parentName) == parents.end()) {
      std::string resolved;
      bool foundRoot = resolveSourceOwnedFamily(declaredParent, resolved);
      if (!foundRoot && !sequentialKParent &&
          (sourceTilingParent || replicaParent)) {
        std::optional<std::string> rootHint =
            sourceOwnedPartitionRootHint(task.op);
        if (rootHint && parents.find(*rootHint) != parents.end()) {
          resolved = std::move(*rootHint);
          foundRoot = true;
        }
      }
      if (!foundRoot || parents.find(resolved) == parents.end()) {
        error = "replica task " + task.name +
                " names an unknown parent task";
        return false;
      }
      parentName = std::move(resolved);
      sourceOwnedTiling = true;
    }
    if (sourceOwnedTiling && !hasOutputRegionProof(task.op, error))
      return false;
    sourceOwnedLineage[task.name] = sourceOwnedTiling;
    ReplicaRecord record;
    record.child = task.op;
    record.parent = parentName;
    record.sourceOwnedTiling = sourceOwnedTiling;
    record.childTripCount = task.tripCount;
    if (sequentialKParent) {
      record.sequentialK = true;
      if (!validateSequentialKShard(parents[record.parent], task, record,
                                    error))
        return false;
      replicas[record.parent].push_back(std::move(record));
      continue;
    }
    if (record.childTripCount <= 0) {
      error = "derived task " + task.name + " has an invalid trip count";
      return false;
    }
    if (!sourceOwnedTiling &&
        (!parseReplicaInteger(task.op, "amoeba.replica.id", record.id,
                              error) ||
         !parseReplicaInteger(task.op, "amoeba.replica.count", record.count,
                              error) ||
         record.id < 0 || record.count <= 0 || record.id >= record.count)) {
      if (error.empty())
        error = "replica identity/count is invalid for task " + task.name;
      return false;
    }
    if (sourceOwnedTiling) {
      // Nested replicas retain local ids, but those ids are nested under the
      // source-owned outer tile and therefore are not the ultimate group's
      // flat ids.
      if (replicaParent &&
          (!parseReplicaInteger(task.op, "amoeba.replica.id", record.id,
                                error) ||
           !parseReplicaInteger(task.op, "amoeba.replica.count", record.count,
                                error) ||
           record.id < 0 || record.count <= 0 || record.id >= record.count))
        return false;
      // Nested lineage authenticates shard_trip_count against the volume at
      // the latest replica step. A later tile reduces this task's current
      // trip count without changing that historical replica attribute.
    } else {
      if (!parseReplicaInteger(task.op, "amoeba.replica.shard_axis",
                              record.shardAxis, error) ||
          record.shardAxis < 0) {
        if (error.empty())
          error = "ordinary replica shard axis is invalid for task " +
                  task.name;
        return false;
      }
      if (auto outputAxis = task.op->getAttrOfType<IntegerAttr>(
              "amoeba.replica.output_shard_axis"))
        record.outputShardAxis = outputAxis.getInt();
      if (auto declared = task.op->getAttrOfType<IntegerAttr>(
              "amoeba.replica.total_trip_count");
          declared && declared.getInt() != parents[record.parent].tripCount) {
        error = "replica total trip count does not match its canonical parent";
        return false;
      }
      if (auto declared = task.op->getAttrOfType<IntegerAttr>(
              "amoeba.replica.shard_trip_count");
          declared && declared.getInt() != record.childTripCount) {
        error = "replica shard trip count does not match child metadata";
        return false;
      }
    }
    replicas[record.parent].push_back(std::move(record));
  }
  for (const auto &parent : parents) {
    auto child = children.find(parent.first);
    auto group = replicas.find(parent.first);
    bool hasReplicas = group != replicas.end() && !group->second.empty();
    if (child != children.end() && hasReplicas) {
      error = "canonical task " + parent.first +
              " remains alongside its replica set";
      return false;
    }
    if (child == children.end() && !hasReplicas) {
      error = "child graph dropped canonical task " + parent.first;
      return false;
    }
    if (!hasReplicas)
      continue;
    const std::vector<ReplicaRecord> &groupRecords = group->second;
    bool hasSequentialK = llvm::any_of(
        groupRecords, [](const ReplicaRecord &record) {
          return record.sequentialK;
        });
    if (hasSequentialK) {
      if (!llvm::all_of(groupRecords, [](const ReplicaRecord &record) {
            return record.sequentialK;
          }) ||
          !validateSequentialKStateGroup(parent.second, groupRecords, error))
        return false;
      continue;
    }
    bool hasSourceOwnedTiling = llvm::any_of(
        groupRecords, [](const ReplicaRecord &record) {
          return record.sourceOwnedTiling;
        });
    if (hasSourceOwnedTiling) {
      int64_t tripSum = 0;
      for (const ReplicaRecord &record : groupRecords) {
        if (record.childTripCount <= 0 ||
            tripSum > std::numeric_limits<int64_t>::max() -
                          record.childTripCount) {
          error = "source-owned nested partition has invalid trip counts";
          return false;
        }
        tripSum += record.childTripCount;
      }
      if (tripSum != parent.second.tripCount) {
        error = "source-owned nested partition trip counts do not cover "
                "the canonical source task";
        return false;
      }
      if (!validateSourceOwnedPartition(parent.second, groupRecords, error))
        return false;
      continue;
    }
    int64_t count = groupRecords.front().count;
    std::set<int64_t> ids;
    int64_t tripSum = 0;
    for (const ReplicaRecord &record : groupRecords) {
      if (record.count != count || !ids.insert(record.id).second ||
          tripSum > std::numeric_limits<int64_t>::max() - record.childTripCount) {
        error = "replica set has inconsistent count or duplicate IDs";
        return false;
      }
      tripSum += record.childTripCount;
    }
    for (int64_t id = 0; id < count; ++id)
      if (ids.find(id) == ids.end()) {
        error = "replica set is missing a shard ID";
        return false;
      }
    if (tripSum != parent.second.tripCount) {
      error = "replica shard trip counts do not partition canonical trip "
              "count";
      return false;
    }
    if (!validateOrdinaryReplicaGroup(parent.second, groupRecords, error))
      return false;
  }
  return true;
}

static bool exactSimpleSourceCounterDomain(
    TaskflowTaskOp task, const SourceIterationDomainInfo &domain,
    std::string &error) {
  if (!domain.complete || domain.internalMultiplicity != 1 ||
      domain.axes.empty()) {
    error = "composite fusion requires a complete taskflow-only source domain "
            "with no internal iteration multiplicity";
    return false;
  }
  SmallVector<TaskflowCounterOp> taskflowCounters;
  task.walk([&](TaskflowCounterOp counter) {
    taskflowCounters.push_back(counter);
  });
  SmallVector<neura::KernelOp> kernels;
  task.walk([&](neura::KernelOp kernel) { kernels.push_back(kernel); });
  if (taskflowCounters.size() != domain.axes.size() || kernels.size() != 1) {
    error = "composite fusion source must have exactly one Taskflow and one "
            "Neura counter for every certified domain axis";
    return false;
  }
  SmallVector<neura::CounterOp> neuraCounters;
  kernels.front().walk([&](neura::CounterOp counter) {
    neuraCounters.push_back(counter);
  });
  if (neuraCounters.size() != domain.axes.size()) {
    error = "composite fusion source Neura counters differ from its complete "
            "Taskflow source domain";
    return false;
  }
  for (auto [index, axis] : llvm::enumerate(domain.axes)) {
    if (!axis.representedByTaskflow || axis.expandedInsideMapperFiring ||
        axis.carriedValues != 0 || axis.resultUses != 0 ||
        axis.ordinal != index) {
      error = "composite fusion source domain contains an internal, recurrent, "
              "or noncanonical counter axis";
      return false;
    }
    CounterBounds taskflowBounds, neuraBounds;
    if (!findTaskflowCounterBounds(task, axis.ordinal, taskflowBounds,
                                   error) ||
        !findNeuraCounterBounds(task, axis.ordinal, neuraBounds, error))
      return false;
    CounterBounds certified{axis.lower, axis.upper, axis.step};
    if (!equalBounds(taskflowBounds, certified) ||
        !equalBounds(neuraBounds, certified)) {
      error = "composite fusion source counter mirrors differ from their exact "
              "certified bounds";
      return false;
    }
  }
  return true;
}

static std::string compositeSourceOriginBinding(
    StringRef fusionKind,
    StringRef firstName, const TaskMetadata &first, StringRef firstBinding,
    StringRef secondName, const TaskMetadata &second,
    StringRef secondBinding, StringRef domainWitness) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  if (fusionKind == "sibling")
    stream << "amoeba-source-iteration-sibling-origin-v1\n";
  else if (fusionKind == "producer-consumer-forwarded")
    stream << "amoeba-source-iteration-producer-consumer-forwarded-origin-v1\n";
  else
    stream << "amoeba-source-iteration-producer-consumer-retained-origin-v1\n";
  stream << "domain_witness_bytes=" << domainWitness.size() << ':'
         << domainWitness << '\n'
         << "first_name_bytes=" << firstName.size() << ':' << firstName
         << '\n'
         << "first_source_work_count=" << first.sourceIterationWorkCount
         << '\n'
         << "first_source_binding_bytes=" << firstBinding.size() << ':'
         << firstBinding << '\n'
         << "second_name_bytes=" << secondName.size() << ':' << secondName
         << '\n'
         << "second_source_work_count=" << second.sourceIterationWorkCount
         << '\n'
         << "second_source_binding_bytes=" << secondBinding.size() << ':'
         << secondBinding << '\n';
  stream.flush();
  return result;
}

static std::string comparableSourceFunctionText(func::FuncOp function) {
  OwningOpRef<func::FuncOp> copy =
      cast<func::FuncOp>(function->clone());
  // Joint neighborhood search writes a state-local variant identity on this
  // function after source-owned fusion materialization. It does not alter the
  // function body or the fusion proof, so omit only this search bookkeeping
  // attribute when comparing the deterministic replay with the selected state.
  copy.get()->removeAttr("amoeba.graph_variant_id");
  copy->walk([&](TaskflowTaskOp task) {
    task->removeAttr(kSourceIterationDomainAttr);
    task->removeAttr(kSourceIterationControlBindingAttr);
    task->removeAttr(kSourceIterationSourceControlBindingAttr);
    task->removeAttr(kSourceIterationPartitionProofAttr);
    task->removeAttr(kSourceIterationCapturePendingAttr);
  });
  std::string text;
  llvm::raw_string_ostream stream(text);
  copy->print(stream, OpPrintingFlags().printGenericOpForm().useLocalScope());
  stream.flush();
  return text;
}

// ReplayJointNeighborhoodActionsPass records an action-history partition
// ledger after the source materializer has run. For a fusion of two
// unchanged canonical parents, that ledger is exactly two identity roots.
// Rebuild it from canonical task counters so it can be checked and included
// in the exact body witness; never accept a candidate-provided ledger as the
// proof itself.
static FailureOr<ArrayAttr> expectedIdentityFusionLineage(
    func::FuncOp canonicalFunction, const CompositeFusionRecord &fusion,
    const SourceIterationDomainInfo &domain, std::string &error) {
  if (!domain.complete || domain.internalMultiplicity != 1 ||
      domain.axes.empty()) {
    error = "composite fusion lineage requires one complete source firing";
    return failure();
  }
  auto findCanonicalTask = [&](StringRef name) -> TaskflowTaskOp {
    TaskflowTaskOp found;
    canonicalFunction.walk([&](TaskflowTaskOp task) {
      if (task.getTaskName() == name)
        found = task;
    });
    return found;
  };
  Builder builder(canonicalFunction.getContext());
  SmallVector<Attribute> roots;
  for (StringRef name : {StringRef(fusion.firstParent),
                         StringRef(fusion.secondParent)}) {
    TaskflowTaskOp task = findCanonicalTask(name);
    if (!task) {
      error = "composite fusion lineage cannot find its canonical parent";
      return failure();
    }
    for (StringRef attr : {StringRef(kNeighborhoodPartitionLineageAttr),
                           StringRef("amoeba.neura.tiling.parent_task"),
                           StringRef("amoeba.replica.parent_task"),
                           StringRef("amoeba.tiling.parent_task"),
                           StringRef("amoeba.semantic.k_tiled")})
      if (task->hasAttr(attr)) {
        error = "composite fusion identity lineage does not support already "
                "partitioned canonical parents";
        return failure();
      }
    SmallVector<int64_t> encodedDomain;
    for (const SourceIterationAxis &axis : domain.axes)
      encodedDomain.append({axis.lower, axis.upper, axis.step});
    NamedAttrList fields;
    fields.set("root", builder.getStringAttr(name));
    fields.set("original", builder.getDenseI64ArrayAttr(encodedDomain));
    SmallVector<Attribute> noSteps;
    fields.set("steps", builder.getArrayAttr(noSteps));
    roots.push_back(fields.getDictionary(canonicalFunction.getContext()));
  }
  return builder.getArrayAttr(roots);
}

static bool replayAndVerifyCompositeFusion(
    func::FuncOp canonicalFunction, func::FuncOp currentFunction,
    const CompositeFusionRecord &fusion,
    const SourceIterationDomainInfo &domain, StringRef domainWitness,
    std::string &error) {
  auto simpleTaskName = [](StringRef name) {
    return !name.empty() && llvm::all_of(name, [](char c) {
             return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '.';
           });
  };
  if (!simpleTaskName(fusion.firstParent) ||
      !simpleTaskName(fusion.secondParent)) {
    error = "composite fusion parent names are outside the supported exact "
            "source-name grammar";
    return false;
  }
  ModuleOp canonicalModule = canonicalFunction->getParentOfType<ModuleOp>();
  if (!canonicalModule) {
    error = "composite fusion source replay has no canonical module";
    return false;
  }
  OwningOpRef<ModuleOp> expectedModule =
      cast<ModuleOp>(canonicalModule->clone());
  std::string selectError;
  FailureOr<func::FuncOp> expectedFunction =
      selectTaskFunction(*expectedModule, canonicalFunction.getSymName(),
                         selectError);
  if (failed(expectedFunction)) {
    error = selectError.empty()
                ? "cannot select canonical composite fusion replay function"
                : selectError;
    return false;
  }
  std::unique_ptr<Pass> materializer =
      mlir::amoeba::neura::createMaterializeNeuraJointRewritePass();
  std::string options =
      "first-task-name=\"" + fusion.firstParent +
      "\" second-task-name=\"" + fusion.secondParent +
      "\" fusion-mode=" + fusion.materializerMode;
  if (!materializer ||
      failed(materializer->initializeOptions(
          options, [&](const llvm::Twine &message) {
            error = message.str();
            return failure();
          }))) {
    if (error.empty())
      error = "cannot initialize source-owned composite fusion replay";
    return false;
  }
  PassManager manager(expectedModule->getContext());
  manager.enableVerifier(true);
  manager.addPass(std::move(materializer));
  std::string diagnosticText;
  LogicalResult runResult = failure();
  {
    ScopedDiagnosticHandler capture(
        expectedModule->getContext(), [&](Diagnostic &diagnostic) {
          if (diagnosticText.empty()) {
            llvm::raw_string_ostream stream(diagnosticText);
            diagnostic.print(stream);
            stream.flush();
          }
          return success();
        });
    runResult = manager.run(*expectedModule);
  }
  if (failed(runResult) || failed(verify(expectedModule->getOperation()))) {
    error = "canonical source-owned composite fusion replay failed";
    if (!diagnosticText.empty())
      error += ": " + diagnosticText;
    return false;
  }
  expectedFunction = selectTaskFunction(*expectedModule,
                                        canonicalFunction.getSymName(),
                                        selectError);
  if (failed(expectedFunction)) {
    error = "cannot select materialized canonical composite fusion: " +
            selectError;
    return false;
  }
  std::string fusedName =
      (Twine(fusion.firstParent) + ".fuse." + fusion.secondParent).str();
  SmallVector<TaskflowTaskOp> expectedFused;
  SmallVector<TaskflowTaskOp> currentFused;
  expectedFunction->walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == fusedName)
      expectedFused.push_back(task);
  });
  currentFunction.walk([&](TaskflowTaskOp task) {
    if (task.getTaskName() == fusedName)
      currentFused.push_back(task);
  });
  if (expectedFused.size() != 1 || currentFused.size() != 1) {
    error = "composite fusion replay did not identify the exact fused child";
    return false;
  }
  for (StringRef attribute : {
           StringRef("amoeba.neura.fusion.mode"),
           StringRef("amoeba.neura.fusion.eliminated_loads"),
           StringRef("amoeba.neura.fusion.eliminated_stores")}) {
    if (expectedFused.front()->getAttr(attribute) !=
        currentFused.front()->getAttr(attribute)) {
      error = "composite fusion elimination metadata differs from the exact "
              "source-owned materializer replay";
      return false;
    }
  }
  FailureOr<ArrayAttr> expectedLineage = expectedIdentityFusionLineage(
      canonicalFunction, fusion, domain, error);
  if (failed(expectedLineage))
    return false;
  Attribute actualLineage =
      currentFused.front()->getAttr(kNeighborhoodPartitionLineageAttr);
  if (actualLineage != *expectedLineage) {
    error = "composite fusion action-history lineage differs from the exact "
            "canonical parent roots";
    return false;
  }
  expectedFused.front()->setAttr(kNeighborhoodPartitionLineageAttr,
                                 *expectedLineage);
  std::string expectedBinding = currentSourceIterationControlBinding(
      expectedFused.front(), domainWitness);
  std::string actualBinding = currentSourceIterationControlBinding(
      currentFused.front(), domainWitness);
  if (expectedBinding.empty() || actualBinding.empty() ||
      expectedBinding != actualBinding) {
    error = "current fused task shell, counters, memory roots, or mapper body "
            "differs from deterministic canonical replay";
    return false;
  }
  if (comparableSourceFunctionText(*expectedFunction) !=
      comparableSourceFunctionText(currentFunction)) {
    error = "composite fusion replay does not preserve exact canonical graph "
            "coverage and remaining parent operations";
    return false;
  }
  return true;
}

static std::optional<int64_t> originalAmoebaInteger(DictionaryAttr dictionary,
                                                    StringRef name) {
  auto value = dictionary ? dictionary.getAs<IntegerAttr>(name) : IntegerAttr();
  if (!value)
    return std::nullopt;
  return value.getInt();
}

static std::optional<StringRef> originalAmoebaString(DictionaryAttr dictionary,
                                                     StringRef name) {
  auto value = dictionary ? dictionary.getAs<StringAttr>(name) : StringAttr();
  if (!value)
    return std::nullopt;
  return value.getValue();
}

static bool buildOriginalAmoebaCounterBounds(
    TaskflowTaskOp task, SmallVectorImpl<CounterBounds> &bounds,
    std::string &error) {
  SmallVector<int64_t> ids;
  if (!collectCounterIds(task, ids, error))
    return false;
  bounds.reserve(ids.size());
  for (int64_t id : ids) {
    CounterBounds current;
    if (!findTaskflowCounterBounds(task, id, current, error))
      return false;
    bounds.push_back(current);
  }
  return true;
}

static ArrayAttr originalAmoebaBoundsAttr(MLIRContext *context,
                                          ArrayRef<CounterBounds> bounds) {
  OpBuilder builder(context);
  SmallVector<Attribute> axes;
  axes.reserve(bounds.size());
  for (auto [ordinal, bound] : llvm::enumerate(bounds))
    axes.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(ordinal)),
         builder.getNamedAttr("lower", builder.getI64IntegerAttr(bound.lower)),
         builder.getNamedAttr("upper", builder.getI64IntegerAttr(bound.upper)),
         builder.getNamedAttr("step", builder.getI64IntegerAttr(bound.step))}));
  return builder.getArrayAttr(axes);
}

// The original f45 scheduler owns only replica count, selected/composed
// shape, and ordered placements.  The native realization's selected output
// axis and source intervals are a separate proof.  Check their explicit
// binding before allowing the three derived attributes through canonical
// task-shell comparison.
static bool validateOriginalAmoebaRealizationMetadata(
    func::FuncOp function, const std::map<std::string, TaskMetadata> &parents,
    const std::map<std::string, TaskMetadata> &children,
    const std::map<std::string, std::vector<ReplicaRecord>> &replicas,
    std::set<std::string> &verifiedChildren, std::string &error) {
  constexpr StringLiteral kDecisionManifestAttr =
      "amoeba.original_amoeba.materialized_decisions";
  constexpr StringLiteral kReplicaDecisionAttr =
      "amoeba.original_amoeba.replica_decision";
  constexpr StringLiteral kReplicaPlacementsAttr =
      "amoeba.original_amoeba.replica_placements";
  constexpr StringLiteral kPartitionRealizationAttr =
      "amoeba.original_amoeba.source_partition_realization";
  auto reject = [&](StringRef reason) {
    error = "original AMOEBA source-realization metadata is invalid: " +
            reason.str();
    return false;
  };

  std::map<std::string, DictionaryAttr> manifests;
  ArrayAttr decisionManifest =
      function->getAttrOfType<ArrayAttr>(kDecisionManifestAttr);
  if (decisionManifest) {
    for (Attribute attribute : decisionManifest) {
      auto decision = dyn_cast<DictionaryAttr>(attribute);
      auto parentName = originalAmoebaString(decision, "parent_task");
      auto schema = originalAmoebaString(decision, "schema");
      if (!decision || decision.size() != 8 || !parentName || !schema ||
          *schema != "amoeba-original-fixed-decision-realization-v1" ||
          !manifests.emplace(parentName->str(), decision).second)
        return reject("function decision manifest has an invalid schema or "
                      "duplicate parent");
    }
  }

  bool foundDerivedMetadata = false;
  for (const auto &entry : children) {
    TaskflowTaskOp child = entry.second.op;
    bool hasDecision = child->hasAttr(kReplicaDecisionAttr);
    bool hasPlacements = child->hasAttr(kReplicaPlacementsAttr);
    bool hasRealization = child->hasAttr(kPartitionRealizationAttr);
    if (!hasDecision && !hasPlacements && !hasRealization)
      continue;
    foundDerivedMetadata = true;
    if (!hasDecision || !hasPlacements || !hasRealization)
      return reject("replica child has only a partial derived-attribute set");
  }
  if (foundDerivedMetadata && !decisionManifest)
    return reject("replica child metadata has no function-level decision "
                  "manifest");
  if (!decisionManifest)
    return true;

  std::set<std::string> verifiedParents;
  for (const auto &manifestEntry : manifests) {
    const std::string &parentName = manifestEntry.first;
    DictionaryAttr manifest = manifestEntry.second;
    auto parent = parents.find(parentName);
    auto group = replicas.find(parentName);
    if (parent == parents.end() || group == replicas.end() ||
        group->second.empty())
      return reject("manifest names a missing canonical parent or replica "
                    "group");
    TaskflowTaskOp canonical = parent->second.op;
    DictionaryAttr scheduler = canonical->getAttrOfType<DictionaryAttr>(
        "amoeba.task_scheduler_schedule_info");
    auto canonicalActive =
        canonical->getAttrOfType<IntegerAttr>("active_replicas");
    auto schedulerActive = originalAmoebaInteger(scheduler, "active_replicas");
    auto schedulerTask = originalAmoebaString(scheduler, "task_name");
    auto schedulerShapes =
        scheduler ? scheduler.getAs<ArrayAttr>("replica_shapes") : ArrayAttr();
    auto schedulerPlacements =
        scheduler ? scheduler.getAs<ArrayAttr>("placements") : ArrayAttr();
    auto schedulerComposedShape =
        originalAmoebaString(scheduler, "composed_cgra_shape");
    auto schedulerComposedCount =
        originalAmoebaInteger(scheduler, "composed_cgra_count");
    auto composedShape =
        canonical->getAttrOfType<StringAttr>("composed_cgra_shape");
    auto composedCount =
        canonical->getAttrOfType<IntegerAttr>("composed_cgra_count");
    auto manifestCount =
        originalAmoebaInteger(manifest, "original_replica_count");
    auto manifestOutputAxis =
        originalAmoebaInteger(manifest, "selected_output_axis");
    auto manifestCounter =
        originalAmoebaInteger(manifest, "selected_counter_ordinal");
    auto manifestScheduler = manifest.getAs<DictionaryAttr>("scheduler_record");
    ArrayAttr manifestDomain = manifest.getAs<ArrayAttr>("original_domain");
    ArrayAttr manifestReplicas = manifest.getAs<ArrayAttr>("replicas");
    int64_t composedRows = 0;
    int64_t composedCols = 0;
    bool validComposedShape = false;
    if (composedShape) {
      auto dimensions = composedShape.getValue().split('x');
      validComposedShape =
          !dimensions.first.empty() && !dimensions.second.empty() &&
          !dimensions.second.contains('x') &&
          !dimensions.first.getAsInteger(10, composedRows) &&
          !dimensions.second.getAsInteger(10, composedCols) &&
          composedRows > 0 && composedRows <= 4 && composedCols > 0 &&
          composedCols <= 4 && composedRows * composedCols <= 4;
    }
    if (!scheduler || !canonicalActive || !schedulerActive || !schedulerTask ||
        !schedulerShapes || !schedulerPlacements || !manifestCount ||
        !manifestOutputAxis || !manifestCounter || !manifestScheduler ||
        !manifestDomain || !manifestReplicas || !validComposedShape ||
        !composedCount ||
        composedCount.getInt() != composedRows * composedCols ||
        !schedulerComposedShape || !schedulerComposedCount ||
        *schedulerComposedShape != composedShape.getValue() ||
        *schedulerComposedCount != composedCount.getInt() ||
        *manifestCount < 2 || *manifestCount > 4 ||
        *schedulerTask != parentName ||
        canonicalActive.getInt() != *schedulerActive ||
        *manifestCount != *schedulerActive ||
        group->second.size() != static_cast<size_t>(*manifestCount) ||
        manifestReplicas.size() != group->second.size() ||
        manifestScheduler != scheduler || *manifestOutputAxis < 0 ||
        *manifestCounter < 0 ||
        !parent->second.sourceIterationDomainCertified ||
        !parent->second.sourceIterationDomainComplete ||
        parent->second.sourceIterationMultiplicity <= 0)
      return reject("manifest is not bound to the canonical f45 decision");

    SmallVector<CounterBounds> originalBounds;
    if (!buildOriginalAmoebaCounterBounds(canonical, originalBounds, error))
      return false;
    ArrayAttr expectedDomain =
        originalAmoebaBoundsAttr(function.getContext(), originalBounds);
    if (*manifestCounter >= static_cast<int64_t>(originalBounds.size()) ||
        manifestDomain != expectedDomain)
      return reject("manifest source domain differs from canonical counter "
                    "bounds");

    SmallVector<DictionaryAttr> shapes(*manifestCount);
    for (Attribute attribute : schedulerShapes) {
      auto shape = dyn_cast<DictionaryAttr>(attribute);
      auto id = originalAmoebaInteger(shape, "replica_id");
      auto cgraCount = originalAmoebaInteger(shape, "cgra_count");
      auto rows = originalAmoebaInteger(shape, "placement_rows");
      auto cols = originalAmoebaInteger(shape, "placement_cols");
      auto row = originalAmoebaInteger(shape, "row");
      auto col = originalAmoebaInteger(shape, "col");
      auto text = originalAmoebaString(shape, "shape");
      int64_t parsedRows = 0, parsedCols = 0;
      bool parsed = text &&
                    !text->split('x').first.getAsInteger(10, parsedRows) &&
                    !text->split('x').second.getAsInteger(10, parsedCols) &&
                    !text->split('x').second.contains('x');
      if (!shape || !id || !cgraCount || !rows || !cols || !row || !col ||
          !text || !parsed || *id < 0 || *id >= *manifestCount || shapes[*id] ||
          *rows <= 0 || *cols <= 0 || *row < 0 || *col < 0 || *rows > 4 ||
          *cols > 4 || *row > 3 || *col > 3 || *rows != parsedRows ||
          *cols != parsedCols || *cgraCount != *rows * *cols ||
          *cgraCount != composedCount.getInt() ||
          !((*rows == composedRows && *cols == composedCols) ||
            (*rows == composedCols && *cols == composedRows)) ||
          *row > 4 - *rows || *col > 4 - *cols)
        return reject("f45 replica-shape inventory is malformed");
      shapes[*id] = shape;
    }
    for (DictionaryAttr shape : shapes)
      if (!shape)
        return reject("f45 replica-shape inventory is incomplete");

    SmallVector<SmallVector<Attribute>> placementsById(*manifestCount);
    std::set<std::pair<int64_t, int64_t>> uniqueCells;
    for (Attribute attribute : schedulerPlacements) {
      auto placement = dyn_cast<DictionaryAttr>(attribute);
      auto id = originalAmoebaInteger(placement, "replica_id");
      auto row = originalAmoebaInteger(placement, "row");
      auto col = originalAmoebaInteger(placement, "col");
      auto context = originalAmoebaInteger(placement, "context_id");
      auto start = originalAmoebaInteger(placement, "scheduler_start_time");
      auto end = originalAmoebaInteger(placement, "scheduler_end_time");
      auto duration = originalAmoebaInteger(placement, "scheduler_duration");
      if (!placement || !id || !row || !col || !context || !start || !end ||
          !duration || *id < 0 || *id >= *manifestCount || *row < 0 ||
          *row >= 4 || *col < 0 || *col >= 4 || *context < 0 || *context >= 6 ||
          *start < 0 || *end <= *start || *duration != *end - *start ||
          !uniqueCells.emplace(*row, *col).second)
        return reject("f45 placement record is malformed or repeats a cell");
      auto baseRow = originalAmoebaInteger(shapes[*id], "row");
      auto baseCol = originalAmoebaInteger(shapes[*id], "col");
      auto rows = originalAmoebaInteger(shapes[*id], "placement_rows");
      auto cols = originalAmoebaInteger(shapes[*id], "placement_cols");
      if (!baseRow || !baseCol || !rows || !cols || *row < *baseRow ||
          *row >= *baseRow + *rows || *col < *baseCol ||
          *col >= *baseCol + *cols)
        return reject("f45 placement cell escapes its recorded shape");
      placementsById[*id].push_back(attribute);
    }
    for (int64_t id = 0; id < *manifestCount; ++id) {
      auto count = originalAmoebaInteger(shapes[id], "cgra_count");
      if (!count || placementsById[id].size() != static_cast<size_t>(*count))
        return reject("f45 placement cells do not cover each replica shape");
    }

    SmallVector<Attribute> summaries(manifestReplicas.begin(),
                                     manifestReplicas.end());
    SmallVector<bool> seenIds(*manifestCount, false);
    for (const ReplicaRecord &record : group->second) {
      TaskflowTaskOp child = record.child;
      if (record.sourceOwnedTiling || record.count != *manifestCount ||
          record.id < 0 || record.id >= *manifestCount || seenIds[record.id])
        return reject("manifest group is not a complete ordinary replica set");
      seenIds[record.id] = true;
      auto summary = dyn_cast<DictionaryAttr>(summaries[record.id]);
      auto summaryId = originalAmoebaInteger(summary, "replica_id");
      auto summaryName = originalAmoebaString(summary, "task_name");
      auto summaryShape = summary ? summary.get("replica_shape") : Attribute();
      auto summaryPlacements =
          summary ? summary.getAs<ArrayAttr>("placements") : ArrayAttr();
      auto summaryBounds =
          summary ? summary.getAs<ArrayAttr>("partition_bounds") : ArrayAttr();
      auto summaryWork = originalAmoebaInteger(summary, "source_work_count");
      auto childMetadata = children.find(child.getTaskName().str());
      auto decision =
          child->getAttrOfType<DictionaryAttr>(kReplicaDecisionAttr);
      auto childPlacements =
          child->getAttrOfType<ArrayAttr>(kReplicaPlacementsAttr);
      auto realization =
          child->getAttrOfType<DictionaryAttr>(kPartitionRealizationAttr);
      auto replicaId = child->getAttrOfType<IntegerAttr>("amoeba.replica.id");
      auto replicaCount =
          child->getAttrOfType<IntegerAttr>("amoeba.replica.count");
      auto shardAxis =
          child->getAttrOfType<IntegerAttr>("amoeba.replica.shard_axis");
      auto outputAxis =
          child->getAttrOfType<IntegerAttr>("amoeba.replica.output_shard_axis");
      auto decisionMode =
          child->getAttrOfType<BoolAttr>(kOriginalAmoebaDecisionModeAttr);
      if (canonical->hasAttr(kOriginalAmoebaDecisionModeAttr) ||
          !decisionMode || !decisionMode.getValue() ||
          child->hasAttr("profile_info") ||
          child->hasAttr("task_orchestration_info"))
        return reject("replica shell does not carry the trusted fixed-decision "
                      "mode or retains parent-only profile metadata");
      if (!summary || summary.size() != 6 || !summaryId || !summaryName ||
          !summaryShape || !summaryPlacements || !summaryBounds ||
          !summaryWork || childMetadata == children.end() || !decision ||
          !childPlacements || decision.size() != 3 || !realization ||
          realization.size() != 13 || !replicaId || !replicaCount ||
          !shardAxis || !outputAxis || summaryId != record.id ||
          *summaryName != child.getTaskName() ||
          summaryShape != shapes[record.id] ||
          summaryPlacements != OpBuilder(function.getContext())
                                   .getArrayAttr(placementsById[record.id]) ||
          childPlacements != summaryPlacements ||
          decision.getAs<DictionaryAttr>("scheduler_record") != scheduler ||
          decision.get("replica_shape") != shapes[record.id] ||
          decision.getAs<ArrayAttr>("placements") != summaryPlacements ||
          replicaId.getInt() != record.id ||
          replicaCount.getInt() != *manifestCount ||
          shardAxis.getInt() != *manifestCounter ||
          outputAxis.getInt() != *manifestOutputAxis ||
          record.shardAxis != *manifestCounter ||
          record.outputShardAxis != *manifestOutputAxis)
        return reject("replica decision metadata differs from its scheduler "
                      "record, id, shape, placement, or proved output axis");

      auto schema = originalAmoebaString(realization, "schema");
      auto status = originalAmoebaString(realization, "status");
      auto origin = originalAmoebaString(realization, "origin");
      auto f45Axes = realization.getAs<BoolAttr>("f45_axis_and_bounds_present");
      auto realizedAxis =
          originalAmoebaInteger(realization, "selected_output_axis");
      auto realizedCounter =
          originalAmoebaInteger(realization, "selected_counter_ordinal");
      auto realizedCount =
          originalAmoebaInteger(realization, "original_replica_count");
      auto realizedId = originalAmoebaInteger(realization, "replica_id");
      auto realizedDomain = realization.getAs<ArrayAttr>("original_domain");
      auto realizedBounds = realization.getAs<ArrayAttr>("partition_bounds");
      auto realizedWork =
          originalAmoebaInteger(realization, "source_work_count");
      auto multiplicity =
          originalAmoebaInteger(realization, "internal_multiplicity");
      auto decisionBinding =
          originalAmoebaString(realization, "decision_binding");
      SmallVector<CounterBounds> childBounds;
      if (!buildOriginalAmoebaCounterBounds(child, childBounds, error))
        return false;
      ArrayAttr expectedBounds =
          originalAmoebaBoundsAttr(function.getContext(), childBounds);
      int64_t volume = 0;
      if (!checkedCounterVolume(childBounds, volume) ||
          parent->second.sourceIterationMultiplicity <= 0 ||
          volume > std::numeric_limits<int64_t>::max() /
                       parent->second.sourceIterationMultiplicity)
        return reject("replica source-work volume is invalid or overflows");
      int64_t expectedWork =
          volume * parent->second.sourceIterationMultiplicity;
      if (!schema ||
          *schema != "amoeba-source-certified-replica-realization-v1" ||
          !status || *status != "verified-source-certified-realization-v1" ||
          !origin ||
          *origin !=
              "orbit-native-materializer-selected-counter-axis-after-f45" ||
          !f45Axes || f45Axes.getValue() || !realizedAxis ||
          *realizedAxis != *manifestOutputAxis || !realizedCounter ||
          *realizedCounter != *manifestCounter || !realizedCount ||
          *realizedCount != *manifestCount || !realizedId ||
          *realizedId != record.id || realizedDomain != expectedDomain ||
          realizedBounds != expectedBounds || summaryBounds != expectedBounds ||
          !realizedWork || *realizedWork != expectedWork ||
          *summaryWork != expectedWork ||
          childMetadata->second.taskflowTripCount != volume ||
          childMetadata->second.sourceIterationWorkCount != expectedWork ||
          childMetadata->second.tripCount != record.childTripCount ||
          !multiplicity ||
          *multiplicity != parent->second.sourceIterationMultiplicity ||
          !decisionBinding ||
          *decisionBinding != "amoeba.task_scheduler_schedule_info")
        return reject("native source partition metadata differs from exact "
                      "child counter bounds or source-work count");
      verifiedChildren.insert(child.getTaskName().str());
    }
    if (llvm::any_of(seenIds, [](bool seen) { return !seen; }))
      return reject("manifest group omits a replica id");
    verifiedParents.insert(parentName);
  }

  for (const auto &entry : children) {
    TaskflowTaskOp child = entry.second.op;
    if (!child->hasAttr(kPartitionRealizationAttr))
      continue;
    auto parentName =
        child->getAttrOfType<StringAttr>("amoeba.replica.parent_task");
    if (!parentName || verifiedParents.find(parentName.getValue().str()) ==
                           verifiedParents.end())
      return reject("derived replica metadata is absent from the complete "
                    "function manifest");
  }
  return true;
}

static std::string sourceTaskName(TaskflowTaskOp task) {
  if (auto parent = task->getAttrOfType<StringAttr>(
          "amoeba.tiling.parent_task"))
    return parent.getValue().str();
  if (auto parent = task->getAttrOfType<StringAttr>(
          "amoeba.replica.parent_task"))
    return parent.getValue().str();
  return task.getTaskName().str();
}

static bool isSequentialKLineage(TaskflowTaskOp task) {
  return task->hasAttr("amoeba.tiling.parent_task") &&
         isSequentialKTask(task);
}

struct InheritReplicaAnalyticalTaskCostCatalogPass
    : PassWrapper<InheritReplicaAnalyticalTaskCostCatalogPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      InheritReplicaAnalyticalTaskCostCatalogPass)

  InheritReplicaAnalyticalTaskCostCatalogPass() = default;
  InheritReplicaAnalyticalTaskCostCatalogPass(
      const InheritReplicaAnalyticalTaskCostCatalogPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "inherit-replica-analytical-task-cost-catalog";
  }
  StringRef getDescription() const override {
    return "Inherit ML II/startup costs after proving replica body equality";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect,
                    affine::AffineDialect, memref::MemRefDialect,
                    scf::SCFDialect, TaskflowDialect, neura::NeuraDialect>();
  }

  Option<std::string> functionName{*this, "function", llvm::cl::desc("Child taskflow function"), llvm::cl::init("")};
  Option<std::string> parentCostFile{*this, "parent-cost-file", llvm::cl::desc("Canonical parent ML catalogue"), llvm::cl::init("")};
  Option<std::string> parentModuleFile{*this, "parent-module", llvm::cl::desc("Canonical parent MLIR module"), llvm::cl::init("")};
  Option<std::string> childSpaceFile{*this, "child-space-file", llvm::cl::desc("Child factored task-shape manifest"), llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output", llvm::cl::desc("Inherited child catalogue"), llvm::cl::init("")};
  Option<std::string> sourceRepository{*this, "source-git-repository", llvm::cl::desc("Expected source Git repository"), llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-git-commit", llvm::cl::desc("Expected source Git commit"), llvm::cl::init("")};
  Option<std::string> architecturePath{*this, "architecture-path", llvm::cl::desc("Expected architecture path"), llvm::cl::init("")};
  Option<std::string> parentGraphVariantId{*this, "parent-graph-variant-id", llvm::cl::desc("Canonical parent graph identity"), llvm::cl::init("")};
  Option<std::string> modelNamespace{*this, "model-namespace", llvm::cl::desc("Expected predictor namespace"), llvm::cl::init("")};
  Option<std::string> model{*this, "model", llvm::cl::desc("Expected predictor model"), llvm::cl::init("")};
  Option<std::string> modelSchema{*this, "model-schema", llvm::cl::desc("Expected predictor model schema"), llvm::cl::init("")};
  Option<std::string> featureContract{*this, "feature-contract-id", llvm::cl::desc("Expected feature contract"), llvm::cl::init("")};
  Option<std::string> featureExtractor{*this, "feature-extractor", llvm::cl::desc("Expected feature extractor"), llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp childModule = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(childModule, functionName.getValue(), error);
    if (failed(selected)) {
      childModule.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp childFunction = *selected;
    if (parentCostFile.getValue().empty() || parentModuleFile.getValue().empty() ||
        childSpaceFile.getValue().empty() || outputFile.getValue().empty() ||
        sourceRepository.getValue().empty() || sourceCommit.getValue().empty() ||
        architecturePath.getValue().empty() ||
        parentGraphVariantId.getValue().empty() ||
        modelNamespace.getValue().empty() || model.getValue().empty() ||
        modelSchema.getValue().empty() || featureContract.getValue().empty() ||
        featureExtractor.getValue().empty() || !isGitCommit(sourceCommit) ||
        samePath(parentCostFile, outputFile) || samePath(childSpaceFile, outputFile) ||
        samePath(parentModuleFile, outputFile)) {
      childFunction.emitError()
          << "replica cost inheritance requires distinct paths and complete "
             "source, graph, architecture, and model identity options";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> childTasks =
        collectAnalyticalTaskMetadata(childFunction, error);
    if (failed(childTasks)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    MLIRContext *context = childModule.getContext();
    context->loadDialect<func::FuncDialect, arith::ArithDialect,
                         affine::AffineDialect, memref::MemRefDialect,
                         scf::SCFDialect, TaskflowDialect,
                         neura::NeuraDialect>();
    OwningOpRef<ModuleOp> parentModule =
        parseSourceFile<ModuleOp>(parentModuleFile, ParserConfig(context));
    if (!parentModule) {
      childFunction.emitError() << "cannot parse canonical parent module "
                                << parentModuleFile;
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> parentFunction = selectTaskFunction(
        *parentModule, functionName.getValue(), error);
    if (failed(parentFunction)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    FailureOr<SmallVector<TaskMetadata>> parentTasks =
        collectAnalyticalTaskMetadata(*parentFunction, error);
    if (failed(parentTasks)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    if (parentFunction->getSymName() != childFunction.getSymName()) {
      childFunction.emitError() << "canonical and child function identities "
                                   "do not match";
      return signalPassFailure();
    }

    ChildSpace childSpace;
    if (!parseFactoredSpace(childSpaceFile, childFunction.getSymName(),
                            childSpace, error)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    if (childSpace.factors.size() != childTasks->size()) {
      childFunction.emitError() << "child factored manifest task count does not "
                                   "match child module";
      return signalPassFailure();
    }
    for (auto [index, factor] : llvm::enumerate(childSpace.factors)) {
      if (factor.task != (*childTasks)[index].name ||
          factor.tripCount != (*childTasks)[index].tripCount) {
        childFunction.emitError()
            << "child factored manifest task/trip metadata does not match "
               "child module";
        return signalPassFailure();
      }
    }

    json::Object root;
    std::map<ShapeKey, const json::Object *> parentEntries;
    if (!parseParentCatalogue(
            parentCostFile, parentFunction->getSymName(), *parentTasks,
            sourceRepository, sourceCommit, architecturePath,
            parentGraphVariantId, modelNamespace, model, modelSchema,
            featureContract, featureExtractor, root, parentEntries, error)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }

    std::map<std::string, TaskMetadata> parents;
    std::map<std::string, TaskMetadata> children;
    std::map<std::string, std::vector<ReplicaRecord>> replicas;
    std::map<std::string, bool> sourceOwnedLineage;
    std::map<std::string, CompositeFusionRecord> compositeFusions;
    if (!collectReplicaLayout(*parentTasks, *childTasks, parents, children,
                              replicas, sourceOwnedLineage, compositeFusions,
                              error)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    if (!compositeFusions.empty()) {
      childFunction.emitError()
          << "composite fusion requires a fresh prediction of the "
             "combined mapper body; parent catalogue inheritance is refused";
      return signalPassFailure();
    }
    std::set<std::string> verifiedOriginalAmoebaChildren;
    if (!validateOriginalAmoebaRealizationMetadata(
            childFunction, parents, children, replicas,
            verifiedOriginalAmoebaChildren, error)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }

    std::map<std::string, BodyProof> proofs;
    for (const TaskMetadata &childTask : *childTasks) {
      auto lineage = sourceOwnedLineage.find(childTask.name);
      std::string sourceName =
          lineage == sourceOwnedLineage.end() || !lineage->second
              ? sourceTaskName(childTask.op)
              : [&]() {
                  for (const auto &group : replicas)
                    for (const ReplicaRecord &record : group.second)
                      if (record.child == childTask.op)
                        return record.parent;
                  return std::string();
                }();
      if (sourceName.empty()) {
        childFunction.emitError() << "source-owned child has no lineage: "
                                  << childTask.name;
        return signalPassFailure();
      }
      auto parent = parents.find(sourceName);
      if (parent == parents.end()) {
        childFunction.emitError() << "no canonical body for child task "
                                  << childTask.name;
        return signalPassFailure();
      }
      BodyProof proof;
      if (!proveTaskBody(parent->second.op, childTask.op, proof, error,
                         isSequentialKLineage(childTask.op),
                         lineage != sourceOwnedLineage.end() && lineage->second,
                         verifiedOriginalAmoebaChildren.count(childTask.name) !=
                             0)) {
        childFunction.emitError() << "replica body proof failed for child task "
                                  << childTask.name << ": " << error;
        return signalPassFailure();
      }
      proofs[childTask.name] = proof;
    }

    json::Array outputEntries;
    std::set<ShapeKey> covered;
    for (const ChildFactor &factor : childSpace.factors) {
      auto child = children.find(factor.task);
      if (child == children.end()) {
        childFunction.emitError() << "child manifest names task absent from IR: "
                                  << factor.task;
        return signalPassFailure();
      }
      auto lineage = sourceOwnedLineage.find(factor.task);
      std::string sourceName =
          lineage == sourceOwnedLineage.end() || !lineage->second
              ? sourceTaskName(child->second.op)
              : [&]() {
                  for (const auto &group : replicas)
                    for (const ReplicaRecord &record : group.second)
                      if (record.child == child->second.op)
                        return record.parent;
                  return std::string();
                }();
      if (sourceName.empty()) {
        childFunction.emitError() << "child task has no source lineage: "
                                  << factor.task;
        return signalPassFailure();
      }
      for (const ShapeKey &shape : factor.shapes) {
        ShapeKey sourceShape{sourceName, shape.rows, shape.cols};
        auto source = parentEntries.find(sourceShape);
        if (source == parentEntries.end()) {
          childFunction.emitError()
              << "parent catalogue has no shape for child task " << shape.task
              << " inherited from " << sourceName;
          return signalPassFailure();
        }
        if (!covered.insert(shape).second) {
          childFunction.emitError() << "child manifest repeats task/shape "
                                    << shape.task;
          return signalPassFailure();
        }
        json::Object entry;
        copyJsonObject(*source->second, entry);
        entry["task"] = shape.task;
        entry["mapper_tile_rows"] = shape.rows;
        entry["mapper_tile_cols"] = shape.cols;
        entry["trip_count"] = factor.tripCount;
        if (shape.task != sourceName) {
          entry["prediction_inherited_from"] = sourceName;
          entry["prediction_inheritance"] =
              isSequentialKLineage(child->second.op)
                  ? "proved-sequential-k-memory-feedback-body"
                  : "proved-mapper-visible-neura-body";
        }
        auto duration = predictedDuration(*source->second, factor.tripCount,
                                          error);
        if (!error.empty()) {
          childFunction.emitError() << error;
          return signalPassFailure();
        }
        if (duration)
          entry["predicted_duration"] = *duration;
        outputEntries.push_back(std::move(entry));
      }
    }
    if (outputEntries.empty()) {
      childFunction.emitError() << "child factored manifest produced no cost "
                                   "queries";
      return signalPassFailure();
    }

    json::Object *metadata = root.getObject("predictor_metadata");
    if (!metadata) {
      childFunction.emitError() << "validated parent catalogue lost predictor "
                                   "metadata";
      return signalPassFailure();
    }
    (*metadata)["source_graph_id"] = childSpace.graphVariantId;
    putDecimal(*metadata, "candidate_count", childSpace.candidateCount);
    json::Array sourceTaskIds;
    for (const TaskMetadata &task : *childTasks)
      sourceTaskIds.push_back(task.name);
    (*metadata)["source_task_ids"] = std::move(sourceTaskIds);
    json::Array proofRecords;
    for (const TaskMetadata &task : *childTasks) {
      const BodyProof &proof = proofs[task.name];
      auto lineage = sourceOwnedLineage.find(task.name);
      std::string sourceName =
          lineage == sourceOwnedLineage.end() || !lineage->second
              ? sourceTaskName(task.op)
              : [&]() {
                  for (const auto &group : replicas)
                    for (const ReplicaRecord &record : group.second)
                      if (record.child == task.op)
                        return record.parent;
                  return std::string();
                }();
      json::Object record{{"task", task.name},
                          {"source_task", sourceName},
                          {"structural_equality", proof.equal},
                          {"parent_operation_count", proof.parentOperations},
                          {"child_operation_count", proof.childOperations},
                          {"parent_kernel_count", proof.parentKernels},
                          {"child_kernel_count", proof.childKernels},
                          {"parent_counter_count", proof.parentCounters},
                          {"child_counter_count", proof.childCounters},
                          {"counter_bound_variation", "ignored-only-in-counter"}};
      if (isSequentialKLineage(task.op)) {
        record["lineage"] = "sequential-k-memory-feedback";
        record["counter_bound_proof"] =
            "taskflow-neura-equal-semantic-range-equal-trip-product";
        auto blockIndex = task.op->getAttrOfType<IntegerAttr>(
            "amoeba.semantic.k_block_index");
        record["state_carry_proof"] =
            blockIndex && blockIndex.getInt() == 0
                ? "initial_C_once"
                : "direct-previous-done-write-ssa-typed-explicit-alias";
        if (blockIndex)
          record["k_block_index"] = blockIndex.getInt();
        if (auto count = task.op->getAttrOfType<IntegerAttr>(
                "amoeba.semantic.k_block_count"))
          record["k_block_count"] = count.getInt();
        if (auto range = task.op->getAttrOfType<DenseI64ArrayAttr>(
                "amoeba.semantic.K_range")) {
          json::Array kRange;
          for (int64_t value : range.asArrayRef())
            kRange.push_back(value);
          record["K_range"] = std::move(kRange);
        }
        for (StringRef dimension : {StringRef("M"), StringRef("N")}) {
          std::string attribute =
              (Twine("amoeba.semantic.") + dimension + "_range").str();
          if (auto range = task.op->getAttrOfType<DenseI64ArrayAttr>(
                  attribute)) {
            json::Array values;
            for (int64_t value : range.asArrayRef())
              values.push_back(value);
            record[(Twine(dimension) + "_range").str()] = std::move(values);
          }
        }
      }
      if (task.name != sourceName) {
        record["child_trip_count"] = task.tripCount;
        record["parent_trip_count"] = parents[sourceName].tripCount;
      }
      proofRecords.push_back(std::move(record));
    }
    (*metadata)["replica_inheritance"] = json::Object{
        {"schema", kInheritanceSchema},
        {"status", "complete"},
        {"body_proof", "mapper-visible-neura-structural-equality"},
        {"bounds_policy", "counter-bound-SSA-and-derived-constants-only"},
        {"source_parent_module", parentModuleFile.getValue()},
        {"source_parent_graph_variant_id", parentGraphVariantId.getValue()},
        {"child_graph_variant_id", childSpace.graphVariantId},
        {"trip_count_role", "ranker-duration-only"},
        {"cache_identity", json::Object{
             {"namespace", modelNamespace.getValue()},
             {"model", model.getValue()},
             {"model_schema", modelSchema.getValue()},
             {"feature_contract_id", featureContract.getValue()},
             {"feature_extractor", featureExtractor.getValue()},
             {"architecture_path", architecturePath.getValue()},
             {"shape_key", "source-task+mapper-tile-rows+mapper-tile-cols"},
             {"trip_count_separate_from_prediction", true}}},
        {"proofs", std::move(proofRecords)}};
    root["entries"] = std::move(outputEntries);
    root["replica_inheritance_schema"] = kInheritanceSchema.str();

    if (!writeAtomically(
            outputFile, [&](llvm::raw_ostream &os) {
              os << json::Value(std::move(root)) << "\n";
              return true;
            },
            error)) {
      childFunction.emitError() << error;
      return signalPassFailure();
    }
    llvm::errs() << "[JointScheduling] inherited " << covered.size()
                 << " child task/mapper-shape costs after proving "
                    "mapper-visible Neura bodies\n";
  }
};

} // namespace

namespace mlir::amoeba::neura::joint_scheduling {

static LogicalResult processSourceIterationDomainPartition(
    func::FuncOp canonicalParent, func::FuncOp currentChild, std::string &error,
    bool refreshBindings) {
  if (!canonicalParent || !currentChild ||
      canonicalParent.getSymName() != currentChild.getSymName()) {
    error = "source-domain partition proof requires matching canonical and "
            "child functions";
    return failure();
  }

  FailureOr<SmallVector<TaskMetadata>> parentTasks =
      collectAnalyticalTaskMetadata(canonicalParent, error);
  if (failed(parentTasks))
    return failure();

  // Candidate rewrites may have narrowed or replaced Taskflow counters, and
  // their exact current bindings are intentionally stale until the complete
  // group is checked. Work on a private clone and remove only source-domain
  // certificates whose current bindings are about to be recomputed. Retain
  // the neighborhood partition ledger: nested tiling uses its authenticated
  // immediate-parent interval, while body comparison ignores this provenance
  // attribute as nonsemantic task-shell metadata.
  // Group counts, typed counter bounds, replica lineage, outputs, joins, and
  // bodies remain unchanged for the C++ proofs below.
  OwningOpRef<func::FuncOp> childProofFunction =
      cast<func::FuncOp>(currentChild->clone());
  func::FuncOp proofFunction = *childProofFunction;
  proofFunction.walk([&](TaskflowTaskOp task) {
    if (task->hasAttr(kSourceIterationCapturePendingAttr))
      error = "child task " + task.getTaskName().str() +
              " still has a pending source-domain capture";
    task->removeAttr(kSourceIterationDomainAttr);
    task->removeAttr(kSourceIterationControlBindingAttr);
    task->removeAttr(kSourceIterationSourceControlBindingAttr);
    task->removeAttr(kSourceIterationPartitionProofAttr);
    task->removeAttr(kSourceIterationCapturePendingAttr);
  });
  if (!error.empty())
    return failure();

  FailureOr<SmallVector<TaskMetadata>> childTasks =
      collectAnalyticalTaskMetadataForSourcePartitionProof(proofFunction,
                                                           error);
  if (failed(childTasks))
    return failure();

  std::map<std::string, TaskMetadata> parents;
  std::map<std::string, TaskMetadata> children;
  std::map<std::string, std::vector<ReplicaRecord>> replicas;
  std::map<std::string, bool> sourceOwnedLineage;
  std::map<std::string, CompositeFusionRecord> compositeFusions;
  if (!collectReplicaLayout(*parentTasks, *childTasks, parents, children,
                            replicas, sourceOwnedLineage, compositeFusions,
                            error))
    return failure();
  std::set<std::string> verifiedOriginalAmoebaChildren;
  if (!validateOriginalAmoebaRealizationMetadata(
          proofFunction, parents, children, replicas,
          verifiedOriginalAmoebaChildren, error))
    return failure();

  std::map<std::string, TaskflowTaskOp> actualChildren;
  currentChild.walk([&](TaskflowTaskOp task) {
    actualChildren.emplace(task.getTaskName().str(), task);
  });
  struct PendingRefresh {
    TaskflowTaskOp task;
    DictionaryAttr domain;
    StringAttr sourceBinding;
    std::string domainWitness;
    std::string currentBinding;
  };
  SmallVector<PendingRefresh> pendingRefreshes;
  pendingRefreshes.reserve(childTasks->size());

  SmallVector<const TaskMetadata *> childProofOrder;
  childProofOrder.reserve(childTasks->size());
  for (const TaskMetadata &child : *childTasks)
    if (compositeFusions.count(child.name))
      childProofOrder.push_back(&child);
  for (const TaskMetadata &child : *childTasks)
    if (!compositeFusions.count(child.name))
      childProofOrder.push_back(&child);
  bool compositeGraphReplayVerified = false;

  // Authenticate the one source-owned rewrite against a canonical clone
  // before checking any remaining tasks. Retained producer-consumer fusion
  // must pass the same complete, non-recurrent domain checks as sibling
  // fusion. The source-owned materializer replay re-proves its ordered RAW
  // edge, aliases, access indices, effects and complete task rewrite.
  // Fusion may redirect downstream operands from both parent completions;
  // exact whole-function replay proves those graph edges and all untouched
  // task bodies/counters.
  for (const TaskMetadata *childPointer : childProofOrder) {
    const TaskMetadata &child = *childPointer;
    auto fusion = compositeFusions.find(child.name);
    if (fusion != compositeFusions.end()) {
      auto first = parents.find(fusion->second.firstParent);
      auto second = parents.find(fusion->second.secondParent);
      auto actual = actualChildren.find(child.name);
      if (first == parents.end() || second == parents.end() ||
          actual == actualChildren.end()) {
        error = "composite fusion source or fused child is absent from the "
                "canonical graph";
        return failure();
      }
      TaskflowTaskOp firstTask = first->second.op;
      TaskflowTaskOp secondTask = second->second.op;
      TaskflowTaskOp actualTask = actual->second;
      auto firstDomain = firstTask->getAttrOfType<DictionaryAttr>(
          kSourceIterationDomainAttr);
      auto secondDomain = secondTask->getAttrOfType<DictionaryAttr>(
          kSourceIterationDomainAttr);
      auto firstSourceBinding = firstTask->getAttrOfType<StringAttr>(
          kSourceIterationSourceControlBindingAttr);
      auto secondSourceBinding = secondTask->getAttrOfType<StringAttr>(
          kSourceIterationSourceControlBindingAttr);
      if (!firstDomain || !secondDomain || firstDomain != secondDomain ||
          !firstSourceBinding || firstSourceBinding.getValue().empty() ||
          !secondSourceBinding || secondSourceBinding.getValue().empty()) {
        error = "composite fusion parents lack the same exact complete source "
                "domain or bound source-origin witnesses";
        return failure();
      }
      std::string firstParseError;
      std::string secondParseError;
      FailureOr<SourceIterationDomainInfo> firstInfo =
          parseSourceIterationDomain(firstTask, firstParseError);
      FailureOr<SourceIterationDomainInfo> secondInfo =
          parseSourceIterationDomain(secondTask, secondParseError);
      if (failed(firstInfo) || failed(secondInfo)) {
        error = "cannot parse complete composite source iteration domains: " +
                (failed(firstInfo) ? firstParseError : secondParseError);
        return failure();
      }
      std::string witness =
          sourceIterationDomainCanonicalWitness(*firstInfo);
      if (witness.empty() ||
          witness != sourceIterationDomainCanonicalWitness(*secondInfo) ||
          firstInfo->internalMultiplicity != 1 ||
          secondInfo->internalMultiplicity != 1 ||
          first->second.tripCount != second->second.tripCount ||
          first->second.taskflowTripCount !=
              second->second.taskflowTripCount ||
          first->second.sourceIterationMultiplicity !=
              second->second.sourceIterationMultiplicity ||
          first->second.sourceIterationWorkCount !=
              second->second.sourceIterationWorkCount ||
          child.tripCount != first->second.tripCount ||
          child.taskflowTripCount != first->second.taskflowTripCount ||
          child.sourceIterationWorkCount !=
              first->second.sourceIterationWorkCount) {
        error = "composite fusion parents do not share one exact complete "
                "source firing domain";
        return failure();
      }
      if (!exactSimpleSourceCounterDomain(firstTask, *firstInfo, error) ||
          !exactSimpleSourceCounterDomain(secondTask, *secondInfo, error) ||
          !exactSimpleSourceCounterDomain(actualTask, *firstInfo, error))
        return failure();
      if (!replayAndVerifyCompositeFusion(canonicalParent, currentChild,
                                          fusion->second, *firstInfo, witness,
                                          error))
        return failure();
      compositeGraphReplayVerified = true;

      std::string compositeSourceBinding = compositeSourceOriginBinding(
          fusion->second.materializerMode, fusion->second.firstParent,
          first->second,
          firstSourceBinding.getValue(), fusion->second.secondParent,
          second->second, secondSourceBinding.getValue(), witness);
      std::string currentBinding =
          currentSourceIterationControlBinding(actualTask, witness);
      if (currentBinding.empty()) {
        error = "fused composite task has no exact current body witness";
        return failure();
      }
      if (!refreshBindings) {
        auto existingDomain = actualTask->getAttrOfType<DictionaryAttr>(
            kSourceIterationDomainAttr);
        auto existingSourceBinding = actualTask->getAttrOfType<StringAttr>(
            kSourceIterationSourceControlBindingAttr);
        auto existingCurrentBinding = actualTask->getAttrOfType<StringAttr>(
            kSourceIterationControlBindingAttr);
        auto partitionProof = actualTask->getAttrOfType<StringAttr>(
            kSourceIterationPartitionProofAttr);
        if (actualTask->hasAttr(kSourceIterationCapturePendingAttr) ||
            !existingDomain || existingDomain != firstDomain ||
            !existingSourceBinding ||
            existingSourceBinding.getValue() != compositeSourceBinding ||
            !existingCurrentBinding ||
            existingCurrentBinding.getValue() != currentBinding ||
            !partitionProof || partitionProof.getValue() != witness) {
          error = "imported composite fusion has a stale or forged combined "
                  "source-origin/body certificate";
          return failure();
        }
      }
      pendingRefreshes.push_back(
          {actualTask, firstDomain,
           StringAttr::get(currentChild.getContext(), compositeSourceBinding),
           std::move(witness), std::move(currentBinding)});
      continue;
    }

    auto lineage = sourceOwnedLineage.find(child.name);
    std::string sourceName =
        lineage == sourceOwnedLineage.end() || !lineage->second
            ? sourceTaskName(child.op)
            : [&]() {
                for (const auto &group : replicas)
                  for (const ReplicaRecord &record : group.second)
                    if (record.child == child.op)
                      return record.parent;
                return std::string();
              }();
    if (sourceName.empty()) {
      error = "source-domain child has no canonical parent: " + child.name;
      return failure();
    }
    auto parent = parents.find(sourceName);
    auto actual = actualChildren.find(child.name);
    if (parent == parents.end() || actual == actualChildren.end()) {
      error = "source-domain child or canonical parent is missing for " +
              child.name;
      return failure();
    }
    TaskflowTaskOp canonicalTask = parent->second.op;
    TaskflowTaskOp actualTask = actual->second;
    auto canonicalDomain = canonicalTask->getAttrOfType<DictionaryAttr>(
        kSourceIterationDomainAttr);
    auto canonicalSourceBinding = canonicalTask->getAttrOfType<StringAttr>(
        kSourceIterationSourceControlBindingAttr);
    if (!canonicalDomain || !canonicalSourceBinding ||
        canonicalSourceBinding.getValue().empty()) {
      error = "canonical parent " + sourceName +
              " has no complete source-domain origin witness";
      return failure();
    }
    if (actualTask->hasAttr(kSourceIterationCapturePendingAttr)) {
      error = "source-domain child " + child.name +
              " has a pending source capture";
      return failure();
    }
    auto existingDomain = actualTask->getAttrOfType<DictionaryAttr>(
        kSourceIterationDomainAttr);
    if (actualTask->hasAttr(kSourceIterationDomainAttr) &&
        (!existingDomain || existingDomain != canonicalDomain)) {
      error = "source-domain child " + child.name +
              " has a certificate different from its canonical source";
      return failure();
    }
    if (!refreshBindings &&
        (!existingDomain || existingDomain != canonicalDomain)) {
      error = "imported source-domain child " + child.name +
              " is missing the exact canonical source certificate";
      return failure();
    }
    auto existingSourceBinding = actualTask->getAttrOfType<StringAttr>(
        kSourceIterationSourceControlBindingAttr);
    if (actualTask->hasAttr(kSourceIterationSourceControlBindingAttr) &&
        (!existingSourceBinding ||
         existingSourceBinding != canonicalSourceBinding)) {
      error = "source-domain child " + child.name +
              " has a source-origin binding different from its canonical "
              "source";
      return failure();
    }
    if (!refreshBindings &&
        (!existingSourceBinding ||
         existingSourceBinding != canonicalSourceBinding)) {
      error = "imported source-domain child " + child.name +
              " is missing the exact canonical source-origin witness";
      return failure();
    }

    if (lineage != sourceOwnedLineage.end() && !lineage->second &&
        !actualTask->hasAttr("amoeba.replica.parent_task") &&
        !actualTask->hasAttr("amoeba.tiling.parent_task")) {
      auto domainInfo = parseSourceIterationDomain(canonicalTask, error);
      if (failed(domainInfo)) return failure();
      for (const SourceIterationAxis &axis : domainInfo->axes) {
        if (!axis.representedByTaskflow) continue;
        CounterBounds parentTF, childTF, parentNeura, childNeura;
        if (!findTaskflowCounterBounds(canonicalTask, axis.ordinal, parentTF, error) ||
            !findTaskflowCounterBounds(child.op, axis.ordinal, childTF, error) ||
            !findNeuraCounterBounds(canonicalTask, axis.ordinal, parentNeura, error) ||
            !findNeuraCounterBounds(child.op, axis.ordinal, childNeura, error) ||
            !equalBounds(parentTF, childTF) || !equalBounds(parentNeura, childNeura)) {
          error = "unchanged source-domain task " + child.name +
                  " changed exact counter bounds: " + error;
          return failure();
        }
      }
    }
    if (!compositeGraphReplayVerified) {
      BodyProof bodyProof;
      if (!proveTaskBody(
              canonicalTask, child.op, bodyProof, error,
              isSequentialKLineage(child.op),
              lineage != sourceOwnedLineage.end() && lineage->second,
              verifiedOriginalAmoebaChildren.count(child.name) != 0)) {
        error = "source-domain child body proof failed for " + child.name +
                ": " + error;
        return failure();
      }
    }

    std::string parseError;
    FailureOr<SourceIterationDomainInfo> sourceInfo =
        parseSourceIterationDomain(canonicalTask, parseError);
    if (failed(sourceInfo)) {
      error = "canonical parent source-domain proof failed for " + sourceName +
              ": " + parseError;
      return failure();
    }
    std::string domainWitness =
        sourceIterationDomainCanonicalWitness(*sourceInfo);
    std::string currentBinding = currentSourceIterationControlBinding(
        actualTask, domainWitness);
    if (currentBinding.empty()) {
      error = "source-domain child " + child.name +
              " has no exact current body witness";
      return failure();
    }
    if (!refreshBindings) {
      auto storedCurrentBinding = actualTask->getAttrOfType<StringAttr>(
          kSourceIterationControlBindingAttr);
      if (!storedCurrentBinding ||
          storedCurrentBinding.getValue() != currentBinding) {
        error = "imported source-domain child " + child.name +
                " has a stale or forged current body binding";
        return failure();
      }
      auto partitionProof = actualTask->getAttrOfType<StringAttr>(
          kSourceIterationPartitionProofAttr);
      if (partitionProof && partitionProof.getValue() != domainWitness) {
        error = "imported source-domain child " + child.name +
                " has a stale or forged partition marker";
        return failure();
      }
      if (child.tripCount != parent->second.tripCount &&
          (!partitionProof || partitionProof.getValue() != domainWitness)) {
        error = "imported sharded source-domain child " + child.name +
                " has no matching verified partition marker";
        return failure();
      }
      if (child.tripCount == parent->second.tripCount && !partitionProof &&
          currentBinding != canonicalSourceBinding.getValue()) {
        error = "imported unsharded source-domain child " + child.name +
                " changed its current body without a trusted refresh";
        return failure();
      }
    }
    pendingRefreshes.push_back({actualTask, canonicalDomain,
                                canonicalSourceBinding,
                                std::move(domainWitness),
                                std::move(currentBinding)});
  }

  if (!refreshBindings)
    return success();

  // Refresh only after every candidate shard, body, counter mirror, output
  // region, join, and state-carry proof has passed for the full task graph.
  OpBuilder builder(currentChild.getContext());
  for (PendingRefresh &refresh : pendingRefreshes) {
    refresh.task->setAttr(kSourceIterationDomainAttr, refresh.domain);
    refresh.task->setAttr(kSourceIterationSourceControlBindingAttr,
                          refresh.sourceBinding);
    refresh.task->setAttr(kSourceIterationControlBindingAttr,
                          builder.getStringAttr(refresh.currentBinding));
    refresh.task->setAttr(kSourceIterationPartitionProofAttr,
                          builder.getStringAttr(refresh.domainWitness));
    refresh.task->removeAttr(kSourceIterationCapturePendingAttr);
  }
  return success();
}

LogicalResult proveAndRefreshSourceIterationDomainPartition(
    func::FuncOp canonicalParent, func::FuncOp currentChild,
    std::string &error) {
  return processSourceIterationDomainPartition(canonicalParent, currentChild,
                                               error,
                                               /*refreshBindings=*/true);
}

LogicalResult verifySourceIterationDomainPartition(
    func::FuncOp canonicalParent, func::FuncOp currentChild,
    std::string &error) {
  return processSourceIterationDomainPartition(canonicalParent, currentChild,
                                               error,
                                               /*refreshBindings=*/false);
}

} // namespace mlir::amoeba::neura::joint_scheduling

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createInheritReplicaAnalyticalTaskCostCatalogPass() {
  return std::make_unique<InheritReplicaAnalyticalTaskCostCatalogPass>();
}
} // namespace mlir::amoeba::neura

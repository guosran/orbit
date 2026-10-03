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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
         name.starts_with("amoeba.replica.") ||
         name == "amoeba.semantic.incoming_edges" ||
         name == "amoeba.tiling.parent_task" ||
         name == "amoeba.tiling.output_region_lowers" ||
         name == "amoeba.tiling.output_region_uppers" ||
         name.starts_with("amoeba.tiling.input_region_");
}

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
      upper.size() != lower.size()) {
    error = "ordinary dual-output replica " + task.getTaskName().str() +
            " has malformed output-region rank";
    return false;
  }
  for (auto [dimension, counter] : llvm::enumerate(proof.outputCounterAxes)) {
    if (counter >= proof.taskLowers.size() ||
        lower[dimension] != proof.taskLowers[counter] ||
        upper[dimension] != proof.taskUppers[counter]) {
      error = "ordinary dual-output replica " + task.getTaskName().str() +
              " output-region metadata disagrees with actual indexed "
              "coordinates";
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
    if (child.getWillWrites().size() > 1) {
      auto childOutputAxis = child->getAttrOfType<IntegerAttr>(
          "amoeba.replica.output_shard_axis");
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
  CounterBounds bounds[2];
  const ReplicaRecord *record = nullptr;
};

// Indexed output coordinates are authenticated from the actual Neura DFG.
// A copied region attribute is only a witness: forwarding through predication
// or a phi is accepted when every incoming value names the same counter, while
// arithmetic or an opaque value is rejected.
static std::optional<unsigned>
sourceOwnedOutputCounterAxis(Value value,
                             ArrayRef<neura::CounterOp> counters,
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
    const CounterBounds parentBounds[2], SourceOwnedPartitionBox &box,
    std::string &error) {
  TaskflowTaskOp task = record.child;
  SmallVector<TaskflowCounterOp> taskCounters;
  task.walk([&](TaskflowCounterOp counter) { taskCounters.push_back(counter); });
  if (taskCounters.size() != 2) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " must have exactly two Taskflow M/N counters";
    return false;
  }
  for (int64_t id = 0; id < 2; ++id) {
    unsigned matches = 0;
    for (TaskflowCounterOp counter : taskCounters)
      if (auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
          counterId && counterId.getInt() == id)
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
  if (neuraCounters.size() != 2) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " must have exactly two Neura M/N counters";
    return false;
  }
  for (int64_t id = 0; id < 2; ++id) {
    unsigned matches = 0;
    for (neura::CounterOp counter : neuraCounters)
      if (auto counterId = counter->getAttrOfType<IntegerAttr>("counter_id");
          counterId && counterId.getInt() == id)
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

  __int128 volume = static_cast<__int128>(box.bounds[0].upper -
                                           box.bounds[0].lower) *
                    static_cast<__int128>(box.bounds[1].upper -
                                           box.bounds[1].lower);
  if (volume <= 0 || volume > std::numeric_limits<int64_t>::max() ||
      record.childTripCount != static_cast<int64_t>(volume)) {
    error = "source-owned nested task " + task.getTaskName().str() +
            " trip count does not match its actual M/N counter volume";
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
    auto outputInput = parseKernelInputReference(store->getAttr("rhs_value"));
    auto outputIt = outputInput
                       ? llvm::find(outputKernelInputByWrite, *outputInput)
                       : outputKernelInputByWrite.end();
    if (outputIt == outputKernelInputByWrite.end() ||
        store.getIndices().size() != 2) {
      if (!outputInput || !outputKernelInputs.contains(*outputInput)) {
        error = "source-owned nested task " + task.getTaskName().str() +
                " has an unclassified indexed output store";
      } else {
        error = "source-owned nested task " + task.getTaskName().str() +
                " has an indexed output store with a non-rectangular "
                "coordinate route";
      }
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

  // A source-owned tiling task may move only its declared M or N axis.  The
  // untouched axis must remain the canonical parent's exact interval.
  if (auto axis = task->getAttrOfType<IntegerAttr>(
          "amoeba.neura.tiling.axis")) {
    auto original = task->getAttrOfType<DenseI64ArrayAttr>(
        "amoeba.neura.tiling.original_range");
    if (axis.getInt() < 0 || axis.getInt() > 1 || !original ||
        original.size() != 2 ||
        original[0] != parentBounds[axis.getInt()].lower ||
        original[1] != parentBounds[axis.getInt()].upper) {
      error = "source-owned nested task " + task.getTaskName().str() +
              " tiling axis/range does not match the canonical parent";
      return false;
    }
    for (int64_t dimension = 0; dimension < 2; ++dimension)
      if (dimension != axis.getInt() &&
          !equalBounds(box.bounds[dimension], parentBounds[dimension])) {
        error = "source-owned nested task " + task.getTaskName().str() +
                " changed an unsharded M/N axis";
        return false;
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

// The final partition check proves global coverage.  This second check proves
// each replaced intermediate tile's own completion join: the children must
// cover that join's region along exactly its selected axis and must agree on
// every unsharded axis.  It also authenticates the join states to the child
// done-write roots, so a forged parent name cannot borrow an unrelated join.
static bool validateSourceOwnedNestedJoins(
    ArrayRef<ReplicaRecord> records,
    ArrayRef<SourceOwnedPartitionBox> boxes, std::string &error) {
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
  }
  return true;
}

static bool validateSourceOwnedPartition(const TaskMetadata &parent,
                                         ArrayRef<ReplicaRecord> records,
                                         std::string &error) {
  if (records.empty()) {
    error = "source-owned nested partition is empty";
    return false;
  }
  CounterBounds parentBounds[2];
  CounterBounds parentNeuraBounds[2];
  for (int64_t id = 0; id < 2; ++id) {
    if (!findTaskflowCounterBounds(parent.op, id, parentBounds[id], error) ||
        !findNeuraCounterBounds(parent.op, id, parentNeuraBounds[id], error) ||
        !equalBounds(parentBounds[id], parentNeuraBounds[id]) ||
        parentBounds[id].step != 1) {
      error = "canonical source task " + parent.name +
              " has invalid or drifting M/N counter bounds";
      return false;
    }
  }
  __int128 parentVolume = static_cast<__int128>(
                              parentBounds[0].upper - parentBounds[0].lower) *
                          static_cast<__int128>(
                              parentBounds[1].upper - parentBounds[1].lower);
  if (parentVolume <= 0 || parentVolume > std::numeric_limits<int64_t>::max() ||
      parent.tripCount != static_cast<int64_t>(parentVolume)) {
    error = "canonical source task " + parent.name +
            " trip count does not match its M/N counter volume";
    return false;
  }

  SmallVector<SourceOwnedPartitionBox> boxes;
  boxes.reserve(records.size());
  __int128 volume = 0;
  for (const ReplicaRecord &record : records) {
    SourceOwnedPartitionBox box;
    if (!validateSourceOwnedTaskBox(parent, record, parentBounds, box, error))
      return false;
    volume += static_cast<__int128>(box.bounds[0].upper -
                                    box.bounds[0].lower) *
              static_cast<__int128>(box.bounds[1].upper -
                                    box.bounds[1].lower);
    boxes.push_back(box);
  }
  for (size_t lhs = 0; lhs < boxes.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < boxes.size(); ++rhs) {
      bool overlap = boxes[lhs].bounds[0].lower < boxes[rhs].bounds[0].upper &&
                     boxes[rhs].bounds[0].lower < boxes[lhs].bounds[0].upper &&
                     boxes[lhs].bounds[1].lower < boxes[rhs].bounds[1].upper &&
                     boxes[rhs].bounds[1].lower < boxes[lhs].bounds[1].upper;
      if (overlap) {
        error = "source-owned nested partition has overlapping actual M/N "
                "counter boxes";
        return false;
      }
    }
  if (volume != parentVolume) {
    error = "source-owned nested partition actual M/N boxes do not cover "
            "the canonical parent";
    return false;
  }
  if (!validateSourceOwnedNestedJoins(records, boxes, error))
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
         name.starts_with("amoeba.neura.tiling.");
}

static bool attrsEquivalent(Operation *lhs, Operation *rhs, bool taskAttrs,
                            bool counterAttrs, bool sequentialK = false,
                            bool sourceOwnedTiling = false) {
  for (NamedAttribute left : lhs->getAttrs()) {
    StringRef name = left.getName().getValue();
    if ((taskAttrs && isIgnoredTaskAttr(name)) ||
        (counterAttrs && isIgnoredCounterAttr(name)) ||
        (sequentialK && isSequentialKDerivedTaskAttr(name)) ||
        (sourceOwnedTiling && isSourceOwnedTilingAttr(name)))
      continue;
    Attribute right = rhs->getAttr(name);
    if (!right || right != left.getValue())
      return false;
  }
  for (NamedAttribute right : rhs->getAttrs()) {
    StringRef name = right.getName().getValue();
    if ((taskAttrs && isIgnoredTaskAttr(name)) ||
        (counterAttrs && isIgnoredCounterAttr(name)) ||
        (sequentialK && isSequentialKDerivedTaskAttr(name)) ||
        (sourceOwnedTiling && isSourceOwnedTilingAttr(name)))
      continue;
    Attribute left = lhs->getAttr(name);
    if (!left || left != right.getValue())
      return false;
  }
  return true;
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
  if (!attrsEquivalent(lhs, rhs, /*taskAttrs=*/false, counter)) {
    reason = "Neura body semantic attributes changed";
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
                          bool sourceOwnedTiling = false) {
  if (parent->getNumOperands() != child->getNumOperands() ||
      parent->getNumResults() != child->getNumResults() ||
      !attrsEquivalent(parent, child, /*taskAttrs=*/true,
                       /*counterAttrs=*/false, sequentialK,
                       sourceOwnedTiling)) {
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
      if (!resolveSourceOwnedFamily(declaredParent, resolved) ||
          parents.find(resolved) == parents.end()) {
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
      if (auto declared = task.op->getAttrOfType<IntegerAttr>(
              "amoeba.replica.shard_trip_count");
          declared && declared.getInt() != record.childTripCount) {
        error = "replica shard trip count does not match child metadata";
        return false;
      }
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
      if (!llvm::all_of(groupRecords, [](const ReplicaRecord &record) {
            return record.sourceOwnedTiling;
          })) {
        error = "source-owned nested partition is mixed with an ordinary "
                "replica set";
        return false;
      }
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
    if (!collectReplicaLayout(*parentTasks, *childTasks, parents, children,
                              replicas, sourceOwnedLineage, error)) {
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
                         lineage != sourceOwnedLineage.end() &&
                             lineage->second)) {
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

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createInheritReplicaAnalyticalTaskCostCatalogPass() {
  return std::make_unique<InheritReplicaAnalyticalTaskCostCatalogPass>();
}
} // namespace mlir::amoeba::neura

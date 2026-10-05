//===- OriginalAmoebaProfileCostAdapter.cpp -------------------*- C++ -*-===//
// Builds a selected-shape cost catalogue from corrected original-AMOEBA
// mapper-body and successful-profile evidence. This pass never searches
// shapes: it evaluates only each task's recorded original selected shape.
//===----------------------------------------------------------------------===//

#include "OriginalAmoebaProfileCostAdapter.h"
#include "NeighborhoodReplaySelection.h"

#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "Conversion/NeuraConversionPasses.h"
#include "MapperCostAnalysis.h"
#include "OriginalAmoebaCtrlToDataFlowPass.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
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
using namespace mlir::amoeba::neura::joint_scheduling;
using mlir::taskflow::TaskflowTaskOp;
using orbit::mapper_features::FeatureVector;
namespace json = llvm::json;

namespace mlir::amoeba::neura::joint_scheduling {

bool verifyOriginalAmoebaSourceDomainCoverage(
    llvm::ArrayRef<TaskMetadata> tasks, std::string &error,
    bool allowStaticInternalExpansion,
    bool allowScheduledReplicaInventory) {
  if (tasks.empty()) {
    error = "original source-domain coverage has no Taskflow tasks";
    return false;
  }
  for (const TaskMetadata &task : tasks) {
    TaskflowTaskOp operation = task.op;
    if (!task.sourceIterationDomainCertified ||
        !task.sourceIterationDomainComplete || !task.tripCountKnown ||
        task.sourceIterationDomainStatus != "certified-complete" ||
        (!allowStaticInternalExpansion && task.sourceIterationMultiplicity != 1) ||
        task.taskflowTripCount != task.tripCount ||
        (!allowStaticInternalExpansion && task.sourceIterationWorkCount != task.tripCount)) {
      error = "original task " + task.name +
              " lacks a complete one-to-one source iteration-domain proof";
      if (!task.sourceIterationDomainReason.empty())
        error += ": " + task.sourceIterationDomainReason;
      return false;
    }
    // Re-resolve the bound current/source certificate before admitting static
    // expansion. The mapper body and sample trip are checked separately: the
    // trip here is a macro firing count, not the expanded source-work count.
    if (allowStaticInternalExpansion) {
      auto resolved = resolveTaskIterationDomain(operation, error);
      int64_t internal = 1;
      int64_t work = 0;
      if (failed(resolved) || !resolved->sourceCertified || !resolved->complete ||
          !resolved->countKnown || resolved->status != "certified-complete" ||
          resolved->effectiveMapperFiringCount != task.tripCount ||
          resolved->taskflowTripCount != task.taskflowTripCount ||
          resolved->internalMultiplicity != task.sourceIterationMultiplicity ||
          resolved->sourceIterationWorkCount != task.sourceIterationWorkCount ||
          resolved->expandedInternalExtents != task.expandedInternalExtents) {
        if (error.empty())
          error = "original static-internal domain is not bound to the current task " + task.name;
        return false;
      }
      for (int64_t extent : resolved->expandedInternalExtents)
        if (extent < 1 || !checkedMultiply(internal, extent, internal)) {
          error = "original static-internal multiplicity is invalid for " + task.name;
          return false;
        }
      if (internal != resolved->internalMultiplicity ||
          !checkedMultiply(task.tripCount, internal, work) ||
          work != task.sourceIterationWorkCount) {
        error = "original macro firings and expanded source work disagree for " + task.name;
        return false;
      }
    }
    auto activeReplicas =
        operation->getAttrOfType<IntegerAttr>("active_replicas");
    if (!activeReplicas || activeReplicas.getInt() < 1 ||
        (!allowScheduledReplicaInventory && activeReplicas.getInt() != 1)) {
      error = "original task " + task.name +
              " has no explicit single-replica source-domain binding";
      return false;
    }
    if (allowScheduledReplicaInventory) {
      auto trace = operation->getAttrOfType<DictionaryAttr>(
          "amoeba.task_scheduler_schedule_info");
      auto traceActive = trace ? trace.getAs<IntegerAttr>("active_replicas")
                               : IntegerAttr();
      auto shapes = trace ? trace.getAs<ArrayAttr>("replica_shapes")
                          : ArrayAttr();
      auto placements = trace ? trace.getAs<ArrayAttr>("placements")
                              : ArrayAttr();
      if (!traceActive || traceActive.getInt() != activeReplicas.getInt() ||
          !shapes || shapes.size() !=
                         static_cast<size_t>(activeReplicas.getInt()) ||
          !placements || placements.empty()) {
        error = "original task " + task.name +
                " has no exact scheduler replica inventory";
        return false;
      }
      std::set<int64_t> replicaIds;
      int64_t expectedCells = 0;
      for (Attribute rawShape : shapes) {
        auto shape = dyn_cast<DictionaryAttr>(rawShape);
        auto id = shape ? shape.getAs<IntegerAttr>("replica_id")
                        : IntegerAttr();
        auto count = shape ? shape.getAs<IntegerAttr>("cgra_count")
                           : IntegerAttr();
        auto name = shape ? shape.getAs<StringAttr>("shape") : StringAttr();
        if (!id || id.getInt() < 0 ||
            id.getInt() >= activeReplicas.getInt() || !count ||
            count.getInt() < 1 || !name || name.getValue().empty() ||
            !replicaIds.insert(id.getInt()).second ||
            expectedCells > std::numeric_limits<int64_t>::max() -
                                count.getInt()) {
          error = "original task " + task.name +
                  " has malformed scheduler replica geometry";
          return false;
        }
        expectedCells += count.getInt();
      }
      std::set<std::tuple<int64_t, int64_t, int64_t>> placementCells;
      for (Attribute rawPlacement : placements) {
        auto placement = dyn_cast<DictionaryAttr>(rawPlacement);
        auto row = placement ? placement.getAs<IntegerAttr>("row")
                             : IntegerAttr();
        auto col = placement ? placement.getAs<IntegerAttr>("col")
                             : IntegerAttr();
        auto context = placement
                           ? placement.getAs<IntegerAttr>("context_id")
                           : IntegerAttr();
        auto replica = placement
                           ? placement.getAs<IntegerAttr>("replica_id")
                           : IntegerAttr();
        if (!row || row.getInt() < 0 || !col || col.getInt() < 0 ||
            !context || context.getInt() < 0 || context.getInt() >= 6 || !replica ||
            !replicaIds.count(replica.getInt()) ||
            !placementCells
                 .emplace(replica.getInt(), row.getInt(), col.getInt())
                 .second) {
          error = "original task " + task.name +
                  " has malformed or duplicate scheduler placement cells";
          return false;
        }
      }
      if (expectedCells != static_cast<int64_t>(placements.size()) ||
          replicaIds.size() !=
              static_cast<size_t>(activeReplicas.getInt())) {
        error = "original task " + task.name +
                " scheduler replica inventory does not exactly cover its "
                "placed cells";
        return false;
      }
    }
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if (name.starts_with("amoeba.replica.") ||
          name.starts_with("amoeba.tiling.")) {
        error = "original task " + task.name +
                " carries replica or tiling lineage; source-domain coverage "
                "is not the unmodified original task graph";
        return false;
      }
    }
  }
  return true;
}

StringRef originalAmoebaSourceDomainCoverageStatus(ArrayRef<TaskMetadata> tasks) {
  return llvm::any_of(tasks, [](const TaskMetadata &task) {
           return task.sourceIterationMultiplicity > 1;
         }) ? kOriginalAmoebaStaticInternalCoverageStatus
            : kOriginalAmoebaSourceDomainCoverageStatus;
}

namespace {

constexpr llvm::StringLiteral kBodyExportFormat =
    "amoeba-pre-mapper-task-bodies-v1";
constexpr llvm::StringLiteral kProfileFormat = "amoeba-task-profile-v1";
constexpr llvm::StringLiteral kCostProvenanceSchema =
    "orbit-cost-provenance-v1";
constexpr llvm::StringLiteral kArchitectureSchema = "neura-architecture-v1";
constexpr llvm::StringLiteral kModelName = "formal-max4-nohash-v2";
constexpr llvm::StringLiteral kModelStatus = "exploratory";
constexpr llvm::StringLiteral kOriginalBaselineGraphVariantId =
    "original-amoeba-full";
constexpr llvm::StringLiteral kDirectUnsupportedStatus =
    "unsupported-model-domain";
constexpr llvm::StringLiteral kDirectUnsupportedReason =
    "analytical-lower-bound-exceeds-model-ceiling";
constexpr llvm::StringLiteral kDirectDiagnosticUnsupportedPolicy =
    "analytical-lower-bound-exceeds-diagnostic-runtime-ceiling-v1";
constexpr llvm::StringLiteral kDirectDiagnosticUnsupportedReason =
    "analytical-lower-bound-exceeds-diagnostic-runtime-ceiling";
constexpr std::array<llvm::StringLiteral, 8> kExpectedProfileShapes = {
    "1x1", "1x2", "2x1", "1x3", "3x1", "2x2", "1x4", "4x1"};

struct SelectedProfile {
  std::string shape;
  int64_t cgraCount = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t sampleTripCount = 0;
  int64_t materializedOperationCount = 0;
  int64_t estimatedLatency = 0;
};

struct SelectedReplicaCell {
  int64_t row = 0;
  int64_t col = 0;
  int64_t contextId = 0;
};

struct SelectedReplicaPlacement {
  int64_t replicaId = 0;
  std::string shape;
  int64_t cgraCount = 0;
  int64_t row = 0;
  int64_t col = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<SelectedReplicaCell> cells;
};

struct SelectedTask {
  TaskMetadata metadata;
  OriginalAmoebaProfileBodyEvidence body;
  SelectedProfile profile;
  int64_t cgraRows = 0;
  int64_t cgraCols = 0;
  int64_t mapperRows = 0;
  int64_t mapperCols = 0;
  double recMII = 0.0;
  double resMII = 0.0;
  double lowerBound = 0.0;
  double startupCycles = 0.0;
  FeatureVector features;
  std::vector<SelectedReplicaPlacement> replicas;
  MLPEnsemblePrediction prediction;
  struct DirectShapePrediction {
    int64_t cgraRows = 0;
    int64_t cgraCols = 0;
    int64_t mapperRows = 0;
    int64_t mapperCols = 0;
    bool supported = false;
    double recMII = 0.0;
    double resMII = 0.0;
    double lowerBound = 0.0;
    double startupCycles = 0.0;
    double predictedIIStd = 0.0;
    std::vector<double> memberPredictions;
  };
  std::vector<DirectShapePrediction> directPredictions;
};

static bool readObjectFile(StringRef path, json::Object &object,
                           std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read JSON file " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  auto parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = "cannot parse JSON file " + path.str() + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  json::Object *parsedObject = parsed->getAsObject();
  if (!parsedObject) {
    error = "JSON file " + path.str() + " does not contain an object";
    return false;
  }
  object = std::move(*parsedObject);
  return true;
}

static bool readTextFile(StringRef path, std::string &text,
                         std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read architecture source " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  text = (*buffer)->getBuffer().str();
  if (text.empty()) {
    error = "architecture source is empty: " + path.str();
    return false;
  }
  return true;
}

static bool isSourceCommit(StringRef value) {
  if (value.size() < 7 || value.size() > 64)
    return false;
  return llvm::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
  });
}

static bool readJsonString(const json::Object &object, StringRef key,
                           std::string &value, std::string &error) {
  std::optional<StringRef> raw = object.getString(key);
  if (!raw || raw->empty()) {
    error = "missing or empty string field \"" + key.str() + "\"";
    return false;
  }
  value = raw->str();
  return true;
}

static bool readJsonInteger(const json::Object &object, StringRef key,
                            int64_t &value, std::string &error) {
  std::optional<int64_t> raw = object.getInteger(key);
  if (!raw) {
    error = "missing or invalid integer field \"" + key.str() + "\"";
    return false;
  }
  value = *raw;
  return true;
}

static std::optional<int64_t> getInteger(DictionaryAttr dictionary,
                                         StringRef key) {
  auto value = dictionary ? dyn_cast_or_null<IntegerAttr>(dictionary.get(key))
                          : IntegerAttr();
  if (!value)
    return std::nullopt;
  return value.getInt();
}

static std::optional<StringRef> getString(DictionaryAttr dictionary,
                                          StringRef key) {
  auto value = dictionary ? dyn_cast_or_null<StringAttr>(dictionary.get(key))
                          : StringAttr();
  if (!value)
    return std::nullopt;
  return value.getValue();
}

static bool parseSelectedShape(StringRef shape, int64_t &rows,
                               int64_t &columns) {
  auto parts = shape.split('x');
  return !parts.first.empty() && !parts.second.empty() &&
         !parts.second.contains('x') && !parts.first.getAsInteger(10, rows) &&
         !parts.second.getAsInteger(10, columns) && rows > 0 && columns > 0 &&
         rows <= 4 && columns <= 4 && rows * columns <= 4;
}

static bool readSelectedReplicaPlacements(
    Operation *operation, int64_t selectedRows, int64_t selectedCols,
    int64_t selectedCount, std::vector<SelectedReplicaPlacement> &result,
    std::string &error) {
  auto fail = [&](StringRef reason) {
    auto name = operation->getAttrOfType<StringAttr>("task_name");
    error = "task " + (name ? name.getValue().str() : std::string("<unknown>")) +
            ": " + reason.str();
    return false;
  };
  auto active = operation->getAttrOfType<IntegerAttr>("active_replicas");
  auto info = operation->getAttrOfType<DictionaryAttr>(
      "amoeba.task_scheduler_schedule_info");
  auto orchestration =
      operation->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  auto shapes = info ? info.getAs<ArrayAttr>("replica_shapes") : ArrayAttr();
  auto placements = info ? info.getAs<ArrayAttr>("placements") : ArrayAttr();
  auto infoActive = info ? getInteger(info, "active_replicas") : std::nullopt;
  auto orchestrationCells =
      orchestration ? orchestration.getAs<ArrayAttr>("cgra_positions")
                    : ArrayAttr();
  if (!active || active.getInt() < 1 || active.getInt() > 4 || !shapes ||
      shapes.size() != static_cast<size_t>(active.getInt()) || !placements ||
      placements.empty() || !infoActive || *infoActive != active.getInt() ||
      !orchestrationCells || orchestrationCells.size() != placements.size())
    return fail("original scheduler replica and physical-cell inventories are incomplete");

  result.assign(active.getInt(), SelectedReplicaPlacement{});
  for (Attribute rawShape : shapes) {
    auto shape = dyn_cast<DictionaryAttr>(rawShape);
    auto id = shape ? getInteger(shape, "replica_id") : std::nullopt;
    auto count = shape ? getInteger(shape, "cgra_count") : std::nullopt;
    auto row = shape ? getInteger(shape, "row") : std::nullopt;
    auto col = shape ? getInteger(shape, "col") : std::nullopt;
    auto rows = shape ? getInteger(shape, "placement_rows") : std::nullopt;
    auto cols = shape ? getInteger(shape, "placement_cols") : std::nullopt;
    auto name = shape ? getString(shape, "shape") : std::nullopt;
    int64_t parsedRows = 0, parsedCols = 0;
    if (!id || *id < 0 || *id >= active.getInt() || !count ||
        *count != selectedCount || !row || *row < 0 || !col || *col < 0 ||
        !rows || *rows < 1 || !cols || *cols < 1 || !name ||
        !parseSelectedShape(*name, parsedRows, parsedCols) ||
        parsedRows != *rows || parsedCols != *cols ||
        ((*rows != selectedRows || *cols != selectedCols) &&
         (*rows != selectedCols || *cols != selectedRows)) ||
        *rows * *cols != *count || *row + *rows > 4 || *col + *cols > 4 ||
        result[*id].cgraCount != 0)
      return fail("captured replica shape/count/orientation is invalid or duplicated");
    SelectedReplicaPlacement &replica = result[*id];
    replica.replicaId = *id;
    replica.shape = name->str();
    replica.cgraCount = *count;
    replica.row = *row;
    replica.col = *col;
    replica.rows = *rows;
    replica.cols = *cols;
  }

  std::set<std::pair<int64_t, int64_t>> allCells;
  std::map<std::pair<int64_t, int64_t>, int64_t> contexts;
  int64_t commonStart = -1, commonEnd = -1, commonDuration = -1;
  for (Attribute rawCell : placements) {
    auto cell = dyn_cast<DictionaryAttr>(rawCell);
    auto row = cell ? getInteger(cell, "row") : std::nullopt;
    auto col = cell ? getInteger(cell, "col") : std::nullopt;
    auto context = cell ? getInteger(cell, "context_id") : std::nullopt;
    auto replicaId = cell ? getInteger(cell, "replica_id") : std::nullopt;
    auto start = cell ? getInteger(cell, "scheduler_start_time") : std::nullopt;
    auto end = cell ? getInteger(cell, "scheduler_end_time") : std::nullopt;
    auto duration = cell ? getInteger(cell, "scheduler_duration") : std::nullopt;
    if (!row || *row < 0 || *row >= 4 || !col || *col < 0 || *col >= 4 ||
        !context || *context < 0 || !replicaId || *replicaId < 0 ||
        *replicaId >= active.getInt() || !start || *start < 0 || !end ||
        *end <= *start || !duration || *duration != *end - *start ||
        !allCells.emplace(*row, *col).second)
      return fail("physical placement cell, time, or replica identity is malformed/duplicated");
    if (commonStart < 0) {
      commonStart = *start;
      commonEnd = *end;
      commonDuration = *duration;
    } else if (commonStart != *start || commonEnd != *end ||
               commonDuration != *duration) {
      return fail("original replicas do not share a single scheduler interval");
    }
    SelectedReplicaPlacement &replica = result[*replicaId];
    if (replica.cgraCount == 0 || *row < replica.row ||
        *row >= replica.row + replica.rows || *col < replica.col ||
        *col >= replica.col + replica.cols)
      return fail("placed cell lies outside its recorded replica rectangle");
    replica.cells.push_back({*row, *col, *context});
    contexts.emplace(std::make_pair(*row, *col), *context);
  }

  for (SelectedReplicaPlacement &replica : result) {
    if (replica.cgraCount != static_cast<int64_t>(replica.cells.size()) ||
        replica.cgraCount != replica.rows * replica.cols)
      return fail("original replica omits cells from its exact rectangular placement");
  }
  if (commonStart < 0 || allCells.size() != placements.size())
    return fail("original scheduler placement does not cover its cells");

  std::map<std::pair<int64_t, int64_t>, int64_t> orchestrationContexts;
  for (Attribute rawCell : orchestrationCells) {
    auto cell = dyn_cast<DictionaryAttr>(rawCell);
    auto row = cell ? getInteger(cell, "row") : std::nullopt;
    auto col = cell ? getInteger(cell, "col") : std::nullopt;
    auto context = cell ? getInteger(cell, "context_id") : std::nullopt;
    if (!row || !col || !context || *context < 0 ||
        !orchestrationContexts.emplace(std::make_pair(*row, *col), *context)
             .second ||
        !contexts.count({*row, *col}) || contexts[{*row, *col}] != *context)
      return fail("task_orchestration_info disagrees with scheduler replica cells");
  }
  if (orchestrationContexts.size() != contexts.size())
    return fail("task_orchestration_info omits a scheduler replica cell");
  return true;
}

static bool checkedProfileDuration(int64_t ii, int64_t trip, int64_t steps,
                                   int64_t &duration) {
  if (ii < 1 || trip < 1 || steps < 1 ||
      trip - 1 > (std::numeric_limits<int64_t>::max() - steps) / ii)
    return false;
  duration = ii * (trip - 1) + steps;
  return duration > 0;
}

// Validate the historical F45 profile_info field only; native child costs use
// each child's full mapper profile and actual trip count.
static bool checkedF45SchedulerDuration(int64_t fullMapperDuration,
                                        int64_t activeReplicas,
                                        int64_t &schedulerDuration) {
  if (fullMapperDuration < 1 || activeReplicas < 1)
    return false;
  schedulerDuration = fullMapperDuration / activeReplicas +
                      (fullMapperDuration % activeReplicas != 0 ? 1 : 0);
  return schedulerDuration > 0;
}

static std::optional<std::string>
buildOriginalMapperWrapperModuleText(::mlir::neura::KernelOp kernel) {
  MLIRContext *context = kernel.getContext();
  Location location = kernel.getLoc();
  ModuleOp wrapperModule = ModuleOp::create(location);
  OpBuilder builder(context);
  builder.setInsertionPointToStart(wrapperModule.getBody());

  Region &kernelBody = kernel.getBody();
  if (kernelBody.empty()) {
    wrapperModule.erase();
    return std::nullopt;
  }

  llvm::SmallVector<Type> argumentTypes;
  for (BlockArgument argument : kernelBody.front().getArguments())
    argumentTypes.push_back(argument.getType());

  llvm::SmallVector<Type> resultTypes;
  bool usesDummyTrigger = true;
  kernel.walk([&](::mlir::neura::YieldOp yield) {
    if (yield.getResults().empty())
      return WalkResult::advance();
    resultTypes.assign(yield.getResults().getTypes().begin(),
                       yield.getResults().getTypes().end());
    usesDummyTrigger = false;
    return WalkResult::interrupt();
  });
  if (resultTypes.empty())
    resultTypes.push_back(builder.getI1Type());

  auto functionType = builder.getFunctionType(argumentTypes, resultTypes);
  auto wrapper =
      builder.create<func::FuncOp>(location, "__task_profile__", functionType);
  wrapper->setAttr("accelerator", builder.getStringAttr("neura"));

  IRMapping mapping;
  kernelBody.cloneInto(&wrapper.getBody(), mapping);
  for (Block &block : wrapper.getBody()) {
    auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator());
    if (!yield)
      continue;
    builder.setInsertionPoint(yield);
    if (usesDummyTrigger) {
      Value trigger = builder.create<arith::ConstantIntOp>(location, 1,
                                                           builder.getI1Type());
      builder.create<func::ReturnOp>(location, trigger);
    } else {
      builder.create<func::ReturnOp>(location, yield.getResults());
    }
    yield.erase();
  }

  std::string moduleText;
  llvm::raw_string_ostream stream(moduleText);
  wrapperModule.print(stream);
  stream.flush();
  wrapperModule.erase();
  return moduleText;
}

static void loadOriginalMapperDialects(MLIRContext &context) {
  context.loadDialect<affine::AffineDialect, arith::ArithDialect,
                      cf::ControlFlowDialect, LLVM::LLVMDialect,
                      func::FuncDialect, memref::MemRefDialect,
                      ::mlir::neura::NeuraDialect, scf::SCFDialect>();
}

static void addOriginalMapperLoweringPipeline(PassManager &manager) {
  manager.addPass(createCSEPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertSCFToCFPass());
  manager.addPass(createConvertControlFlowToLLVMPass());
  manager.addPass(::mlir::neura::createAssignAcceleratorPass());
  manager.addPass(::mlir::createLowerMemRefToNeuraPass());
  manager.addPass(::mlir::createLowerArithToNeuraPass());
  manager.addPass(::mlir::createLowerBuiltinToNeuraPass());
  manager.addPass(::mlir::createLowerLlvmToNeuraPass());
  manager.addPass(::mlir::neura::createPromoteInputArgToConstPass());
  manager.addPass(::mlir::neura::createFoldConstantPass());
  manager.addPass(::mlir::neura::createCanonicalizeReturnPass());
  manager.addPass(::mlir::neura::createCanonicalizeLiveInPass());
  manager.addPass(::mlir::neura::createLeveragePredicatedValuePass());
  manager.addPass(createOriginalAmoebaCtrlToDataFlowPass());
  manager.addPass(::mlir::neura::createFoldConstantPass());
  manager.addPass(::mlir::neura::createInsertDataMovPass());
}

static bool
reconstructOriginalNormalizedMapperBody(::mlir::neura::KernelOp kernel,
                                        std::string &normalizedMapperBody,
                                        std::string &error) {
  std::optional<std::string> wrapperText =
      buildOriginalMapperWrapperModuleText(kernel);
  if (!wrapperText) {
    error = "current neura.kernel has no body for the original mapper wrapper";
    return false;
  }

  MLIRContext mapperContext;
  mapperContext.disableMultithreading();
  loadOriginalMapperDialects(mapperContext);
  ParserConfig parserConfig(&mapperContext);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      *wrapperText, parserConfig, "task-profile-body-equivalence.mlir");
  if (!module) {
    error = "current neura.kernel could not be parsed as the original mapper "
            "wrapper";
    return false;
  }

  PassManager loweringManager(&mapperContext);
  loweringManager.enableVerifier(false);
  addOriginalMapperLoweringPipeline(loweringManager);
  if (failed(loweringManager.run(*module))) {
    error = "exact original TaskProfiler pre-mapper lowering pipeline failed "
            "for the current neura.kernel";
    return false;
  }
  func::FuncOp loweredFunction =
      module->lookupSymbol<func::FuncOp>("__task_profile__");
  if (!loweredFunction) {
    error = "original lowering pipeline removed __task_profile__";
    return false;
  }

  llvm::raw_string_ostream bodyStream(normalizedMapperBody);
  OpPrintingFlags printingFlags;
  printingFlags.useLocalScope();
  loweredFunction->print(bodyStream, printingFlags);
  bodyStream.flush();
  return true;
}

static bool getSelectedProfile(const json::Object &profileRoot,
                               StringRef taskName, StringRef selectedShape,
                               int64_t expectedCount, int64_t expectedTrip,
                               bool allowLegacyNoAttemptInventory,
                               SelectedProfile &result, std::string &error) {
  const json::Array *profileTasks = profileRoot.getArray("tasks");
  const json::Object *taskRecord = nullptr;
  if (profileTasks) {
    for (const json::Value &rawTask : *profileTasks) {
      const json::Object *candidate = rawTask.getAsObject();
      std::optional<StringRef> name =
          candidate ? candidate->getString("task") : std::nullopt;
      if (name && *name == taskName) {
        if (taskRecord) {
          error = "profile file repeats task " + taskName.str();
          return false;
        }
        taskRecord = candidate;
      }
    }
  }
  const json::Array *profiles =
      taskRecord ? taskRecord->getArray("profiles") : nullptr;
  const bool hasAttemptsField = profileRoot.get("candidate_attempts") != nullptr;
  const json::Array *attempts = profileRoot.getArray("candidate_attempts");
  if (hasAttemptsField && !attempts) {
    error = "profile candidate-attempt inventory is not an array";
    return false;
  }
  if (!profiles || (!attempts && !allowLegacyNoAttemptInventory)) {
    error = "profile file omits profile records for task " + taskName.str();
    return false;
  }

  const json::Object *selected = nullptr;
  std::map<std::string, const json::Object *> profileRowsByShape;
  for (const json::Value &rawProfile : *profiles) {
    const json::Object *candidate = rawProfile.getAsObject();
    std::optional<StringRef> shape =
        candidate ? candidate->getString("composed_cgra_shape") : std::nullopt;
    if (!candidate || !shape || shape->empty() ||
        !profileRowsByShape.emplace(shape->str(), candidate).second) {
      error = "profile file has malformed or duplicate profile shapes for " +
              taskName.str();
      return false;
    }
    int64_t candidateCount = 0;
    int64_t candidateRows = 0;
    int64_t candidateColumns = 0;
    std::optional<int64_t> candidateII = candidate->getInteger("compiled_ii");
    std::optional<int64_t> candidateSteps = candidate->getInteger("steps");
    std::optional<int64_t> candidateTrip =
        candidate->getInteger("sample_trip_count");
    std::optional<int64_t> candidateOperations =
        candidate->getInteger("materialized_operation_count");
    std::optional<int64_t> candidateLatency =
        candidate->getInteger("estimated_latency");
    std::optional<bool> candidateSucceeded =
        candidate->getBoolean("mapper_succeeded");
    if (!parseSelectedShape(*shape, candidateRows, candidateColumns) ||
        !readJsonInteger(*candidate, "composed_cgra_count", candidateCount,
                         error) ||
        candidateCount != candidateRows * candidateColumns || !candidateII ||
        !candidateSteps || !candidateTrip || !candidateOperations ||
        !candidateLatency || !candidateSucceeded) {
      if (error.empty())
        error = "profile file shape, mapper fields, or explicit success flag "
                "is invalid for " +
                taskName.str();
      return false;
    }
    if (!llvm::any_of(kExpectedProfileShapes,
                      [&](StringRef expected) { return expected == *shape; })) {
      error = "profile file contains a shape outside the exact original "
              "four-CGRA candidate domain for " +
              taskName.str();
      return false;
    }
    if (*shape == selectedShape) {
      if (selected) {
        error = "profile file repeats selected shape " + selectedShape.str() +
                " for " + taskName.str();
        return false;
      }
      selected = candidate;
    }
  }
  if (!attempts) {
    if (profileRowsByShape.size() != kExpectedProfileShapes.size() ||
        !llvm::all_of(kExpectedProfileShapes, [&](StringRef expected) {
          return profileRowsByShape.count(expected.str()) != 0;
        })) {
      error = "profile file does not contain the exact original four-CGRA "
              "shape domain for " +
              taskName.str();
      return false;
    }
  }
  std::map<std::string, const json::Object *> attemptsByShape;
  if (attempts) {
    for (const json::Value &rawAttempt : *attempts) {
      const json::Object *attempt = rawAttempt.getAsObject();
      auto attemptTask = attempt ? attempt->getString("task") : std::nullopt;
      if (!attemptTask || *attemptTask != taskName)
        continue;
      auto candidateIndex = attempt->getInteger("candidate_index_in_task");
      auto shape = attempt->getString("shape");
      auto count = attempt->getInteger("composed_cgra_count");
      auto profileCreated = attempt->getBoolean("profile_created");
      auto mapperSucceeded = attempt->getBoolean("mapper_succeeded");
      int64_t rows = 0;
      int64_t columns = 0;
      if (!candidateIndex || *candidateIndex < 1 ||
          *candidateIndex >
              static_cast<int64_t>(kExpectedProfileShapes.size()) ||
          !shape ||
          *shape != kExpectedProfileShapes[*candidateIndex - 1] || !count ||
          !parseSelectedShape(*shape, rows, columns) ||
          *count != rows * columns || !profileCreated || !mapperSucceeded ||
          !attemptsByShape.emplace(shape->str(), attempt).second) {
        error =
            "profile candidate-attempt inventory is malformed or duplicated "
            "for " +
            taskName.str();
        return false;
      }
    }
    if (attemptsByShape.size() != kExpectedProfileShapes.size()) {
      error = "profile file does not contain all eight explicit candidate "
              "attempts for " +
              taskName.str();
      return false;
    }
    for (StringRef shape : kExpectedProfileShapes) {
      const json::Object &attempt = *attemptsByShape.at(shape.str());
      const bool created = *attempt.getBoolean("profile_created");
      const bool succeeded = *attempt.getBoolean("mapper_succeeded");
      auto row = profileRowsByShape.find(shape.str());
      const bool hasRow = row != profileRowsByShape.end();
      auto rowSucceeded =
          hasRow ? row->second->getBoolean("mapper_succeeded") : std::nullopt;
      if (created != hasRow || succeeded != (created && rowSucceeded &&
                                              *rowSucceeded)) {
        error = "candidate attempt and emitted profile rows do not agree for " +
                taskName.str() + " shape=" + shape.str();
        return false;
      }
    }
  }
  if (!selected) {
    error = "profile file has no exact-orientation selected shape " +
            selectedShape.str() + " for " + taskName.str();
    return false;
  }

  auto successful = selected->getBoolean("mapper_succeeded");
  if (!successful || !*successful ||
      !readJsonInteger(*selected, "composed_cgra_count", result.cgraCount,
                       error) ||
      !readJsonInteger(*selected, "compiled_ii", result.compiledII, error) ||
      !readJsonInteger(*selected, "steps", result.steps, error) ||
      !readJsonInteger(*selected, "sample_trip_count", result.sampleTripCount,
                       error) ||
      !readJsonInteger(*selected, "materialized_operation_count",
                       result.materializedOperationCount, error) ||
      !readJsonInteger(*selected, "estimated_latency", result.estimatedLatency,
                       error)) {
    if (error.empty())
      error =
          "selected mapper profile does not explicitly report success for " +
          taskName.str();
    return false;
  }
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t formulaDuration = 0;
  if (!parseSelectedShape(selectedShape, rows, columns) ||
      result.cgraCount != rows * columns || result.compiledII < 1 ||
      result.steps < 1 || result.sampleTripCount != expectedTrip ||
      result.materializedOperationCount < 1 || result.estimatedLatency < 1 ||
      !checkedProfileDuration(result.compiledII, result.sampleTripCount,
                              result.steps, formulaDuration) ||
      formulaDuration != result.estimatedLatency) {
    error = "selected profile fields, trip count, shape, or profiled latency "
            "are inconsistent for " +
            taskName.str();
    return false;
  }
  result.shape = selectedShape.str();
  if (result.cgraCount != expectedCount) {
    error = "selected profile CGRA count differs from the source IR for " +
            taskName.str();
    return false;
  }
  return true;
}

static func::FuncOp parseMapperFunction(MLIRContext *context,
                                        StringRef mapperBody,
                                        OwningOpRef<ModuleOp> &parsedModule,
                                        std::string &error) {
  std::string moduleText = "module {\n" + mapperBody.str() + "\n}\n";
  parsedModule = parseSourceString<ModuleOp>(moduleText, context);
  if (!parsedModule) {
    error = "normalized_mapper_body is not parseable MLIR function text";
    return {};
  }
  if (failed(verify(*parsedModule))) {
    error = "normalized_mapper_body does not verify as MLIR";
    return {};
  }
  func::FuncOp found;
  unsigned count = 0;
  parsedModule->walk([&](func::FuncOp function) {
    found = function;
    ++count;
  });
  if (count != 1 || !found || found.getBody().empty()) {
    error = "normalized_mapper_body must contain exactly one nonempty func";
    return {};
  }
  return found;
}

static json::Object
graphFacts(const orbit::mapper_features::RouteExpandedGraph &graph) {
  json::Array types;
  for (int type : graph.nodeTypes)
    types.push_back(static_cast<int64_t>(type));
  json::Array edges;
  for (auto [source, target] : graph.edges)
    edges.push_back(json::Array{static_cast<int64_t>(source),
                                static_cast<int64_t>(target)});
  json::Object object;
  object["node_types"] = std::move(types);
  object["edges"] = std::move(edges);
  return object;
}

static bool fillFeatures(SelectedTask &task, MLIRContext *context,
                         const FormalMax4MLPEnsemble &model,
                         std::string &error) {
  if (!isFormalMax4MapperShape(task.mapperRows, task.mapperCols)) {
    error = "original selected shape maps to an unsupported formal model "
            "orientation for " +
            task.metadata.name;
    return false;
  }
  OwningOpRef<ModuleOp> parsedModule;
  func::FuncOp mapper = parseMapperFunction(
      context, task.body.normalizedMapperBody, parsedModule, error);
  if (!mapper)
    return false;
  std::string mapperText;
  llvm::raw_string_ostream mapperOS(mapperText);
  mapper.print(mapperOS);
  mapperOS.flush();

  orbit::mapper_features::RouteExpandedGraph graph;
  if (!orbit::mapper_features::parseRouteExpandedDFG(
          mapperText, graph, error, /*requireExpanded=*/true)) {
    error = "selected profiled body for " + task.metadata.name +
            " is not a complete route-expanded DFG: " + error;
    return false;
  }
  if (!computeMapperAnalyticalFacts(mapper.getBody(), task.mapperRows,
                                    task.mapperCols, task.recMII, task.resMII,
                                    task.lowerBound, error)) {
    error = "cannot derive mapper RecMII/ResMII from selected body for " +
            task.metadata.name + ": " + error;
    return false;
  }
  if (!orbit::mapper_features::computeMapperFeatures(
          graph, static_cast<int>(task.mapperRows),
          static_cast<int>(task.mapperCols), task.recMII, task.resMII,
          task.lowerBound, task.features, error)) {
    error = "cannot compute formal mapper features for " + task.metadata.name +
            ": " + error;
    return false;
  }
  json::Object graphJSON = graphFacts(graph);
  if (!deriveStartupCyclesFromCppFeatureOutput(graphJSON, task.startupCycles,
                                               error)) {
    error = "cannot derive structural startup for " + task.metadata.name +
            ": " + error;
    return false;
  }
  const auto &names = orbit::mapper_features::mapperFeatureNames();
  if (!model.featureNamesMatch(
          ArrayRef<std::string>(names.data(), names.size()))) {
    error = "C++ feature frontend feature_names differ from the formal model";
    return false;
  }
  if (!model.predict(ArrayRef<double>(task.features.values.data(),
                                      task.features.values.size()),
                     task.lowerBound, task.prediction, error))
    return false;
  if (!std::isfinite(task.prediction.predictedII) ||
      task.prediction.predictedII < task.lowerBound ||
      task.prediction.predictedII > 20.0 + 1.0e-6 ||
      !std::isfinite(task.prediction.predictedIIStd) ||
      task.prediction.predictedIIStd < 0.0) {
    error = "selected-shape model prediction violates finite/lower-bound/"
            "ceiling contract for " +
            task.metadata.name;
    return false;
  }
  return true;
}

static bool fillDirectShapePredictions(
    SelectedTask &task, MLIRContext *context,
    const PerCgra2x2DirectEnsemble &model, int64_t perCgraRows,
    int64_t perCgraCols, std::string &error) {
  static constexpr std::array<std::pair<int64_t, int64_t>, 8> kCgraShapes =
      {{{1, 1}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, {1, 4}, {2, 2}, {4, 1}}};
  const double runtimeCeiling = model.getRuntimeIICeiling();
  const auto &names = orbit::mapper_features::perCgra2x2MapperFeatureNames();
  if (!model.featureNamesMatch(ArrayRef<std::string>(names.data(),
                                                     names.size()))) {
    error = "C++ feature frontend feature_names differ from the direct 2x2 "
            "model";
    return false;
  }

  OwningOpRef<ModuleOp> parsedModule;
  func::FuncOp mapper = parseMapperFunction(
      context, task.body.normalizedMapperBody, parsedModule, error);
  if (!mapper)
    return false;
  std::string mapperText;
  llvm::raw_string_ostream mapperOS(mapperText);
  mapper.print(mapperOS);
  mapperOS.flush();

  orbit::mapper_features::RouteExpandedGraph graph;
  if (!orbit::mapper_features::parseRouteExpandedDFG(
          mapperText, graph, error, /*requireExpanded=*/true)) {
    error = "selected profiled body for " + task.metadata.name +
            " is not a complete route-expanded DFG: " + error;
    return false;
  }
  const json::Object graphJSON = graphFacts(graph);
  double startupCycles = 0.0;
  if (!deriveStartupCyclesFromCppFeatureOutput(graphJSON, startupCycles,
                                              error)) {
    error = "cannot derive structural startup for " + task.metadata.name +
            ": " + error;
    return false;
  }

  task.directPredictions.clear();
  task.directPredictions.reserve(kCgraShapes.size());
  for (auto [cgraRows, cgraCols] : kCgraShapes) {
    if (cgraRows > std::numeric_limits<int64_t>::max() / perCgraRows ||
        cgraCols > std::numeric_limits<int64_t>::max() / perCgraCols) {
      error = "direct selected-shape mapper dimensions overflow for " +
              task.metadata.name;
      return false;
    }
    SelectedTask::DirectShapePrediction item;
    item.cgraRows = cgraRows;
    item.cgraCols = cgraCols;
    item.mapperRows = cgraRows * perCgraRows;
    item.mapperCols = cgraCols * perCgraCols;
    item.startupCycles = startupCycles;
    if (!isPerCgra2x2MapperShape(item.mapperRows, item.mapperCols)) {
      error = "original selected shape maps outside the direct 2x2 model "
              "protocol for " +
              task.metadata.name;
      return false;
    }
    if (!computeMapperAnalyticalFacts(mapper.getBody(), item.mapperRows,
                                      item.mapperCols, item.recMII,
                                      item.resMII, item.lowerBound, error)) {
      error = "cannot derive direct mapper RecMII/ResMII for task=" +
              task.metadata.name + " shape=" + std::to_string(cgraRows) +
              "x" + std::to_string(cgraCols) + ": " + error;
      return false;
    }
    if (item.lowerBound > runtimeCeiling) {
      item.supported = false;
      task.directPredictions.push_back(std::move(item));
      continue;
    }

    orbit::mapper_features::PerCgra2x2FeatureVector features;
    if (!orbit::mapper_features::computePerCgra2x2MapperFeatures(
            graph, static_cast<int>(item.mapperRows),
            static_cast<int>(item.mapperCols), item.recMII, item.resMII,
            item.lowerBound, features, error,
            model.hasDiagnosticOverride()
                ? orbit::mapper_features::PerCgra2x2MapperFeatureInterval::Diagnostic23
                : orbit::mapper_features::PerCgra2x2MapperFeatureInterval::Training20)) {
      error = "direct feature extraction failed for task=" +
              task.metadata.name + " shape=" + std::to_string(cgraRows) +
              "x" + std::to_string(cgraCols) + ": " + error;
      return false;
    }
    DirectMapperIIPrediction prediction;
    if (!model.predict(ArrayRef<double>(features.values.data(),
                                        features.values.size()),
                       item.recMII, item.resMII, item.lowerBound, prediction,
                       error)) {
      error = "direct model inference failed for task=" + task.metadata.name +
              " shape=" + std::to_string(cgraRows) + "x" +
              std::to_string(cgraCols) + ": " + error;
      return false;
    }
    if (!std::isfinite(prediction.predictedII) ||
        prediction.predictedII < item.lowerBound ||
        prediction.predictedII > runtimeCeiling + 1.0e-6 ||
        prediction.memberPredictions.size() != 4) {
      error = "direct model prediction violates finite/lower-bound/ceiling/"
              "four-member contract for task=" +
              task.metadata.name + " shape=" + std::to_string(cgraRows) +
              "x" + std::to_string(cgraCols);
      return false;
    }
    double variance = 0.0;
    for (double member : prediction.memberPredictions) {
      if (!std::isfinite(member) || member < item.lowerBound || member > runtimeCeiling) {
        error = "direct member prediction violates its interval contract for " +
                task.metadata.name;
        return false;
      }
      const double delta = member - prediction.predictedII;
      variance += delta * delta;
    }
    item.predictedIIStd = std::sqrt(variance / 4.0);
    item.memberPredictions = std::move(prediction.memberPredictions);
    item.supported = true;
    task.directPredictions.push_back(std::move(item));
  }
  return true;
}

static json::Array makeReplicaInventoryJson(
    ArrayRef<SelectedReplicaPlacement> replicas) {
  json::Array inventory;
  for (const SelectedReplicaPlacement &replica : replicas) {
    json::Object record;
    record["replica_id"] = replica.replicaId;
    record["shape"] = replica.shape;
    record["cgra_count"] = replica.cgraCount;
    record["row"] = replica.row;
    record["col"] = replica.col;
    record["rows"] = replica.rows;
    record["cols"] = replica.cols;
    json::Array cells;
    json::Array contexts;
    for (const SelectedReplicaCell &cell : replica.cells) {
      cells.push_back(json::Object{{"row", cell.row}, {"col", cell.col},
                                   {"replica_id", replica.replicaId},
                                   {"context_id", cell.contextId}});
      contexts.push_back(cell.contextId);
    }
    record["actual_cells"] = std::move(cells);
    record["context_ids"] = std::move(contexts);
    inventory.push_back(std::move(record));
  }
  return inventory;
}

static json::Object
makeBinding(const OriginalAmoebaProfileBodyExport &bodies,
            StringRef profilePath, StringRef graphVariant,
            StringRef architecturePath, StringRef ensemblePath,
            StringRef checkpointDirectory, StringRef sourceRepository,
            StringRef sourceCommit, const FormalMax4MLPEnsemble &model,
            const MLCostCacheResources &modelResources,
            ArrayRef<SelectedTask> selected) {
  json::Object binding;
  binding["schema"] = kOriginalAmoebaProfileBindingSchema.str();
  binding["candidate_id"] = "candidate-0";
  binding["function"] = bodies.function;
  binding["graph_variant_id"] = graphVariant.str();
  binding["body_export_format"] = kBodyExportFormat.str();
  binding["profile_file_format"] = kProfileFormat.str();
  binding["function_signature"] = bodies.functionSignature;
  binding["function_result_signature"] = bodies.functionResultSignature;
  binding["profile_file"] = profilePath.str();
  binding["architecture_path"] = architecturePath.str();
  binding["architecture_contract"] = model.getArchitectureContract().str();
  binding["architecture_text"] = modelResources.architectureText;
  binding["source_repository"] = sourceRepository.str();
  binding["source_commit"] = sourceCommit.str();
  binding["model"] = kModelName.str();
  binding["model_schema"] = model.getModelSchema().str();
  binding["feature_contract_id"] = model.getFeatureContractId().str();
  binding["feature_extractor"] = model.getFeatureExtractor().str();
  binding["shape_protocol_id"] = model.getShapeProtocolId().str();
  binding["ensemble_path"] = ensemblePath.str();
  binding["checkpoint_directory"] = checkpointDirectory.str();
  binding["ensemble_text"] = modelResources.ensembleText;
  json::Array checkpoints;
  for (const auto &[path, text] : modelResources.checkpointTexts)
    checkpoints.push_back(json::Object{{"name", path}, {"text", text}});
  binding["checkpoint_texts"] = std::move(checkpoints);
  binding["source_iteration_domain_coverage_verified"] = true;
  binding["current_ir_body_equivalence_verified"] = true;
  json::Array taskBindings;
  for (const SelectedTask &task : selected) {
    const auto &body = task.body;
    const auto &profile = task.profile;
    json::Object record;
    record["task"] = task.metadata.name;
    record["task_signature"] = body.taskSignature;
    record["counter_signature"] = body.counterSignature;
    record["kernel_binding_signature"] = body.kernelBindingSignature;
    record["normalized_mapper_body"] = body.normalizedMapperBody;
    record["static_trip_count"] = body.staticTripCount;
    record["selected_cgra_shape"] = profile.shape;
    record["selected_cgra_count"] = profile.cgraCount;
    record["mapper_tile_rows"] = task.mapperRows;
    record["mapper_tile_cols"] = task.mapperCols;
    record["compiled_ii"] = profile.compiledII;
    record["steps"] = profile.steps;
    record["sample_trip_count"] = profile.sampleTripCount;
    record["materialized_operation_count"] = profile.materializedOperationCount;
    record["estimated_latency"] = profile.estimatedLatency;
    record["mapper_succeeded"] = true;
    if (task.replicas.size() > 1)
      record["replicas"] = makeReplicaInventoryJson(task.replicas);
    record["source_iteration_domain_status"] =
        task.metadata.sourceIterationDomainStatus;
    record["source_iteration_domain_complete"] =
        task.metadata.sourceIterationDomainComplete;
    record["source_iteration_work_count"] =
        task.metadata.sourceIterationWorkCount;
    taskBindings.push_back(std::move(record));
  }
  binding["tasks"] = std::move(taskBindings);
  return binding;
}

static bool
writeCostCatalog(raw_ostream &os, func::FuncOp function,
                 StringRef modelNamespace, StringRef sourceRepository,
                 StringRef sourceCommit, StringRef architecturePath,
                 StringRef architectureContract, StringRef graphVariant,
                 const FormalMax4MLPEnsemble &model,
                 ArrayRef<SelectedTask> tasks, json::Object profileBinding) {
  json::Object root;
  root["schema"] = "amoeba-task-shape-cost";
  root["function"] = function.getSymName().str();
  root["namespace"] = modelNamespace.str();

  json::Object metadata;
  metadata["provenance_schema"] = kCostProvenanceSchema.str();
  metadata["predictor_source"] = "original-amoeba-selected-profile-adapter";
  metadata["source_repository"] = sourceRepository.str();
  metadata["source_commit"] = sourceCommit.str();
  metadata["architecture_path"] = architecturePath.str();
  metadata["architecture_schema"] = kArchitectureSchema.str();
  metadata["architecture_contract"] = architectureContract.str();
  metadata["source_graph_id"] = graphVariant.str();
  metadata["graph_variant_id_source"] =
      "preserved-source-id-or-original-amoeba-full-baseline-label";
  metadata["candidate_id_scheme"] = "candidate-<sequential-index>";
  metadata["candidate_count"] = 1;
  json::Array sourceTaskIds;
  for (const SelectedTask &task : tasks)
    sourceTaskIds.push_back(task.metadata.name);
  metadata["source_task_ids"] = std::move(sourceTaskIds);
  metadata["model"] = kModelName.str();
  metadata["model_schema"] = model.getModelSchema().str();
  metadata["model_status"] = kModelStatus.str();
  metadata["quality_status"] = kModelStatus.str();
  metadata["production_ready"] = false;
  metadata["iteration_domain_coverage_status"] =
      kOriginalAmoebaSourceDomainCoverageStatus.str();
  metadata["iteration_domain_coverage_verified"] = true;
  const bool hasMultiReplicaTask = llvm::any_of(
      tasks, [](const SelectedTask &task) { return task.replicas.size() > 1; });
  metadata["iteration_domain_coverage_evidence"] =
      hasMultiReplicaTask
          ? "every original task has a complete bound source-domain certificate, "
            "unit internal multiplicity, unchanged task firing extent, and its "
            "complete original scheduler replica placement inventory"
          : "every original task has a complete bound source-domain certificate, "
            "unit internal multiplicity, unchanged task firing extent, one active "
            "replica, and no replica or tiling lineage";
  metadata["body_equivalence_checked"] = true;
  metadata["feature_contract_id"] = model.getFeatureContractId().str();
  metadata["feature_extractor"] = model.getFeatureExtractor().str();
  metadata["feature_frontend"] =
      "archived-task-profiler-normalized-mapper-body-v1";
  metadata["shape_protocol_id"] = model.getShapeProtocolId().str();
  metadata["communication_latency"] = "scored_by_scheduler";
  metadata["ranking_policy"] =
      json::Object{{"objective", "predicted_scheduler_makespan"},
                   {"mapper_success_probability", "not_predicted"},
                   {"uses_mapper_success_probability", false}};
  metadata["selection_policy"] = "original-amoeba-selected-shape-only";
  root["predictor_metadata"] = std::move(metadata);
  root["original_amoeba_profile_binding"] = std::move(profileBinding);

  json::Array entries;
  for (const SelectedTask &task : tasks) {
    entries.push_back(
        json::Object{{"task", task.metadata.name},
                     {"mapper_tile_rows", task.mapperRows},
                     {"mapper_tile_cols", task.mapperCols},
                     {"support_status", "supported"},
                     {"predicted_ii", task.prediction.predictedII},
                     {"startup_cycles", task.startupCycles},
                     {"predicted_ii_std", task.prediction.predictedIIStd},
                     {"analytical_lower_bound", task.lowerBound},
                     {"rec_mii", task.recMII},
                     {"res_mii", task.resMII},
                     {"ii_mean_source", kModelName.str()},
                     {"model_status", kModelStatus.str()},
                     {"production_ready", false}});
  }
  root["entries"] = std::move(entries);
  os << json::Value(std::move(root)) << "\n";
  return true;
}

static json::Object makeDirectBinding(
    const OriginalAmoebaProfileBodyExport &bodies, StringRef profilePath,
    StringRef graphVariant, StringRef architecturePath,
    StringRef architectureContract, StringRef architectureText,
    StringRef ensemblePath, StringRef ensembleText,
    StringRef sourceRepository, StringRef sourceCommit,
    const json::Object &sourceModel, ArrayRef<int64_t> memberSeeds,
    ArrayRef<SelectedTask> selected, StringRef canonicalModuleWitness,
    int64_t runtimeIICeiling,
    StringRef sourceDomainCoverageStatus,
    const json::Object *diagnosticOverride) {
  json::Object binding;
  binding["schema"] = kOriginalAmoebaProfileBindingSchema.str();
  binding["candidate_id"] = "candidate-0";
  binding["function"] = bodies.function;
  binding["graph_variant_id"] = graphVariant.str();
  binding["body_export_format"] = kBodyExportFormat.str();
  binding["profile_file_format"] = kProfileFormat.str();
  binding["function_signature"] = bodies.functionSignature;
  binding["function_result_signature"] = bodies.functionResultSignature;
  binding["profile_file"] = profilePath.str();
  binding["architecture_path"] = architecturePath.str();
  binding["architecture_contract"] = architectureContract.str();
  binding["architecture_text"] = architectureText.str();
  binding["source_repository"] = sourceRepository.str();
  binding["source_commit"] = sourceCommit.str();
  binding["model_namespace"] = kPerCgra2x2ModelNamespace.str();
  binding["model"] = "orbit-cgra-ii-per-cgra-2x2-direct-4member-v1";
  binding["model_schema"] = kPerCgra2x2EnsembleSchema.str();
  binding["feature_contract_id"] = kPerCgra2x2FeatureContractId.str();
  binding["feature_extractor"] =
      "cgra_ii_predictor.mapper_model:mapper_feature_vector";
  binding["shape_protocol_id"] = kPerCgra2x2ShapeProtocolId.str();
  binding["ensemble_path"] = ensemblePath.str();
  binding["ensemble_text"] = ensembleText.str();
  binding["source_model"] = json::Object(sourceModel);
  binding["source_iteration_domain_coverage_verified"] = true;
  if (llvm::any_of(selected, [](const SelectedTask &task) {
        return task.replicas.size() > 1;
      })) {
    binding["canonical_module_witness"] = canonicalModuleWitness.str();
  }
  if (diagnosticOverride) {
    binding["source_iteration_domain_coverage_status"] =
        sourceDomainCoverageStatus.str();
    binding["diagnostic_override"] = json::Object(*diagnosticOverride);
  }
  binding["current_ir_body_equivalence_verified"] = true;
  json::Array members;
  for (auto [index, seed] : llvm::enumerate(memberSeeds))
    members.push_back(json::Object{{"member_index", static_cast<int64_t>(index)},
                                   {"seed", seed}});
  binding["direct_ensemble_members"] = std::move(members);

  json::Array taskBindings;
  for (const SelectedTask &task : selected) {
    json::Object record;
    record["task"] = task.metadata.name;
    record["task_signature"] = task.body.taskSignature;
    record["counter_signature"] = task.body.counterSignature;
    record["kernel_binding_signature"] = task.body.kernelBindingSignature;
    record["normalized_mapper_body"] = task.body.normalizedMapperBody;
    record["static_trip_count"] = task.body.staticTripCount;
    record["selected_cgra_shape"] = task.profile.shape;
    record["selected_cgra_count"] = task.profile.cgraCount;
    record["mapper_tile_rows"] = task.mapperRows;
    record["mapper_tile_cols"] = task.mapperCols;
    record["compiled_ii"] = task.profile.compiledII;
    record["steps"] = task.profile.steps;
    record["sample_trip_count"] = task.profile.sampleTripCount;
    record["materialized_operation_count"] =
        task.profile.materializedOperationCount;
    record["estimated_latency"] = task.profile.estimatedLatency;
    record["mapper_succeeded"] = true;
    if (task.replicas.size() > 1)
      record["replicas"] = makeReplicaInventoryJson(task.replicas);
    record["source_iteration_domain_status"] =
        task.metadata.sourceIterationDomainStatus;
    record["source_iteration_domain_complete"] =
        task.metadata.sourceIterationDomainComplete;
    record["source_iteration_work_count"] =
        task.metadata.sourceIterationWorkCount;
    if (diagnosticOverride) {
      record["source_iteration_multiplicity"] =
          task.metadata.sourceIterationMultiplicity;
      json::Array expandedInternalExtents;
      for (int64_t extent : task.metadata.expandedInternalExtents)
        expandedInternalExtents.push_back(extent);
      record["expanded_internal_extents"] =
          std::move(expandedInternalExtents);
    }
    json::Array shapePredictions;
    for (const auto &prediction : task.directPredictions) {
      json::Object shape;
      shape["cgra_rows"] = prediction.cgraRows;
      shape["cgra_cols"] = prediction.cgraCols;
      shape["mapper_tile_rows"] = prediction.mapperRows;
      shape["mapper_tile_cols"] = prediction.mapperCols;
      shape["support_status"] =
          prediction.supported ? "supported" : "unsupported";
      shape["analytical_lower_bound"] = prediction.lowerBound;
      if (diagnosticOverride) {
        shape["training_ceiling_ii"] = 20;
        shape["runtime_ceiling_ii"] = runtimeIICeiling;
      }
      if (prediction.supported) {
        shape["rec_mii"] = prediction.recMII;
        shape["res_mii"] = prediction.resMII;
      shape["startup_cycles"] = prediction.startupCycles;
      shape["predicted_ii_std"] = prediction.predictedIIStd;
      shape["ii_mean_source"] = "direct_four_member_arithmetic_mean";
      if (diagnosticOverride) {
        const bool extrapolated =
            prediction.lowerBound > 20.0 ||
            llvm::any_of(prediction.memberPredictions,
                         [](double member) { return member > 20.0; });
        shape["extrapolation_status"] =
            extrapolated ? "out-of-training-ceiling"
                         : "within-training-ceiling";
      }
      json::Array directMembers;
        for (auto [index, member] :
             llvm::enumerate(prediction.memberPredictions))
          directMembers.push_back(json::Object{
              {"member_index", static_cast<int64_t>(index)},
              {"seed", memberSeeds[index]},
              {"predicted_ii", member}});
        shape["direct_ensemble_members"] = std::move(directMembers);
      } else {
        if (diagnosticOverride)
          shape["status"] = kDirectUnsupportedStatus.str();
        shape["unsupported_reason"] = diagnosticOverride
                                           ? kDirectDiagnosticUnsupportedReason.str()
                                           : kDirectUnsupportedReason.str();
        shape["model_interval_max_ii"] = 20.0;
        if (diagnosticOverride)
          shape["extrapolation_status"] =
              "outside-diagnostic-runtime-domain";
      }
      shapePredictions.push_back(std::move(shape));
    }
    record["direct_shape_predictions"] = std::move(shapePredictions);
    taskBindings.push_back(std::move(record));
  }
  binding["tasks"] = std::move(taskBindings);
  return binding;
}

static bool writeDirectCostCatalog(
    raw_ostream &os, func::FuncOp function, StringRef modelNamespace,
    StringRef sourceRepository, StringRef sourceCommit,
    StringRef architecturePath, StringRef architectureContract,
    StringRef graphVariant, StringRef profilePath,
    const OriginalAmoebaProfileBodyExport &bodyExport,
    const PerCgra2x2DirectEnsemble &model, StringRef ensemblePath,
    StringRef ensembleText, StringRef architectureText,
    const json::Object &sourceModel, ArrayRef<int64_t> memberSeeds,
    ArrayRef<SelectedTask> selected, int64_t runtimeIICeiling,
    StringRef sourceDomainCoverageStatus,
    const json::Object *diagnosticOverride) {
  json::Object root;
  root["schema"] = "amoeba-task-shape-cost";
  root["function"] = function.getSymName().str();
  root["namespace"] = modelNamespace.str();

  json::Object metadata;
  metadata["provenance_schema"] = kCostProvenanceSchema.str();
  metadata["predictor_source"] =
      "per-cgra-2x2-direct-four-member-ii-predictor";
  metadata["source_repository"] = sourceRepository.str();
  metadata["source_commit"] = sourceCommit.str();
  metadata["architecture_path"] = architecturePath.str();
  metadata["architecture_schema"] = kArchitectureSchema.str();
  metadata["architecture_contract"] = architectureContract.str();
  metadata["source_graph_id"] = graphVariant.str();
  if (runtimeIICeiling == 23)
    metadata["canonical_module_witness"] =
        neighborhoodReplaySourceText(function->getParentOfType<ModuleOp>());
  metadata["candidate_id_scheme"] = "candidate-<sequential-index>";
  metadata["candidate_count"] = 1;
  json::Array sourceTaskIds;
  for (const SelectedTask &task : selected)
    sourceTaskIds.push_back(task.metadata.name);
  metadata["source_task_ids"] = std::move(sourceTaskIds);
  metadata["model"] = "orbit-cgra-ii-per-cgra-2x2-direct-4member-v1";
  metadata["model_namespace"] = modelNamespace.str();
  metadata["model_schema"] = kPerCgra2x2EnsembleSchema.str();
  metadata["model_status"] =
      "candidate_pending_amoeba_benchmark_overlap_audit";
  metadata["quality_status"] =
      "candidate_pending_amoeba_benchmark_overlap_audit";
  metadata["candidate_only"] = true;
  metadata["amoeba_benchmark_overlap_audit_complete"] = false;
  metadata["old_4x4_labels_reused"] = false;
  metadata["supports_whole_program_latency_or_throughput_claim"] = false;
  metadata["production_ready"] = false;
  metadata["iteration_domain_coverage_status"] =
      sourceDomainCoverageStatus.str();
  metadata["iteration_domain_coverage_verified"] = true;
  const bool hasMultiReplicaTask = llvm::any_of(
      selected, [](const SelectedTask &task) { return task.replicas.size() > 1; });
  metadata["iteration_domain_coverage_evidence"] =
      diagnosticOverride
          ? (hasMultiReplicaTask
                 ? "every original task has a re-resolved complete source-domain "
                   "certificate; static internal loops are admitted only when fully "
                   "expanded into the mapper body, macro firings remain separate "
                   "from expanded source work, and all original scheduler replica "
                   "placements are retained"
                 : "every original task has a re-resolved complete source-domain "
                   "certificate; static internal loops are admitted only when fully "
                   "expanded into the mapper body, with macro firings distinct from "
                   "expanded source work; one active replica and no replica or tiling lineage")
          : (hasMultiReplicaTask
                 ? "every original task has a complete bound source-domain certificate, "
                   "unit internal multiplicity, unchanged task firing extent, and its "
                   "complete original scheduler replica placement inventory"
                 : "every original task has a complete bound source-domain certificate, "
                   "unit internal multiplicity, unchanged task firing extent, one active "
                   "replica, and no replica or tiling lineage");
  metadata["body_equivalence_checked"] = true;
  metadata["feature_contract_id"] = model.getFeatureContractId().str();
  metadata["feature_extractor"] =
      "cgra_ii_predictor.mapper_model:mapper_feature_vector";
  metadata["feature_frontend"] =
      "c++-mlir-current-body-direct-148-v1";
  metadata["shape_protocol_id"] = model.getShapeProtocolId().str();
  metadata["communication_latency"] = "scored_by_scheduler";
  metadata["ranking_policy"] =
      json::Object{{"objective", "predicted_scheduler_makespan"},
                   {"mapper_success_probability", "not_predicted"},
                   {"uses_mapper_success_probability", false}};
  metadata["selection_policy"] = "original-amoeba-selected-shape-only";
  metadata["model_interval_max_ii"] = 20;
  metadata["unsupported_prediction_policy"] =
      diagnosticOverride
          ? kDirectDiagnosticUnsupportedPolicy.str()
          : "analytical-lower-bound-exceeds-model-ceiling-v1";
  if (diagnosticOverride) {
    metadata["diagnostic_only"] = true;
    metadata["formal"] = false;
    metadata["diagnostic_override"] = json::Object(*diagnosticOverride);
  }
  metadata["source_model"] = json::Object(sourceModel);
  json::Array memberSeedsJson;
  for (int64_t seed : memberSeeds)
    memberSeedsJson.push_back(seed);
  metadata["direct_ensemble"] =
      json::Object{{"member_count", 4},
                   {"member_seeds", std::move(memberSeedsJson)},
                   {"reduction", "arithmetic_mean"},
                   {"uncertainty", "population_standard_deviation"}};
  root["predictor_metadata"] = std::move(metadata);
  const std::string canonicalModuleWitness =
      hasMultiReplicaTask ? neighborhoodReplaySourceText(
                                function->getParentOfType<ModuleOp>())
                          : std::string();
  if (hasMultiReplicaTask && canonicalModuleWitness.empty())
    return false;
  root["original_amoeba_profile_binding"] = makeDirectBinding(
      bodyExport, profilePath, graphVariant, architecturePath,
      architectureContract, architectureText, ensemblePath, ensembleText,
      sourceRepository, sourceCommit, sourceModel, memberSeeds, selected,
      canonicalModuleWitness, runtimeIICeiling, sourceDomainCoverageStatus,
      diagnosticOverride);

  json::Array entries;
  for (const SelectedTask &task : selected) {
    for (const auto &prediction : task.directPredictions) {
      if (!prediction.supported) {
        json::Object entry{
            {"task", task.metadata.name},
            {"mapper_tile_rows", prediction.mapperRows},
            {"mapper_tile_cols", prediction.mapperCols},
            {"support_status", "unsupported"},
            {"status", kDirectUnsupportedStatus.str()},
            {"unsupported_reason", diagnosticOverride
                                       ? kDirectDiagnosticUnsupportedReason.str()
                                       : kDirectUnsupportedReason.str()},
            {"analytical_lower_bound", prediction.lowerBound},
            {"model_interval_max_ii", 20}};
        if (diagnosticOverride) {
          entry["training_ceiling_ii"] = 20;
          entry["runtime_ceiling_ii"] = runtimeIICeiling;
          entry["extrapolation_status"] =
              "outside-diagnostic-runtime-domain";
        }
        entries.push_back(std::move(entry));
        continue;
      }
      double predictedII = 0.0;
      for (double member : prediction.memberPredictions)
        predictedII += member;
      predictedII /= static_cast<double>(prediction.memberPredictions.size());
      json::Array members;
      for (auto [index, member] :
           llvm::enumerate(prediction.memberPredictions))
        members.push_back(json::Object{
            {"member_index", static_cast<int64_t>(index)},
            {"seed", memberSeeds[index]},
            {"predicted_ii", member}});
      json::Object entry{
          {"task", task.metadata.name},
          {"mapper_tile_rows", prediction.mapperRows},
          {"mapper_tile_cols", prediction.mapperCols},
          {"support_status", "supported"},
          {"candidate_only", true},
          {"predicted_ii", predictedII},
          {"predicted_ii_std", prediction.predictedIIStd},
          {"direct_ensemble_members", std::move(members)},
          {"analytical_lower_bound", prediction.lowerBound},
          {"rec_mii", prediction.recMII},
          {"res_mii", prediction.resMII},
          {"startup_cycles", prediction.startupCycles},
          {"ii_mean_source", "direct_four_member_arithmetic_mean"},
          {"model_status",
           "candidate_pending_amoeba_benchmark_overlap_audit"},
          {"production_ready", false}};
      if (diagnosticOverride) {
        const bool extrapolated =
            prediction.lowerBound > 20.0 ||
            llvm::any_of(prediction.memberPredictions,
                         [](double member) { return member > 20.0; });
        entry["training_ceiling_ii"] = 20;
        entry["runtime_ceiling_ii"] = runtimeIICeiling;
        entry["extrapolation_status"] =
            extrapolated ? "out-of-training-ceiling"
                         : "within-training-ceiling";
      }
      entries.push_back(std::move(entry));
    }
  }
  root["entries"] = std::move(entries);
  os << json::Value(std::move(root)) << "\n";
  return true;
}

} // namespace

bool computeOriginalAmoebaNormalizedMapperBody(
    Operation *taskOperation, std::string &normalizedMapperBody,
    std::string &error) {
  auto task = dyn_cast_or_null<::mlir::taskflow::TaskflowTaskOp>(taskOperation);
  if (!task) {
    error = "current body-equivalence operation is not a taskflow.task";
    return false;
  }
  llvm::SmallVector<::mlir::neura::KernelOp> kernels;
  task.walk([&](::mlir::neura::KernelOp kernel) { kernels.push_back(kernel); });
  if (kernels.size() != 1) {
    error = "current task must contain exactly one neura.kernel for exact "
            "original mapper-body reconstruction";
    return false;
  }
  return reconstructOriginalNormalizedMapperBody(kernels.front(),
                                                 normalizedMapperBody, error);
}

bool readOriginalAmoebaProfileBodyExport(
    StringRef path, StringRef expectedFunction,
    ArrayRef<std::string> expectedTasks,
    OriginalAmoebaProfileBodyExport &result, std::string &error) {
  json::Object root;
  if (!readObjectFile(path, root, error))
    return false;
  std::string format;
  if (!readJsonString(root, "format", format, error) ||
      format != kBodyExportFormat) {
    if (error.empty())
      error = "body export format is not " + kBodyExportFormat.str();
    return false;
  }
  std::string function;
  int64_t taskCount = 0;
  if (!readJsonString(root, "function", function, error) ||
      !readJsonInteger(root, "task_count", taskCount, error) ||
      !readJsonString(root, "function_signature", result.functionSignature,
                      error) ||
      !readJsonString(root, "function_result_signature",
                      result.functionResultSignature, error))
    return false;
  if (function != expectedFunction || taskCount < 1 ||
      taskCount != static_cast<int64_t>(expectedTasks.size())) {
    error = "body export function/task count does not exactly match current IR";
    return false;
  }
  const json::Array *rawTasks = root.getArray("tasks");
  if (!rawTasks || rawTasks->size() != expectedTasks.size()) {
    error = "body export does not contain exact task coverage";
    return false;
  }
  result.function = function;
  result.tasks.clear();
  const std::set<std::string> expectedTaskSet(expectedTasks.begin(),
                                              expectedTasks.end());
  for (const json::Value &rawTask : *rawTasks) {
    const json::Object *object = rawTask.getAsObject();
    if (!object) {
      error = "body export task entry is not an object";
      return false;
    }
    OriginalAmoebaProfileBodyEvidence body;
    if (!readJsonString(*object, "task", body.task, error) ||
        !readJsonInteger(*object, "static_trip_count", body.staticTripCount,
                         error) ||
        !readJsonString(*object, "task_signature", body.taskSignature, error) ||
        !readJsonString(*object, "counter_signature", body.counterSignature,
                        error) ||
        !readJsonString(*object, "kernel_binding_signature",
                        body.kernelBindingSignature, error) ||
        !readJsonString(*object, "normalized_mapper_body",
                        body.normalizedMapperBody, error))
      return false;
    std::string taskName = body.task;
    if (!expectedTaskSet.count(taskName) || body.staticTripCount < 1 ||
        !result.tasks.emplace(std::move(taskName), std::move(body)).second) {
      error = "body export task identity is not an exact permutation of "
              "current IR tasks";
      return false;
    }
  }
  return true;
}

namespace {

struct OriginalAmoebaProfileCostAdapterPass
    : PassWrapper<OriginalAmoebaProfileCostAdapterPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OriginalAmoebaProfileCostAdapterPass)

  OriginalAmoebaProfileCostAdapterPass() = default;
  OriginalAmoebaProfileCostAdapterPass(
      const OriginalAmoebaProfileCostAdapterPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "adapt-original-amoeba-profile-costs";
  }
  StringRef getDescription() const override {
    return "Predict selected original AMOEBA shape costs from exact profile "
           "body evidence";
  }

  Option<std::string> functionName{
      *this, "function", llvm::cl::desc("Original Taskflow function."),
      llvm::cl::init("")};
  Option<std::string> bodyExportFile{
      *this, "body-export-file",
      llvm::cl::desc("Exact archived mapper-body proof JSON."),
      llvm::cl::init("")};
  Option<std::string> profileFile{
      *this, "profile-file",
      llvm::cl::desc("Complete original AMOEBA selected-shape profile JSON."),
      llvm::cl::init("")};
  Option<std::string> ensembleFile{*this, "ensemble-file",
                                   llvm::cl::desc("Formal JSON ensemble."),
                                   llvm::cl::init("")};
  Option<std::string> checkpointDirectory{
      *this, "checkpoint-dir",
      llvm::cl::desc("Formal JSON checkpoint directory."), llvm::cl::init("")};
  Option<std::string> architectureContract{
      *this, "architecture-contract",
      llvm::cl::desc("Neura architecture and shape contract."),
      llvm::cl::init("")};
  Option<std::string> architecturePath{
      *this, "architecture-path", llvm::cl::desc("Target architecture path."),
      llvm::cl::init("")};
  Option<std::string> sourceRepository{*this, "source-git-repository",
                                       llvm::cl::desc("Source repository."),
                                       llvm::cl::init("")};
  Option<std::string> sourceCommit{*this, "source-git-commit",
                                   llvm::cl::desc("Source Git commit."),
                                   llvm::cl::init("")};
  Option<std::string> modelNamespace{
      *this, "model-namespace", llvm::cl::desc("Output catalogue namespace."),
      llvm::cl::init("formal-max4-nohash-v2-original-amoeba-exploratory")};
  Option<std::string> outputFile{
      *this, "output", llvm::cl::desc("Atomic task-shape catalogue path."),
      llvm::cl::init("")};
  Option<int64_t> diagnosticIICeiling{
      *this, "diagnostic-ii-ceiling",
      llvm::cl::desc("Training ceiling 20 or explicit non-formal runtime 23."),
      llvm::cl::init(20)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](StringRef message) {
      module.emitError() << message;
      signalPassFailure();
    };
    if (diagnosticIICeiling != 20 && diagnosticIICeiling != 23) {
      fail("diagnostic-ii-ceiling accepts only 20 (default) or 23");
      return;
    }
    if (diagnosticIICeiling == 23 &&
        StringRef(modelNamespace.getValue()) != kPerCgra2x2ModelNamespace) {
      fail("diagnostic II23 requires the direct per-CGRA 2x2 model namespace");
      return;
    }
    if (functionName.empty() || bodyExportFile.empty() || profileFile.empty() ||
        ensembleFile.empty() || checkpointDirectory.empty() ||
        architectureContract.empty() || architecturePath.empty() ||
        sourceRepository.empty() || sourceCommit.empty() ||
        outputFile.empty()) {
      fail("adapt-original-amoeba-profile-costs requires function, "
           "body-export-file, profile-file, ensemble-file, checkpoint-dir, "
           "architecture-contract, architecture-path, source-git-repository, "
           "source-git-commit, and output");
      return;
    }
    if (!StringRef(architectureContract)
             .starts_with("neura-architecture-v1:") ||
        !llvm::sys::fs::exists(architecturePath)) {
      fail("architecture contract/path is not a configured Neura architecture");
      return;
    }

    std::string error;
    FailureOr<func::FuncOp> selected =
        selectTaskFunction(module, functionName, error);
    if (failed(selected)) {
      fail(error);
      return;
    }
    func::FuncOp function = *selected;
    auto graph = function->getAttrOfType<StringAttr>("amoeba.graph_variant_id");
    const bool assignedOriginalGraphId = !graph || graph.getValue().empty();
    if (assignedOriginalGraphId) {
      graph = StringAttr::get(module.getContext(),
                              kOriginalBaselineGraphVariantId);
      function->setAttr("amoeba.graph_variant_id", graph);
    }
    auto orchestration =
        function->getAttrOfType<DictionaryAttr>("task_orchestration_summary");
    auto strategy = orchestration ? orchestration.getAs<StringAttr>("strategy")
                                  : StringAttr();
    if (!graph || graph.getValue().empty() || !strategy ||
        strategy.getValue() != "throughput-guided") {
      fail("selected function must carry its graph variant and original "
           "throughput-guided strategy evidence");
      return;
    }

    FailureOr<llvm::SmallVector<TaskMetadata>> collected =
        collectAnalyticalTaskMetadata(function, error);
    if (failed(collected) || collected->empty()) {
      fail(error.empty() ? "selected function has no Taskflow tasks" : error);
      return;
    }
    llvm::SmallVector<TaskMetadata> tasks = std::move(*collected);
    const bool hasScheduledReplicaInventory = llvm::any_of(
        tasks, [](const TaskMetadata &task) {
          auto active =
              task.op->getAttrOfType<IntegerAttr>("active_replicas");
          return active && active.getInt() > 1;
        });
    if (!verifyOriginalAmoebaSourceDomainCoverage(
            tasks, error, diagnosticIICeiling == 23,
            hasScheduledReplicaInventory)) {
      fail(error);
      return;
    }
    std::vector<std::string> taskNames;
    taskNames.reserve(tasks.size());
    for (const TaskMetadata &task : tasks)
      taskNames.push_back(task.name);
    OriginalAmoebaProfileBodyExport bodyExport;
    if (!readOriginalAmoebaProfileBodyExport(bodyExportFile,
                                             function.getSymName(), taskNames,
                                             bodyExport, error)) {
      fail(error);
      return;
    }
    json::Object profiles;
    if (!readObjectFile(profileFile, profiles, error)) {
      fail(error);
      return;
    }
    std::string profileFormat;
    std::string profileFunction;
    int64_t profileTaskCount = 0;
    int64_t completedCandidates = 0;
    int64_t expectedCandidates = 0;
    if (!readJsonString(profiles, "format", profileFormat, error) ||
        !readJsonString(profiles, "function", profileFunction, error) ||
        !readJsonInteger(profiles, "task_count", profileTaskCount, error) ||
        !readJsonInteger(profiles, "completed_candidate_count",
                         completedCandidates, error) ||
        !readJsonInteger(profiles, "expected_candidate_count",
                         expectedCandidates, error)) {
      fail(error);
      return;
    }
    const int64_t expectedProfileCount =
        static_cast<int64_t>(tasks.size()) * kExpectedProfileShapes.size();
    if (profileFormat != kProfileFormat ||
        profileFunction != function.getSymName() ||
        profileTaskCount != static_cast<int64_t>(tasks.size()) ||
        completedCandidates != expectedCandidates ||
        expectedCandidates != expectedProfileCount) {
      fail("profile file format/function/task coverage is incomplete or stale");
      return;
    }
    const json::Array *profileTasks = profiles.getArray("tasks");
    const bool hasCandidateAttemptsField =
        profiles.get("candidate_attempts") != nullptr;
    const json::Array *candidateAttempts =
        profiles.getArray("candidate_attempts");
    if (!profileTasks || profileTasks->size() != tasks.size() ||
        (hasCandidateAttemptsField && !candidateAttempts) ||
        (!candidateAttempts && diagnosticIICeiling != 20) ||
        (candidateAttempts &&
         candidateAttempts->size() !=
             static_cast<size_t>(expectedProfileCount))) {
      fail("profile file does not contain exact task coverage");
      return;
    }
    std::set<std::string> expectedProfileTasks(taskNames.begin(),
                                               taskNames.end());
    std::set<std::string> observedProfileTasks;
    for (const json::Value &rawTask : *profileTasks) {
      const json::Object *record = rawTask.getAsObject();
      std::optional<StringRef> taskName =
          record ? record->getString("task") : std::nullopt;
      if (!taskName || !expectedProfileTasks.count(taskName->str()) ||
          !observedProfileTasks.insert(taskName->str()).second) {
        fail("profile task identities do not form an exact unique permutation "
             "of current Taskflow tasks");
        return;
      }
    }
    if (candidateAttempts) {
      std::set<std::string> attemptedProfileTasks;
      for (const json::Value &rawAttempt : *candidateAttempts) {
        const json::Object *attempt = rawAttempt.getAsObject();
        auto taskName = attempt ? attempt->getString("task") : std::nullopt;
        if (!taskName || !expectedProfileTasks.count(taskName->str())) {
          fail("candidate-attempt inventory names a malformed or unknown task");
          return;
        }
        attemptedProfileTasks.insert(taskName->str());
      }
      if (attemptedProfileTasks != expectedProfileTasks) {
        fail("candidate-attempt inventory does not cover every current task");
        return;
      }
    }

    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    const int64_t perCgraRows = architecture.getPerCgraRows();
    const int64_t perCgraCols = architecture.getPerCgraColumns();
    if (architecture.getMultiCgraRows() != 4 ||
        architecture.getMultiCgraColumns() != 4 || perCgraRows <= 0 ||
        perCgraCols <= 0) {
      fail("configured target must be a validated 4x4 multi-CGRA architecture");
      return;
    }

    std::string architectureText;
    if (!readTextFile(architecturePath, architectureText, error)) {
      fail(error);
      return;
    }
    json::Object ensembleObject;
    if (!readObjectFile(ensembleFile, ensembleObject, error)) {
      fail(error);
      return;
    }
    std::string ensembleText;
    auto ensembleBuffer = llvm::MemoryBuffer::getFile(ensembleFile);
    if (!ensembleBuffer) {
      fail("cannot read exact model bundle text: " +
           ensembleBuffer.getError().message());
      return;
    }
    ensembleText = (*ensembleBuffer)->getBuffer().str();
    std::string bundleSchema;
    if (!readJsonString(ensembleObject, "schema", bundleSchema, error)) {
      fail(error);
      return;
    }
    const bool directModelMode =
        bundleSchema == kPerCgra2x2EnsembleSchema;
    if (directModelMode !=
        (StringRef(modelNamespace) == kPerCgra2x2ModelNamespace)) {
      fail("model bundle schema and explicit model namespace disagree; the "
           "direct 2x2 model requires its own namespace");
      return;
    }
    FormalMax4MLPEnsemble model;
    PerCgra2x2DirectEnsemble directModel;
    MLCostCacheResources modelResources;
    json::Object sourceModel;
    std::string trainingArchitectureText;
    std::vector<int64_t> directMemberSeeds;
    if (directModelMode) {
      const std::string expectedArchitectureContract =
          "neura-architecture-v1:amoeba_4x4_cgra_2x2_context6";
      const json::Object *bundleArchitecture =
          ensembleObject.getObject("architecture");
      const auto exactArchitecture =
          bundleArchitecture
              ? bundleArchitecture->getString("exact_yaml_text")
              : std::nullopt;
      const json::Object *sourceModelObject =
          ensembleObject.getObject("source_model");
      const json::Object *featureContract =
          ensembleObject.getObject("feature_contract");
      const auto extractor = featureContract
                                 ? featureContract->getString("extractor")
                                 : std::nullopt;
      const json::Array *members = ensembleObject.getArray("members");
      if (architectureContract != expectedArchitectureContract ||
          architecture.getPerCgraRows() != 2 ||
          architecture.getPerCgraColumns() != 2 || !exactArchitecture ||
          !sourceModelObject ||
          !extractor || !members || members->size() != 4) {
        fail("direct 2x2 model requires its exact context6 architecture text, "
             "four-member package, and explicit 2x2 architecture contract");
        return;
      }
      trainingArchitectureText = exactArchitecture->str();
      if (!validatePerCgra2x2RuntimeContract(
              diagnosticIICeiling, *exactArchitecture, architectureText, error)) {
        fail(error);
        return;
      }
      for (auto [index, rawMember] : llvm::enumerate(*members)) {
        const json::Object *member = rawMember.getAsObject();
        int64_t memberIndex = -1;
        int64_t seed = 0;
        if (!member ||
            !readJsonInteger(*member, "member_index", memberIndex, error) ||
            !readJsonInteger(*member, "seed", seed, error) ||
            memberIndex != static_cast<int64_t>(index)) {
          fail(error.empty()
                   ? "direct 2x2 model member order or seed metadata is invalid"
                   : error);
          return;
        }
        directMemberSeeds.push_back(seed);
      }
      const std::vector<int64_t> expectedSeeds = {17, 41, 113, 239};
      if (directMemberSeeds != expectedSeeds ||
          !directModel.load(ensembleFile, architectureText, diagnosticIICeiling, error) ||
          directModel.getModelNamespace() != kPerCgra2x2ModelNamespace ||
          directModel.getFeatureContractId() !=
              kPerCgra2x2FeatureContractId ||
          directModel.getShapeProtocolId() != kPerCgra2x2ShapeProtocolId) {
        fail(error.empty()
                 ? "direct 2x2 model package does not satisfy its bound schema"
                 : error);
        return;
      }
      sourceModel = json::Object(*sourceModelObject);
    } else {
      if (StringRef(modelNamespace) == kPerCgra2x2ModelNamespace) {
        fail("direct 2x2 namespace cannot load the legacy formal-max4 bundle");
        return;
      }
      if (!model.load(ensembleFile, checkpointDirectory, architectureContract,
                      error)) {
        fail(error);
        return;
      }
      modelResources = model.makeCacheResources(architectureText);
    }

    std::vector<SelectedTask> selectedTasks;
    selectedTasks.reserve(tasks.size());
    for (unsigned index = 0; index < tasks.size(); ++index) {
      TaskMetadata metadata = tasks[index];
      const auto bodyIt = bodyExport.tasks.find(metadata.name);
      if (bodyIt == bodyExport.tasks.end() ||
          bodyIt->second.staticTripCount != metadata.tripCount) {
        TaskflowTaskOp operation = metadata.op;
        operation.emitError() << "body export static trip count does not match "
                                 "the compiler-inferred current trip count";
        return signalPassFailure();
      }
      std::string currentMapperBody;
      if (!computeOriginalAmoebaNormalizedMapperBody(
              metadata.op.getOperation(), currentMapperBody, error) ||
          currentMapperBody != bodyIt->second.normalizedMapperBody) {
        metadata.op.emitError()
            << (error.empty()
                    ? "exported normalized mapper body does not exactly match "
                      "the current kernel after original TaskProfiler lowering"
                    : error);
        return signalPassFailure();
      }
      auto shapeAttr =
          metadata.op->getAttrOfType<StringAttr>("composed_cgra_shape");
      auto countAttr =
          metadata.op->getAttrOfType<IntegerAttr>("composed_cgra_count");
      auto profileInfo =
          metadata.op->getAttrOfType<DictionaryAttr>("profile_info");
      auto traceInfo = metadata.op->getAttrOfType<DictionaryAttr>(
          "amoeba.task_scheduler_schedule_info");
      auto activeReplicas =
          metadata.op->getAttrOfType<IntegerAttr>("active_replicas");
      int64_t selectedRows = 0;
      int64_t selectedCols = 0;
      if (!shapeAttr || !countAttr || !profileInfo || !traceInfo ||
          !activeReplicas || activeReplicas.getInt() < 1 ||
          activeReplicas.getInt() > 4 ||
          !parseSelectedShape(shapeAttr.getValue(), selectedRows,
                              selectedCols) ||
          countAttr.getInt() != selectedRows * selectedCols) {
        metadata.op.emitError() << "original selected shape/profile/replica "
                                   "evidence is missing or unsupported";
        return signalPassFailure();
      }
      auto traceShape = getString(traceInfo, "composed_cgra_shape");
      auto traceCount = getInteger(traceInfo, "composed_cgra_count");
      auto traceActive = getInteger(traceInfo, "active_replicas");
      if (!traceShape || *traceShape != shapeAttr.getValue() || !traceCount ||
          *traceCount != countAttr.getInt() || !traceActive ||
          *traceActive != activeReplicas.getInt()) {
        metadata.op.emitError() << "task and original schedule selected shape "
                                   "or replica evidence disagree";
        return signalPassFailure();
      }

      SelectedTask item;
      item.metadata = metadata;
      item.body = bodyIt->second;
      if (!readSelectedReplicaPlacements(metadata.op.getOperation(),
                                         selectedRows, selectedCols,
                                         countAttr.getInt(), item.replicas,
                                         error)) {
        metadata.op.emitError() << error;
        return signalPassFailure();
      }
      item.cgraRows = selectedRows;
      item.cgraCols = selectedCols;
      if (selectedRows > std::numeric_limits<int64_t>::max() / perCgraRows ||
          selectedCols > std::numeric_limits<int64_t>::max() / perCgraCols) {
        metadata.op.emitError() << "selected mapper tile dimensions overflow";
        return signalPassFailure();
      }
      item.mapperRows = selectedRows * perCgraRows;
      item.mapperCols = selectedCols * perCgraCols;
      if (!getSelectedProfile(profiles, metadata.name, shapeAttr.getValue(),
                              countAttr.getInt(), metadata.tripCount,
                              diagnosticIICeiling == 20 &&
                                  candidateAttempts == nullptr,
                              item.profile, error)) {
        metadata.op.emitError() << error;
        return signalPassFailure();
      }
      if (item.profile.compiledII > diagnosticIICeiling) {
        metadata.op.emitError()
            << "selected mapper II exceeds the runtime control-memory ceiling";
        return signalPassFailure();
      }
      auto irII = getInteger(profileInfo, "compiled_ii");
      auto irDuration = getInteger(profileInfo, "duration");
      auto irOperations =
          getInteger(profileInfo, "materialized_operation_count");
      auto irTrip = getInteger(profileInfo, "sample_trip_count");
      auto irSteps = getInteger(profileInfo, "steps");
      int64_t formulaDuration = 0;
      int64_t expectedF45Duration = 0;
      if (!irII || !irDuration || !irOperations || !irTrip || !irSteps ||
          *irII != item.profile.compiledII || *irDuration < 1 ||
          *irOperations != item.profile.materializedOperationCount ||
          *irTrip != metadata.tripCount ||
          *irTrip != item.profile.sampleTripCount ||
          *irSteps != item.profile.steps ||
          !checkedProfileDuration(*irII, *irTrip, *irSteps, formulaDuration) ||
          formulaDuration != item.profile.estimatedLatency ||
          !checkedF45SchedulerDuration(item.profile.estimatedLatency,
                                       activeReplicas.getInt(),
                                       expectedF45Duration) ||
          *irDuration != expectedF45Duration) {
        metadata.op.emitError()
            << "selected full mapper profile, original F45 scheduler duration, "
               "and compiler-inferred trip count do not match the exact "
               "replica-duration contract";
        return signalPassFailure();
      }
      const bool featuresReady = directModelMode
                                     ? fillDirectShapePredictions(
                                           item, module.getContext(),
                                           directModel, perCgraRows,
                                           perCgraCols, error)
                                     : fillFeatures(item, module.getContext(),
                                                    model, error);
      if (!featuresReady) {
        metadata.op.emitError() << error;
        return signalPassFailure();
      }
      selectedTasks.push_back(std::move(item));
    }

    if (sourceRepository.empty() || !isSourceCommit(sourceCommit)) {
      fail("source repository and valid Git commit provenance are required");
      return;
    }
    json::Object diagnosticOverride;
    const bool hasDiagnosticOverride = diagnosticIICeiling == 23;
    if (hasDiagnosticOverride &&
        !makePerCgra2x2DiagnosticOverrideMetadata(
            trainingArchitectureText, architectureText, diagnosticOverride,
            error)) {
      fail(error);
      return;
    }
    const StringRef sourceDomainCoverageStatus =
        originalAmoebaSourceDomainCoverageStatus(tasks);
    if (!writeAtomically(
            outputFile,
            [&](raw_ostream &os) {
              if (directModelMode) {
                return writeDirectCostCatalog(
                    os, function, modelNamespace, sourceRepository, sourceCommit,
                    architecturePath, architectureContract, graph.getValue(),
                    profileFile, bodyExport, directModel, ensembleFile,
                    ensembleText, architectureText, sourceModel,
                    directMemberSeeds, selectedTasks, diagnosticIICeiling,
                    sourceDomainCoverageStatus,
                    hasDiagnosticOverride ? &diagnosticOverride : nullptr);
              }
              auto binding = makeBinding(
                  bodyExport, profileFile, graph.getValue(), architecturePath,
                  ensembleFile, checkpointDirectory, sourceRepository,
                  sourceCommit, model, modelResources, selectedTasks);
              return writeCostCatalog(os, function, modelNamespace,
                                      sourceRepository, sourceCommit,
                                      architecturePath, architectureContract,
                                      graph.getValue(), model, selectedTasks,
                                      std::move(binding));
            },
            error)) {
      fail(error);
      return;
    }
  }
};

static std::unique_ptr<Pass> createOriginalAmoebaProfileCostAdapterImpl() {
  return std::make_unique<OriginalAmoebaProfileCostAdapterPass>();
}

} // namespace

} // namespace mlir::amoeba::neura::joint_scheduling

namespace mlir::amoeba::neura {
std::unique_ptr<Pass> createOriginalAmoebaProfileCostAdapterPass() {
  return joint_scheduling::createOriginalAmoebaProfileCostAdapterImpl();
}
} // namespace mlir::amoeba::neura

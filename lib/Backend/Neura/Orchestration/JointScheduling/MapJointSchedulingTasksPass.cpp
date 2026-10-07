//===- MapJointSchedulingTasksPass.cpp ----------------------------------===//
//
// Replays Neura mapping independently for each selected task rectangle.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "MapperCostAnalysis.h"
#include "CommonMapperReplayWrapper.h"
#include "SpatialTaskCandidateSpace.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"

#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <memory>

#include <cmath>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mlir;
using namespace mlir::taskflow;
using ::mlir::amoeba::neura::joint_scheduling::RectShape;
using ::mlir::amoeba::neura::joint_scheduling::buildCommonMapperReplayWrapper;
using ::mlir::amoeba::neura::joint_scheduling::printCommonMapperReplayWrapper;
namespace json = llvm::json;
namespace json = llvm::json;

namespace {

// A key stores exact compiler input text and architecture contents. Task
// names and graph identities do not enter it; no digest substitutes equality.
class MappingCacheLock {
  int fd = -1;
public:
  explicit MappingCacheLock(const std::string &path) {
    fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd >= 0) while (::flock(fd, LOCK_EX) != 0) {
      if (errno == EINTR) continue;
      ::close(fd); fd = -1; break;
    }
  }
  bool valid() const { return fd >= 0; }
  ~MappingCacheLock() { if (fd >= 0) { ::flock(fd, LOCK_UN); ::close(fd); } }
};

static bool hasAllUnitRewriteEvidence(func::FuncOp function,
                                     std::string &error) {
  bool foundEvidence = false;
  function->walk([&](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef name = attribute.getName().strref();
      if ((name == "joint_scheduling_graph_variant_id" ||
           name == "amoeba.graph_variant_id") &&
          operation == function.getOperation()) {
        auto graphVariant = dyn_cast<StringAttr>(attribute.getValue());
        // The source-owned shape enumerator labels its unchanged graph.
        // Permit only that identity label; rewrite evidence below and the
        // complete source-domain/shape checks remain mandatory.
        if (graphVariant && graphVariant.getValue() == "identity")
          continue;
      }
      if (name.starts_with("amoeba.replica.") ||
          name.starts_with("amoeba.tiling.") ||
          name.starts_with("amoeba.neura.tiling.") ||
          name.starts_with("amoeba.neura.fusion.") ||
          name.starts_with("amoeba.semantic.fusion_") ||
          name.starts_with("amoeba.fission.") ||
          name.starts_with("amoeba.fusion.") ||
          name == ::mlir::amoeba::neura::joint_scheduling::
                     kSourceIterationPartitionProofAttr ||
          name == "joint_scheduling_graph_variant_id" ||
          name == "amoeba.graph_variant_id") {
        error = "all-unit baseline rejects candidate graph rewrite evidence "
                "attribute " +
                name.str();
        foundEvidence = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return foundEvidence;
}

static bool hasParentProfileRewriteEvidence(func::FuncOp function,
                                            std::string &error) {
  if (hasAllUnitRewriteEvidence(function, error)) {
    error.replace(0, error.find(" rejects"), "parent profile export rejects");
    return true;
  }
  bool foundEvidence = false;
  function->walk([&](Operation *operation) {
    for (NamedAttribute attribute : operation->getAttrs()) {
      StringRef name = attribute.getName().strref();
      const bool aggregateMarker =
          name.starts_with("amoeba.aggregate.") ||
          name.starts_with("amoeba.original_amoeba.") ||
          ((name.starts_with("amoeba.") ||
            name.starts_with("joint_scheduling_")) &&
           name.contains("aggregate")) ||
          name == "amoeba.semantic.completion_only";
      if (!aggregateMarker)
        continue;
      error = "parent profile export rejects aggregate/replica rewrite "
              "evidence attribute " +
              name.str();
      foundEvidence = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return foundEvidence;
}

struct ParentProfileRow {
  std::string task;
  std::string shape;
  std::string preMapperWrapper;
  int64_t candidateIndex = 0;
  int64_t cgraCount = 0;
  int64_t mapperRows = 0;
  int64_t mapperCols = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t materializedOperationCount = 0;
  int64_t sampleTripCount = 0;
  int64_t estimatedLatency = 0;
  int64_t stepsFormulaLatency = 0;
  int64_t sourceIterationWorkCount = 0;
  std::string sourceIterationDomainStatus;
  double structuralStartupCycles = 0.0;
};

struct ParentProfileAttempt {
  std::string task;
  std::string shape;
  int64_t candidateIndex = 0;
  int64_t cgraCount = 0;
  bool profileCreated = false;
  bool mapperSucceeded = false;
  std::string failureReason;
};

static bool isNestedInFusedMapperOperation(Operation *operation) {
  for (Operation *parent = operation ? operation->getParentOp() : nullptr;
       parent; parent = parent->getParentOp()) {
    if (parent->getName().getStringRef().contains(
            ::mlir::neura::attr::val::kOpFused))
      return true;
  }
  return false;
}

static bool collectActualMapperProfileFacts(Region &body, int64_t &steps,
                                            int64_t &materializedOperations,
                                            std::string &error) {
  int64_t maximumTimeStep = -1;
  body.walk([&](Operation *operation) {
    // The wrapper terminator is a control-flow artifact, not a mapped Neura
    // operation. The mapper's materialization predicate intentionally only
    // filters Neura operations, so exclude func.return before applying it.
    if (isa<func::ReturnOp>(operation) ||
        isNestedInFusedMapperOperation(operation))
      return WalkResult::advance();

    const bool materialized =
        !::mlir::neura::is_non_materialized(operation);
    auto locations = operation->getAttrOfType<ArrayAttr>("mapping_locs");
    if (materialized) {
      if (materializedOperations == std::numeric_limits<int64_t>::max()) {
        error = "materialized mapper operation count exceeds int64";
        return WalkResult::interrupt();
      }
      ++materializedOperations;
      if (!locations) {
        error = "a materialized mapper operation has no mapping_locs";
        return WalkResult::interrupt();
      }
      if (locations.empty()) {
        error = "a materialized mapper operation has empty mapping_locs";
        return WalkResult::interrupt();
      }
    }

    if (!locations || locations.empty()) {
      return WalkResult::advance();
    }
    for (Attribute location : locations) {
      auto dictionary = dyn_cast<DictionaryAttr>(location);
      auto timeStep = dictionary
                          ? dictionary.getAs<IntegerAttr>("time_step")
                          : IntegerAttr();
      if (!timeStep || timeStep.getInt() < 0) {
        error = "mapper emitted a malformed mapping_locs time_step";
        return WalkResult::interrupt();
      }
      maximumTimeStep = std::max(maximumTimeStep, timeStep.getInt());
    }
    return WalkResult::advance();
  });
  if (!error.empty())
    return false;
  if (maximumTimeStep < 0 ||
      maximumTimeStep == std::numeric_limits<int64_t>::max() ||
      materializedOperations <= 0) {
    error = "mapper emitted no complete materialized mapping profile";
    return false;
  }
  steps = maximumTimeStep + 1;
  return true;
}

static bool deriveStructuralStartupCycles(func::FuncOp mapperWrapper,
                                          double &startup,
                                          std::string &error) {
  // This is the same C++ mapper frontend used by the model cost catalogue.
  // It builds structural graph facts only; no model or prediction is loaded.
  OwningOpRef<ModuleOp> featureModule = ModuleOp::create(mapperWrapper.getLoc());
  featureModule->getBody()->push_back(mapperWrapper->clone());
  PassManager featureFrontend(mapperWrapper.getContext());
  featureFrontend.addPass(::mlir::neura::createAssignAcceleratorPass());
  featureFrontend.addPass(::mlir::neura::createInsertDataMovPass());
  if (failed(featureFrontend.run(*featureModule))) {
    error = "C++ structural-startup frontend failed for the mapper body";
    return false;
  }
  if (failed(verify(*featureModule))) {
    error = "C++ structural-startup frontend produced invalid mapper IR";
    return false;
  }
  func::FuncOp featureWrapper =
      featureModule->lookupSymbol<func::FuncOp>(mapperWrapper.getSymName());
  if (!featureWrapper) {
    error = "C++ structural-startup frontend removed the mapper wrapper";
    return false;
  }
  std::string mapperText;
  llvm::raw_string_ostream mapperStream(mapperText);
  featureWrapper.print(mapperStream);
  mapperStream.flush();

  orbit::mapper_features::RouteExpandedGraph graph;
  if (!orbit::mapper_features::parseRouteExpandedDFG(
          mapperText, graph, error, /*requireExpanded=*/true)) {
    error = "C++ structural-startup frontend did not produce a certified "
            "route-expanded graph: " +
            error;
    return false;
  }
  json::Array nodeTypes;
  for (int type : graph.nodeTypes)
    nodeTypes.push_back(static_cast<int64_t>(type));
  json::Array edges;
  for (auto [source, target] : graph.edges)
    edges.push_back(json::Array{static_cast<int64_t>(source),
                                static_cast<int64_t>(target)});
  json::Object featureOutput;
  featureOutput["node_types"] = std::move(nodeTypes);
  featureOutput["edges"] = std::move(edges);
  return ::mlir::amoeba::neura::joint_scheduling::
      deriveStartupCyclesFromCppFeatureOutput(featureOutput, startup, error);
}

struct MapJointSchedulingTasksPass
    : public PassWrapper<MapJointSchedulingTasksPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapJointSchedulingTasksPass)

  MapJointSchedulingTasksPass() = default;
  MapJointSchedulingTasksPass(const MapJointSchedulingTasksPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "map-joint-scheduling-tasks";
  }
  StringRef getDescription() const override {
    return "Map each selected task on its own fixed oriented rectangle";
  }

  Option<std::string> scoreFile{*this, "scores",
                                llvm::cl::desc("Validated score JSONL path."),
                                llvm::cl::init("")};
  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Unique Taskflow function selected by the manifest."),
      llvm::cl::init("")};
  Option<std::string> candidateId{*this, "candidate-id",
                                  llvm::cl::desc("Selected candidate ID."),
                                  llvm::cl::init("")};
  Option<bool> allUnitBaseline{
      *this, "all-unit-baseline",
      llvm::cl::desc("Use source-certified all-1x1 mapper durations without "
                     "ML costs."),
      llvm::cl::init(false)};
  Option<std::string> parentProfileOutput{
      *this, "parent-profile-output",
      llvm::cl::desc("Export real mapper profiles for every legal area<=4 "
                     "parent-task rectangle without scheduling."),
      llvm::cl::init("")};
  Option<std::string> expectedTraceSha256{
      *this, "expected-trace-sha256",
      llvm::cl::desc("Expected selected schedule trace identity."),
      llvm::cl::init("")};
  Option<std::string> expectedScoreSha256{
      *this, "expected-score-sha256",
      llvm::cl::desc("SHA-256 of the exact validated score JSONL."),
      llvm::cl::init("")};

  Option<std::string> mappingCacheDir{*this, "mapping-cache-dir", llvm::cl::init("")};
  uint64_t mappingCacheHits = 0, mappingCacheMisses = 0;
  llvm::StringMap<double> startupCycles;
  llvm::StringMap<double> predictedIIs;
  llvm::StringMap<DictionaryAttr> exactPlacements;
  ArrayAttr exactDispatch;
  std::string materializedCandidateId;
  bool exactScores = false;
  bool productionScheduler = false;
  std::string baselineArchitectureText;
  llvm::StringMap<int64_t> baselineSourceWorkCounts;
  llvm::StringMap<std::string> baselineSourceDomainStatuses;
  std::string parentProfileCanonicalModuleText;
  std::string parentProfileCanonicalFunctionText;
  int64_t parentProfileGridRows = 0;
  int64_t parentProfileGridCols = 0;
  int64_t parentProfilePerCgraRows = 0;
  int64_t parentProfilePerCgraCols = 0;
  std::string productionDispatchPolicy;
  int64_t productionFixedPointIterations = 10;
  std::vector<ParentProfileRow> parentProfileRows;
  std::vector<ParentProfileAttempt> parentProfileAttempts;

  bool isParentProfileMode() const {
    return !parentProfileOutput.getValue().empty();
  }

  LogicalResult loadStartupCycles(ModuleOp module, StringRef function) {
    if (isParentProfileMode()) {
      if (allUnitBaseline.getValue() || !scoreFile.getValue().empty())
        return module.emitError(
            "parent profile export is mutually exclusive with scores and "
            "all-unit baseline");
      if (candidateId.getValue().empty())
        return module.emitError(
            "parent profile export requires a materialized candidate-id");
      if (!expectedTraceSha256.getValue().empty() ||
          !expectedScoreSha256.getValue().empty())
        return module.emitError(
            "parent profile export has no score or trace fingerprints");
      exactScores = false;
      productionScheduler = false;
      materializedCandidateId = candidateId.getValue();
      return success();
    }
    if (allUnitBaseline.getValue()) {
      if (!scoreFile.getValue().empty())
        return module.emitError("all-unit baseline must not consume an ML "
                                "score file");
      if (candidateId.getValue() != "candidate-0")
        return module.emitError("all-unit baseline requires shape-space "
                                "candidate-0");
      if (!expectedScoreSha256.getValue().empty())
        return module.emitError("all-unit baseline has no score JSONL to "
                                "bind with expected-score-sha256");
      exactScores = false;
      productionScheduler = true;
      productionDispatchPolicy = "fixed";
      productionFixedPointIterations = 10;
      materializedCandidateId = candidateId.getValue();
      return success();
    }
    if (scoreFile.getValue().empty() || candidateId.getValue().empty())
      return module.emitError("per-task mapper replay requires scores and "
                              "candidate-id");
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(scoreFile.getValue());
    if (!buffer)
      return module.emitError("cannot read score JSONL for mapper replay");
    bool found = false;
    bool sawHeader = false;
    bool sawFooter = false;
    uint64_t scoreCount = 0;
    std::optional<int64_t> declaredScoreCount;
    for (llvm::line_iterator lines(**buffer, /*SkipBlanks=*/true);
         !lines.is_at_end(); ++lines) {
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*lines);
      if (!parsed)
        return module.emitError("invalid score JSONL for mapper replay");
      llvm::json::Object *object = parsed->getAsObject();
      auto recordType =
          object ? object->getString("record_type") : std::nullopt;
      auto schema = object ? object->getString("schema") : std::nullopt;
      if (!recordType || !schema ||
          (*schema != ::mlir::amoeba::neura::joint_scheduling::kScoreSchema &&
           *schema != "amoeba-exact-joint-task-scores-v1"))
        return module.emitError("score JSONL contract is invalid");
      if (*recordType == "header") {
        if (sawHeader || sawFooter)
          return module.emitError("score JSONL header is misplaced");
        auto scoredFunction = object->getString("function");
        if (!scoredFunction || *scoredFunction != function)
          return module.emitError("score JSONL function identity mismatch");
        exactScores = *schema == "amoeba-exact-joint-task-scores-v1";
        productionScheduler = object->getString("schedule_space") ==
            std::optional<StringRef>("production-scheduler");
        if (productionScheduler) {
          auto dispatch = object->getString("dispatch_policy");
          if (!exactScores || !dispatch ||
              (*dispatch != "fixed" && *dispatch != "critical-path") ||
              object->getString("start_policy") != std::optional<StringRef>("production-spatial-temporal-scheduler") ||
              object->getString("temporal_search_scope") != std::optional<StringRef>("production-task-scheduler-only"))
            return module.emitError("production scheduler score contract is invalid");
          productionDispatchPolicy = dispatch->str();
          // Older score contracts used the production scheduler's legacy ten
          // iterations. New records carry the exact scoring policy into replay.
          auto iterations = object->getInteger("production_fixed_point_max_iterations");
          if (object->get("production_fixed_point_max_iterations") && !iterations)
            return module.emitError("production fixed-point iteration limit is not an integer");
          productionFixedPointIterations = iterations.value_or(10);
          if (productionFixedPointIterations <= 0 ||
              productionFixedPointIterations > std::numeric_limits<int>::max())
            return module.emitError("production fixed-point iteration limit is invalid");
        }
        materializedCandidateId = candidateId.getValue();
        sawHeader = true;
        continue;
      }
      if (*recordType == "footer") {
        if (!sawHeader || sawFooter)
          return module.emitError("score JSONL footer is misplaced");
        sawFooter = true;
        declaredScoreCount = object->getInteger("scored_count");
        if (!declaredScoreCount || *declaredScoreCount < 0)
          return module.emitError("score JSONL footer count is invalid");
        continue;
      }
      if (*recordType != "score")
        return module.emitError("score JSONL record type is unsupported");
      if (!sawHeader || sawFooter)
        return module.emitError("score record is misplaced");
      ++scoreCount;
      if (object->getString("candidate_id") != candidateId.getValue())
        continue;
      if (found)
        return module.emitError("duplicate selected score record");
      std::optional<bool> valid = object->getBoolean("valid");
      llvm::json::Array *costs = object->getArray("task_costs");
      if ((!exactScores && (!valid || !*valid)) || !costs)
        return module.emitError("selected score record is not valid");
      // The score pass emits the complete schedule trace directly.  Legacy
      // expected-*sha256 options remain accepted by the CLI for transition,
      // but no file or trace fingerprint is required or consumed.
      for (llvm::json::Value &value : *costs) {
        llvm::json::Object *cost = value.getAsObject();
        std::optional<StringRef> task =
            cost ? cost->getString("task") : std::nullopt;
        std::optional<double> startup =
            cost ? cost->getNumber("startup_cycles") : std::nullopt;
        if (!task || !startup || !std::isfinite(*startup) || *startup < 0 ||
            !startupCycles.try_emplace(*task, *startup).second)
          return module.emitError("selected task startup cost is invalid");
      }
      if (exactScores) {
        auto shapeId = object->getString("shape_candidate_id");
        auto schedule = object->getArray("task_schedule");
        auto dispatch = object->getArray("dispatch_order");
        if (!shapeId || !schedule || !dispatch || schedule->size() != costs->size() || dispatch->size() != costs->size())
          return module.emitError("exact mapper replay requires full shape, schedule and dispatch records");
        materializedCandidateId = shapeId->str();
        Builder builder(module.getContext());
        SmallVector<Attribute> order;
        llvm::StringSet<> names;
        for (const auto &v : *dispatch) {
          auto name = v.getAsString();
          if (!name || !names.insert(*name).second) return module.emitError("invalid exact dispatch order");
          order.push_back(builder.getStringAttr(*name));
        }
        exactDispatch = builder.getArrayAttr(order);
        for (const auto &v : *schedule) {
          const auto *entry = v.getAsObject();
          auto task = entry ? entry->getString("task") : std::nullopt;
          SmallVector<NamedAttribute> fields;
          if (!task || !names.count(*task)) return module.emitError("exact schedule task absent from dispatch");
          for (StringRef key : {"row", "col", "rows", "cols", "start_cycle", "end_cycle"}) {
            auto value = entry->getInteger(key);
            if (!value || *value < 0) return module.emitError("invalid exact schedule coordinate or time");
            fields.push_back(builder.getNamedAttr(key, builder.getI64IntegerAttr(*value)));
          }
          auto idle = entry->getInteger("idle_cycles").value_or(0);
          if (idle < 0) return module.emitError("negative selected idle interval");
          fields.push_back(builder.getNamedAttr("idle_cycles", builder.getI64IntegerAttr(idle)));
          if (!exactPlacements.try_emplace(*task, builder.getDictionaryAttr(fields)).second)
            return module.emitError("duplicate exact task schedule");
        }
        for (const auto &v : *costs) {
          const auto *entry = v.getAsObject();
          auto task = entry ? entry->getString("task") : std::nullopt;
          auto ii = entry ? entry->getNumber("predicted_ii") : std::nullopt;
          if (!task || !ii || !std::isfinite(*ii) || *ii <= 0 || !predictedIIs.try_emplace(*task, *ii).second)
            return module.emitError("invalid exact predicted II");
        }
      }
      found = true;
    }
    if (!found || !sawHeader || !sawFooter || !declaredScoreCount ||
        static_cast<uint64_t>(*declaredScoreCount) != scoreCount)
      return module.emitError("score JSONL is incomplete or selected record "
                              "is absent");
    return success();
  }

  LogicalResult mapTask(TaskflowTaskOp task,
                        const RectShape *profileShape = nullptr,
                        int64_t profileIndex = 0,
                        bool *expectedMapperFailure = nullptr) {
    if (expectedMapperFailure)
      *expectedMapperFailure = false;
    const bool profileOnly = profileShape != nullptr;
    if (!task->hasAttr("amoeba.joint_shape_orientation_fixed"))
      return task.emitError("task has no fixed joint-scheduling shape");
    auto selectedMapperRows =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_rows");
    auto selectedMapperCols =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_cols");
    const int64_t mapperRows =
        profileOnly ? profileShape->mapperRows
                    : (selectedMapperRows ? selectedMapperRows.getInt() : 0);
    const int64_t mapperCols =
        profileOnly ? profileShape->mapperCols
                    : (selectedMapperCols ? selectedMapperCols.getInt() : 0);
    if (mapperRows <= 0 || mapperCols <= 0 ||
        mapperRows > std::numeric_limits<int>::max() ||
        mapperCols > std::numeric_limits<int>::max())
      return task.emitError("selected mapper rectangle is missing or invalid");

    MLIRContext *context = task.getContext();
    Location location = task.getLoc();
    OpBuilder builder(context);
    ::mlir::neura::KernelOp kernel;
    task.walk([&](::mlir::neura::KernelOp candidate) { kernel = candidate; });
    ModuleOp temporary = ModuleOp::create(location);
    std::string wrapperError;
    func::FuncOp wrapper = buildCommonMapperReplayWrapper(
        task, temporary, "__joint_task_mapper_replay__", wrapperError);
    if (!wrapper)
      return task.emitError() << wrapperError;

    std::string preMapperWrapper;
    if (profileOnly) {
      preMapperWrapper = printCommonMapperReplayWrapper(wrapper);
    }

    if (allUnitBaseline.getValue() || profileOnly) {
      auto currentArchitecture = llvm::MemoryBuffer::getFile(
          ::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!currentArchitecture ||
          (*currentArchitecture)->getBuffer() != baselineArchitectureText) {
        if (allUnitBaseline.getValue())
          return task.emitError("mapper architecture changed during all-unit "
                                "baseline replay");
        return task.emitError("mapper architecture changed during source-owned "
                              "mapper replay");
      }
      if (!profileOnly ||
          startupCycles.find(task.getTaskName()) == startupCycles.end()) {
        double startup = 0.0;
        std::string startupError;
        if (!deriveStructuralStartupCycles(wrapper, startup, startupError))
          return task.emitError()
                 << "cannot derive structural startup cycles: "
                 << startupError;
        const bool inserted =
            startupCycles.try_emplace(task.getTaskName(), startup).second;
        if (!std::isfinite(startup) || startup <= 0.0)
          return task.emitError(
              allUnitBaseline.getValue()
                  ? "all-unit baseline structural startup is invalid or duplicated"
                  : "source-owned mapper structural startup is invalid");
        if (!inserted && !profileOnly)
          return task.emitError(
              "all-unit baseline structural startup is invalid or duplicated");
      }
    }

    std::string errorForCache;
    std::unique_ptr<MappingCacheLock> entryLock;
    std::string cacheEntry;
    bool cacheHit = false;
    if (!mappingCacheDir.empty()) {
      if (std::error_code ec = llvm::sys::fs::create_directories(mappingCacheDir))
        return task.emitError() << "cannot create mapper cache: " << ec.message();
      std::string key;
      llvm::raw_string_ostream keyStream(key);
      keyStream << "orbit-exact-task-mapper-cache-v1\nrows=" << mapperRows
                << " cols=" << mapperCols << "\n";
      wrapper.print(keyStream, OpPrintingFlags().useLocalScope());
      auto architecture = llvm::MemoryBuffer::getFile(::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!architecture) return task.emitError("cannot read mapper cache architecture");
      if ((allUnitBaseline.getValue() || profileOnly) &&
          (*architecture)->getBuffer() != baselineArchitectureText) {
        if (allUnitBaseline.getValue())
          return task.emitError("mapper architecture changed during all-unit "
                                "baseline replay");
        return task.emitError("mapper architecture changed during source-owned "
                              "mapper replay");
      }
      keyStream << "\narchitecture:\n" << (*architecture)->getBuffer();
      keyStream.flush();
      {
        MappingCacheLock registry(mappingCacheDir.getValue() + "/registry.lock");
        if (!registry.valid()) return task.emitError("cannot lock mapper cache registry");
        for (uint64_t index = 0;; ++index) {
          cacheEntry = mappingCacheDir.getValue() + "/task-" + std::to_string(index);
          auto input = llvm::MemoryBuffer::getFile(cacheEntry + "/input.txt");
          if (input && (*input)->getBuffer() == key) break;
          if (!input && !llvm::sys::fs::exists(cacheEntry)) {
            if (llvm::sys::fs::create_directory(cacheEntry)) return task.emitError("cannot allocate mapper cache entry");
            if (!::mlir::amoeba::neura::joint_scheduling::writeAtomically(cacheEntry + "/input.txt",
                [&](llvm::raw_ostream &stream) { stream << key; return true; }, errorForCache))
              return task.emitError() << errorForCache;
            break;
          }
        }
      }
      entryLock = std::make_unique<MappingCacheLock>(cacheEntry + "/mapping.lock");
      if (!entryLock->valid()) return task.emitError("cannot lock mapper cache task");
      if (llvm::sys::fs::exists(cacheEntry + "/mapped.mlir")) {
        auto cached = parseSourceFile<ModuleOp>(cacheEntry + "/mapped.mlir", context);
        auto cachedWrapper = cached ? cached->lookupSymbol<func::FuncOp>("__joint_task_mapper_replay__") : func::FuncOp();
        if (!cachedWrapper || failed(verify(*cached))) return task.emitError("mapper cache mapped module is invalid");
        wrapper->setAttrs(cachedWrapper->getAttrs());
        wrapper.getBody().takeBody(cachedWrapper.getBody());
        cacheHit = true;
        ++mappingCacheHits;
      }
    }
    if (!cacheHit) {
    PassManager manager(context);
    manager.addPass(::mlir::neura::createInsertDataMovPass());
    ::mlir::neura::MapToAcceleratorOptions options;
    options.x_tiles = static_cast<int>(mapperCols);
    options.y_tiles = static_cast<int>(mapperRows);
    manager.addPass(::mlir::neura::createMapToAcceleratorPass(options));
    if (failed(manager.run(temporary))) {
      if (profileOnly && expectedMapperFailure) {
        *expectedMapperFailure = true;
        return failure();
      }
      return task.emitError("real per-task mapper replay failed");
    }

      ++mappingCacheMisses;
    }
    if (!profileOnly)
      task->setAttr("amoeba.mapper_cache_hit", builder.getBoolAttr(cacheHit));

    auto mapping_info = wrapper->getAttrOfType<DictionaryAttr>(
        ::mlir::neura::attr::kMappingInfo);
    auto compiled_ii =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kCompiledII)
            : IntegerAttr();
    auto mapped_rows =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kYTiles)
            : IntegerAttr();
    auto mapped_cols =
        mapping_info
            ? mapping_info.getAs<IntegerAttr>(::mlir::neura::attr::kXTiles)
            : IntegerAttr();
    if (!compiled_ii || compiled_ii.getInt() <= 0 || !mapped_rows ||
        !mapped_cols || mapped_rows.getInt() != mapperRows ||
        mapped_cols.getInt() != mapperCols)
      return task.emitError(
          "mapper result does not match the selected oriented rectangle");

    if (!cacheHit && !cacheEntry.empty() &&
        !::mlir::amoeba::neura::joint_scheduling::writeAtomically(
            cacheEntry + "/mapped.mlir",
            [&](llvm::raw_ostream &stream) {
              temporary.print(stream);
              stream << "\n";
              return true;
            }, errorForCache))
      return task.emitError() << errorForCache;

    if (!profileOnly) {
      // Preserve the actual mapped operations, not merely the mapper metadata.
      for (Block &block : wrapper.getBody()) {
        if (auto return_op = dyn_cast<func::ReturnOp>(block.getTerminator())) {
          builder.setInsertionPoint(return_op);
          builder.create<::mlir::neura::YieldOp>(location, ValueRange{},
                                                 return_op.getOperands());
          return_op.erase();
        }
      }
      kernel.getBody().takeBody(wrapper.getBody());
    }

    if (!profileOnly) {
      task->setAttr("compiled_ii", compiled_ii);
      task->setAttr("amoeba.mapper_mapping_info", mapping_info);
      task->setAttr("amoeba.mapper_replay_verified", builder.getUnitAttr());
    }
    auto tripCount =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_trip_count");
    auto startup = startupCycles.find(task.getTaskName());
    if (!tripCount || tripCount.getInt() <= 0 || startup == startupCycles.end())
      return task.emitError("mapper replay has no bound trip/startup cost");
    const long double duration =
        static_cast<long double>(startup->second) +
        static_cast<long double>(compiled_ii.getInt()) *
            static_cast<long double>(tripCount.getInt() - 1);
    if (!std::isfinite(duration) || duration < 1 ||
        duration > std::numeric_limits<int64_t>::max())
      return task.emitError("real mapped task duration exceeds int64");
    const int64_t roundedDuration = static_cast<int64_t>(std::ceil(duration));
    int64_t steps = 0;
    int64_t materializedOperationCount = 0;
    if (profileOnly) {
      std::string profileError;
      if (!collectActualMapperProfileFacts(wrapper.getBody(), steps,
                                           materializedOperationCount,
                                           profileError))
        return task.emitError() << "cannot extract actual mapper profile: "
                                << profileError;
    }
    SmallVector<NamedAttribute> profile;
    profile.push_back(builder.getNamedAttr(
        "duration", builder.getI64IntegerAttr(roundedDuration)));
    auto sourceWork = baselineSourceWorkCounts.find(task.getTaskName());
    if ((allUnitBaseline.getValue() || profileOnly) &&
        (sourceWork == baselineSourceWorkCounts.end() ||
         sourceWork->second < tripCount.getInt())) {
      if (allUnitBaseline.getValue())
        return task.emitError("all-unit baseline has no validated source work "
                              "count");
      return task.emitError("source-owned mapper profile has no validated "
                            "source work count");
    }
    if (allUnitBaseline.getValue()) {
      StringRef durationProvenance =
          "source-owned-all-unit-mapped-duration";
      profile.push_back(builder.getNamedAttr(
          "duration_provenance", builder.getStringAttr(durationProvenance)));
      profile.push_back(builder.getNamedAttr(
          "structural_startup_cycles", builder.getF64FloatAttr(startup->second)));
      profile.push_back(builder.getNamedAttr(
          "actual_mapper_ii", builder.getI64IntegerAttr(compiled_ii.getInt())));
      profile.push_back(builder.getNamedAttr(
          "ml_predicted_ii", builder.getStringAttr("unknown")));
      profile.push_back(builder.getNamedAttr(
          "current_taskflow_firing_count",
          builder.getI64IntegerAttr(tripCount.getInt())));
      profile.push_back(builder.getNamedAttr(
          "source_iteration_work_count",
          builder.getI64IntegerAttr(sourceWork->second)));
      task->setAttr("amoeba.mapper_duration_provenance",
                    builder.getStringAttr(durationProvenance));
      task->setAttr("amoeba.mapper_structural_startup_cycles",
                    builder.getF64FloatAttr(startup->second));
      task->setAttr("amoeba.mapper_actual_ii",
                    builder.getI64IntegerAttr(compiled_ii.getInt()));
      task->setAttr("amoeba.mapper_ml_prediction_status",
                    builder.getStringAttr("unknown"));
    }
    if (profileOnly) {
      const long double stepsDuration =
          static_cast<long double>(steps) +
          static_cast<long double>(compiled_ii.getInt()) *
              static_cast<long double>(tripCount.getInt() - 1);
      if (!std::isfinite(stepsDuration) || stepsDuration < 1 ||
          stepsDuration > std::numeric_limits<int64_t>::max())
        return task.emitError("steps-based profile duration exceeds int64");
      ParentProfileRow row;
      row.task = task.getTaskName().str();
      row.shape = profileShape->toCgraShapeAttrValue();
      row.preMapperWrapper = std::move(preMapperWrapper);
      row.candidateIndex = profileIndex;
      row.cgraCount = profileShape->cgraCount();
      row.mapperRows = mapperRows;
      row.mapperCols = mapperCols;
      row.compiledII = compiled_ii.getInt();
      row.steps = steps;
      row.materializedOperationCount = materializedOperationCount;
      row.sampleTripCount = tripCount.getInt();
      row.estimatedLatency = roundedDuration;
      row.stepsFormulaLatency = static_cast<int64_t>(std::ceil(stepsDuration));
      row.sourceIterationWorkCount = sourceWork->second;
      auto domainStatus =
          baselineSourceDomainStatuses.find(task.getTaskName());
      if (domainStatus == baselineSourceDomainStatuses.end())
        return task.emitError("parent profile export has no source-domain "
                              "provenance status");
      row.sourceIterationDomainStatus = domainStatus->second;
      row.structuralStartupCycles = startup->second;
      parentProfileRows.push_back(std::move(row));
      parentProfileAttempts.push_back(
          {task.getTaskName().str(), profileShape->toCgraShapeAttrValue(),
           profileIndex, profileShape->cgraCount(), true, true, ""});
    } else {
      task->setAttr("profile_info", builder.getDictionaryAttr(profile));
    }
    return success();
  }

  LogicalResult writeParentProfile(ModuleOp module, func::FuncOp function,
                                   ArrayRef<TaskflowTaskOp> tasks,
                                   ArrayRef<RectShape> shapes) {
    if (tasks.empty() || shapes.empty() ||
        tasks.size() > static_cast<size_t>(
                           std::numeric_limits<int64_t>::max()) /
                           shapes.size())
      return module.emitError("parent profile task/shape inventory is invalid");
    const int64_t expectedAttempts =
        static_cast<int64_t>(tasks.size() * shapes.size());
    if (parentProfileAttempts.size() !=
        static_cast<size_t>(expectedAttempts))
      return module.emitError("parent profile attempt inventory is incomplete");
    llvm::StringSet<> attemptKeys;
    llvm::StringSet<> profileKeys;
    for (const ParentProfileAttempt &attempt : parentProfileAttempts) {
      if (attempt.candidateIndex <= 0 ||
          attempt.candidateIndex > static_cast<int64_t>(shapes.size()))
        return module.emitError("parent profile attempt inventory is malformed "
                                "or duplicated");
      const RectShape &expectedShape =
          shapes[static_cast<size_t>(attempt.candidateIndex - 1)];
      if (attempt.shape != expectedShape.toCgraShapeAttrValue() ||
          attempt.cgraCount != expectedShape.cgraCount() ||
          attempt.profileCreated != attempt.mapperSucceeded ||
          !attemptKeys.insert(attempt.task + "\n" + attempt.shape).second)
        return module.emitError("parent profile attempt inventory is malformed "
                                "or duplicated");
    }
    for (TaskflowTaskOp task : tasks) {
      for (const RectShape &shape : shapes) {
        if (!attemptKeys.count(task.getTaskName().str() + "\n" +
                               shape.toCgraShapeAttrValue()))
          return module.emitError("parent profile attempt inventory omits a "
                                  "task/shape pair");
      }
    }
    for (const ParentProfileRow &row : parentProfileRows) {
      const std::string key = row.task + "\n" + row.shape;
      if (!attemptKeys.count(key) || !profileKeys.insert(key).second)
        return module.emitError("parent profile rows do not match unique "
                                "successful attempts");
    }
    for (const ParentProfileAttempt &attempt : parentProfileAttempts) {
      const std::string key = attempt.task + "\n" + attempt.shape;
      if (attempt.profileCreated != static_cast<bool>(profileKeys.count(key)))
        return module.emitError("parent profile success attempt and row "
                                "inventories disagree");
    }

    constexpr StringLiteral profileProvenance =
        "source-owned-common-parent-profile-only-mapped-duration-v1";
    json::Array taskRecords;
    taskRecords.reserve(tasks.size());
    for (TaskflowTaskOp task : tasks) {
      json::Array profiles;
      for (const ParentProfileRow &row : parentProfileRows) {
        if (row.task != task.getTaskName())
          continue;
        json::Object profile;
        profile["composed_cgra_count"] = row.cgraCount;
        profile["composed_cgra_shape"] = row.shape;
        profile["compiled_ii"] = row.compiledII;
        profile["steps"] = row.steps;
        profile["materialized_operation_count"] =
            row.materializedOperationCount;
        profile["sample_trip_count"] = row.sampleTripCount;
        profile["estimated_latency"] = row.estimatedLatency;
        profile["mapper_succeeded"] = true;
        profile["mapper_tile_rows"] = row.mapperRows;
        profile["mapper_tile_cols"] = row.mapperCols;
        profile["structural_startup_cycles"] =
            row.structuralStartupCycles;
        profile["duration_provenance"] = profileProvenance.str();
        profile["duration_formula"] =
            "ceil(structural_startup_cycles + compiled_ii * "
            "(sample_trip_count - 1))";
        profile["steps_formula_estimated_latency"] = row.stepsFormulaLatency;
        profile["startup_steps_match"] =
            row.estimatedLatency == row.stepsFormulaLatency;
        profile["source_iteration_work_count"] =
            row.sourceIterationWorkCount;
        profile["source_iteration_domain_status"] =
            row.sourceIterationDomainStatus;
        profile["source_iteration_domain_certified"] = true;
        profile["source_iteration_domain_complete"] = true;
        profile["pre_mapper_wrapper_byte_count"] =
            static_cast<int64_t>(row.preMapperWrapper.size());
        profile["pre_mapper_wrapper_bytes"] = row.preMapperWrapper;
        profile["candidate_index_in_task"] = row.candidateIndex;
        profiles.push_back(std::move(profile));
      }
      auto sourceWork = baselineSourceWorkCounts.find(task.getTaskName());
      auto sourceDomainStatus =
          baselineSourceDomainStatuses.find(task.getTaskName());
      auto tripCount = task->getAttrOfType<IntegerAttr>(
          "amoeba.selected_trip_count");
      if (sourceWork == baselineSourceWorkCounts.end() ||
          sourceDomainStatus == baselineSourceDomainStatuses.end() ||
          !tripCount || tripCount.getInt() <= 0)
        return module.emitError(
            "parent profile task has incomplete source iteration evidence");
      json::Object taskRecord;
      taskRecord["task"] = task.getTaskName().str();
      taskRecord["sample_trip_count"] = tripCount.getInt();
      taskRecord["source_iteration_work_count"] = sourceWork->second;
      taskRecord["source_iteration_domain_status"] = sourceDomainStatus->second;
      taskRecord["source_iteration_domain_certified"] = true;
      taskRecord["source_iteration_domain_complete"] = true;
      taskRecord["profiles"] = std::move(profiles);
      taskRecords.push_back(std::move(taskRecord));
    }

    json::Array attempts;
    attempts.reserve(parentProfileAttempts.size());
    for (const ParentProfileAttempt &attempt : parentProfileAttempts) {
      json::Object record;
      record["task"] = attempt.task;
      record["candidate_index_in_task"] = attempt.candidateIndex;
      record["shape"] = attempt.shape;
      record["composed_cgra_count"] = attempt.cgraCount;
      record["profile_created"] = attempt.profileCreated;
      record["mapper_succeeded"] = attempt.mapperSucceeded;
      if (!attempt.failureReason.empty())
        record["failure_reason"] = attempt.failureReason;
      attempts.push_back(std::move(record));
    }

    json::Array shapeDomain;
    shapeDomain.reserve(shapes.size());
    for (const RectShape &shape : shapes)
      shapeDomain.push_back(shape.toCgraShapeAttrValue());

    json::Object root;
    root["format"] = "amoeba-task-profile-v1";
    root["function"] = function.getSymName().str();
    root["task_count"] = static_cast<int64_t>(tasks.size());
    root["expected_candidate_count"] = expectedAttempts;
    root["completed_candidate_count"] = expectedAttempts;
    root["candidate_id"] = candidateId.getValue();
    root["candidate_scope"] =
        ::mlir::amoeba::neura::joint_scheduling::kSearchScope.str();
    root["profile_provenance"] = profileProvenance.str();
    root["source_iteration_domain_coverage_verified"] = true;
    root["startup_provenance"] =
        "source-owned-cpp-route-expanded-structural-critical-path-v1";
    root["mapper_ii_provenance"] = "native-neura-map-to-accelerator-v1";
    root["duration_formula"] =
        "ceil(structural_startup_cycles + compiled_ii * "
        "(sample_trip_count - 1))";
    root["shape_domain"] = std::move(shapeDomain);
    root["architecture_spec_path"] =
        ::mlir::amoeba::getNeuraArchitectureSpecFile();
    root["architecture_spec_text"] = baselineArchitectureText;
    root["hardware_coordinates"] = json::Object{
        {"multi_cgra_grid_rows", parentProfileGridRows},
        {"multi_cgra_grid_cols", parentProfileGridCols},
        {"per_cgra_tile_rows", parentProfilePerCgraRows},
        {"per_cgra_tile_cols", parentProfilePerCgraCols},
        {"total_mapper_rows",
         parentProfileGridRows * parentProfilePerCgraRows},
        {"total_mapper_cols",
         parentProfileGridCols * parentProfilePerCgraCols}};
    root["canonical_witness_format"] =
        "mlir-generic-print-use-local-scope-v1";
    root["canonical_module_witness_byte_count"] =
        static_cast<int64_t>(parentProfileCanonicalModuleText.size());
    root["canonical_module_witness_bytes"] =
        parentProfileCanonicalModuleText;
    root["canonical_function_witness_byte_count"] =
        static_cast<int64_t>(parentProfileCanonicalFunctionText.size());
    root["canonical_function_witness_bytes"] =
        parentProfileCanonicalFunctionText;
    root["whole_program_scheduler_invoked"] = false;
    root["tasks"] = std::move(taskRecords);
    root["candidate_attempts"] = std::move(attempts);

    std::string writeError;
    if (!::mlir::amoeba::neura::joint_scheduling::writeAtomically(
            parentProfileOutput.getValue(),
            [&](llvm::raw_ostream &stream) {
              stream << json::Value(std::move(root)) << "\n";
              return true;
            },
            writeError))
      return module.emitError() << "cannot publish parent profile: "
                                << writeError;
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        ::mlir::amoeba::neura::joint_scheduling::selectTaskFunction(
            module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    if (isParentProfileMode()) {
      llvm::raw_string_ostream moduleStream(parentProfileCanonicalModuleText);
      module.print(moduleStream,
                   OpPrintingFlags().printGenericOpForm().useLocalScope());
      moduleStream.flush();
      llvm::raw_string_ostream functionStream(
          parentProfileCanonicalFunctionText);
      (*selectedFunction).print(
          functionStream,
          OpPrintingFlags().printGenericOpForm().useLocalScope());
      functionStream.flush();
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function == *selectedFunction)
        continue;
      bool hasTasks = false;
      function.walk([&](TaskflowTaskOp) { hasTasks = true; });
      if (hasTasks) {
        function.emitError("mapper replay rejects additional task-bearing "
                           "functions outside its selected function");
        return signalPassFailure();
      }
    }
    if (failed(loadStartupCycles(module, selectedFunction->getSymName())))
      return signalPassFailure();
    const bool parentProfileMode = isParentProfileMode();
    if (allUnitBaseline.getValue()) {
      auto candidate = (*selectedFunction)->getAttrOfType<StringAttr>(
          "joint_scheduling_candidate_id");
      auto scope = (*selectedFunction)->getAttrOfType<StringAttr>(
          "joint_scheduling_candidate_scope");
      if (!candidate || candidate.getValue() != "candidate-0" ||
          !scope || scope.getValue() !=
                        ::mlir::amoeba::neura::joint_scheduling::kSearchScope) {
        selectedFunction->emitError(
            "all-unit baseline requires shape-only static candidate-0");
        return signalPassFailure();
      }
      std::string rewriteError;
      if (hasAllUnitRewriteEvidence(*selectedFunction, rewriteError)) {
        selectedFunction->emitError() << rewriteError;
        return signalPassFailure();
      }
    }
    llvm::SmallVector<RectShape> parentProfileShapes;
    if (parentProfileMode) {
      auto candidate = (*selectedFunction)->getAttrOfType<StringAttr>(
          "joint_scheduling_candidate_id");
      auto scope = (*selectedFunction)->getAttrOfType<StringAttr>(
          "joint_scheduling_candidate_scope");
      if (!candidate || candidate.getValue() != materializedCandidateId ||
          !scope || scope.getValue() !=
                        ::mlir::amoeba::neura::joint_scheduling::kSearchScope) {
        selectedFunction->emitError(
            "parent profile export requires a source-bound static-shape "
            "candidate with the requested candidate-id");
        return signalPassFailure();
      }
      std::string rewriteError;
      if (hasParentProfileRewriteEvidence(*selectedFunction, rewriteError)) {
        selectedFunction->emitError() << rewriteError;
        return signalPassFailure();
      }
      const auto &architecture = ::mlir::neura::getArchitecture();
      parentProfileGridRows = architecture.getMultiCgraRows();
      parentProfileGridCols = architecture.getMultiCgraColumns();
      parentProfilePerCgraRows = architecture.getPerCgraRows();
      parentProfilePerCgraCols = architecture.getPerCgraColumns();
      parentProfileShapes =
          ::mlir::amoeba::neura::joint_scheduling::enumerateStaticRectShapes(
              architecture.getMultiCgraRows(),
              architecture.getMultiCgraColumns(),
              architecture.getPerCgraRows(),
              architecture.getPerCgraColumns(), 4);
      if (parentProfileShapes.empty()) {
        selectedFunction->emitError(
            "parent profile export has no legal area<=4 mapper shapes");
        return signalPassFailure();
      }
      constexpr std::array<const char *, 8> canonicalShapeOrder = {
          "1x1", "1x2", "2x1", "1x3", "3x1", "2x2", "1x4", "4x1"};
      llvm::SmallVector<RectShape> orderedShapes;
      for (const char *rawSpelling : canonicalShapeOrder) {
        StringRef spelling(rawSpelling);
        auto shape = llvm::find_if(parentProfileShapes, [&](const RectShape &v) {
          return spelling == v.toCgraShapeAttrValue();
        });
        if (shape != parentProfileShapes.end())
          orderedShapes.push_back(*shape);
      }
      if (orderedShapes.size() != parentProfileShapes.size()) {
        selectedFunction->emitError(
            "parent profile export found a legal rectangle outside the "
            "canonical area<=4 profile domain");
        return signalPassFailure();
      }
      parentProfileShapes = std::move(orderedShapes);
    }
    SmallVector<TaskflowTaskOp> tasks;
    selectedFunction->walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
    if (tasks.empty()) {
      module.emitError("per-task mapper replay requires taskflow tasks");
      return signalPassFailure();
    }
    for (TaskflowTaskOp task : tasks) {
      auto function = task->getParentOfType<func::FuncOp>();
      auto selected = function ? function->getAttrOfType<StringAttr>(
                                     "joint_scheduling_candidate_id")
                               : StringAttr();
      if (!selected || selected.getValue() != materializedCandidateId) {
        task.emitError("mapper replay candidate identity does not match IR");
        return signalPassFailure();
      }
    }
    if (parentProfileMode) {
      for (TaskflowTaskOp task : tasks) {
        auto selectedCount = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_cgra_count");
        auto mapperRows = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_mapper_tile_rows");
        auto mapperCols = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_mapper_tile_cols");
        auto shape =
            task->getAttrOfType<StringAttr>("amoeba.selected_cgra_shape");
        auto cgraCount = task->getAttrOfType<IntegerAttr>("cgra_count");
        auto cgraShape = task->getAttrOfType<StringAttr>("cgra_shape");
        const RectShape *canonicalShape = nullptr;
        if (shape) {
          for (const RectShape &candidateShape : parentProfileShapes)
            if (shape.getValue() == candidateShape.toCgraShapeAttrValue()) {
              canonicalShape = &candidateShape;
              break;
            }
        }
        if (!canonicalShape || !selectedCount ||
            selectedCount.getInt() != canonicalShape->cgraCount() ||
            !mapperRows || mapperRows.getInt() != canonicalShape->mapperRows ||
            !mapperCols || mapperCols.getInt() != canonicalShape->mapperCols ||
            !cgraCount || cgraCount.getInt() != canonicalShape->cgraCount() ||
            !cgraShape || cgraShape.getValue() != shape.getValue() ||
            !task->getAttrOfType<UnitAttr>(
                "amoeba.joint_shape_orientation_fixed") ||
            task->hasAttr("active_replicas") || task->hasAttr("replicas") ||
            task->hasAttr("replica_shapes") ||
            task->hasAttr("original_replica_count")) {
          task.emitError("parent profile export requires a well-formed fixed "
                         "orientation legal area<=4 shape");
          return signalPassFailure();
        }
      }
    }
    if (allUnitBaseline.getValue()) {
      const auto &physicalArchitecture = ::mlir::neura::getArchitecture();
      for (TaskflowTaskOp task : tasks) {
        auto selectedCount = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_cgra_count");
        auto mapperRows = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_mapper_tile_rows");
        auto mapperCols = task->getAttrOfType<IntegerAttr>(
            "amoeba.selected_mapper_tile_cols");
        auto shape = task->getAttrOfType<StringAttr>(
            "amoeba.selected_cgra_shape");
        auto cgraCount = task->getAttrOfType<IntegerAttr>("cgra_count");
        auto cgraShape = task->getAttrOfType<StringAttr>("cgra_shape");
        if (!selectedCount || selectedCount.getInt() != 1 || !mapperRows ||
            mapperRows.getInt() != physicalArchitecture.getPerCgraRows() ||
            !mapperCols ||
            mapperCols.getInt() != physicalArchitecture.getPerCgraColumns() ||
            !shape || shape.getValue() != "1x1" || !cgraCount ||
            cgraCount.getInt() != 1 || !cgraShape ||
            cgraShape.getValue() != "1x1" ||
            !task->getAttrOfType<UnitAttr>(
                "amoeba.joint_shape_orientation_fixed")) {
          task.emitError("all-unit baseline requires the identity-only 1x1 "
                         "fixed-orientation shape for every task");
          return signalPassFailure();
        }
      }
    }
    if (allUnitBaseline.getValue() || parentProfileMode) {
      const StringRef modeLabel = parentProfileMode ? "parent profile export"
                                                    : "all-unit baseline";
      std::string metadataError;
      FailureOr<llvm::SmallVector<
          ::mlir::amoeba::neura::joint_scheduling::TaskMetadata>> metadata =
          ::mlir::amoeba::neura::joint_scheduling::collectAnalyticalTaskMetadata(
              *selectedFunction, metadataError);
      if (failed(metadata)) {
        if (parentProfileMode)
          module.emitError() << modeLabel
                            << " requires a valid complete source-domain "
                               "certificate: "
                            << metadataError;
        else
          module.emitError() << "all-unit baseline requires a valid complete "
                                "source-domain certificate: "
                             << metadataError;
        return signalPassFailure();
      }
      for (const auto &entry : *metadata) {
        auto selectedTrip = entry.op->getAttrOfType<IntegerAttr>(
            "amoeba.selected_trip_count");
        if (!entry.sourceIterationDomainCertified ||
            !entry.sourceIterationDomainComplete || !entry.tripCountKnown ||
            entry.tripCount <= 0 ||
            entry.sourceIterationWorkCount < entry.tripCount) {
          if (parentProfileMode)
            module.emitError() << modeLabel
                              << " requires a complete, source-owned "
                                 "iteration-domain certificate and proven "
                                 "current count for task "
                              << entry.name;
          else
            module.emitError() << "all-unit baseline requires a complete, "
                                  "source-owned iteration-domain certificate "
                                  "and proven current count for task "
                               << entry.name;
          return signalPassFailure();
        }
        if (!selectedTrip || selectedTrip.getInt() != entry.tripCount) {
          if (parentProfileMode)
            module.emitError() << modeLabel
                              << " current Taskflow firing count does not "
                                 "match its source-certified count for task "
                              << entry.name;
          else
            module.emitError() << "all-unit baseline current Taskflow firing "
                                  "count does not match its selected candidate "
                                  "count for task "
                               << entry.name;
          return signalPassFailure();
        }
        baselineSourceWorkCounts[entry.name] =
            entry.sourceIterationWorkCount;
        if (parentProfileMode)
          baselineSourceDomainStatuses[entry.name] =
              entry.sourceIterationDomainStatus;
      }
      if (metadata->size() != tasks.size()) {
        if (parentProfileMode)
          module.emitError() << modeLabel
                             << " source count coverage differs from the "
                                "selected task set";
        else
          module.emitError("all-unit baseline source count coverage differs "
                           "from the selected task set");
        return signalPassFailure();
      }
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> architecture =
          llvm::MemoryBuffer::getFile(
              ::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!architecture || (*architecture)->getBuffer().empty()) {
        if (parentProfileMode)
          module.emitError() << modeLabel
                             << " cannot read the current mapper architecture";
        else
          module.emitError("all-unit baseline cannot read the current mapper "
                           "architecture");
        return signalPassFailure();
      }
      baselineArchitectureText = (*architecture)->getBuffer().str();
    }
    if (!allUnitBaseline.getValue() && !parentProfileMode &&
        startupCycles.size() != tasks.size()) {
      module.emitError("selected score task coverage does not match replay");
      return signalPassFailure();
    }
    if (parentProfileMode) {
      for (TaskflowTaskOp task : tasks) {
        for (auto [shapeIndex, shape] : llvm::enumerate(parentProfileShapes)) {
          bool expectedMapperFailure = false;
          if (failed(mapTask(task, &shape,
                             static_cast<int64_t>(shapeIndex + 1),
                             &expectedMapperFailure))) {
            if (!expectedMapperFailure)
              return signalPassFailure();
            parentProfileAttempts.push_back(
                {task.getTaskName().str(), shape.toCgraShapeAttrValue(),
                 static_cast<int64_t>(shapeIndex + 1), shape.cgraCount(),
                 false, false,
                 "native-neura-map-to-accelerator-failed"});
          }
        }
      }
    } else {
      for (TaskflowTaskOp task : tasks)
        if (failed(mapTask(task)))
          return signalPassFailure();
    }
    if (startupCycles.size() != tasks.size()) {
      module.emitError("structural startup coverage does not match mapper replay");
      return signalPassFailure();
    }
    if (exactScores) {
      bool predictionEqual = true;
      for (TaskflowTaskOp task : tasks) {
        auto placement = exactPlacements.find(task.getTaskName());
        auto expectedII = predictedIIs.find(task.getTaskName());
        if (placement == exactPlacements.end() || expectedII == predictedIIs.end()) {
          module.emitError("exact score does not cover every materialized task");
          return signalPassFailure();
        }
        auto attr = placement->second;
        auto rows = task->getAttrOfType<IntegerAttr>("amoeba.selected_cgra_count");
        auto mapperRows = task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_rows");
        auto shapeRows = attr.getAs<IntegerAttr>("rows").getInt();
        auto shapeCols = attr.getAs<IntegerAttr>("cols").getInt();
        if (!rows || rows.getInt() != shapeRows * shapeCols || !mapperRows || shapeRows < 1 || shapeCols < 1) {
          module.emitError("exact score rectangle does not match materialized shape");
          return signalPassFailure();
        }
        if (productionScheduler)
          task->removeAttr("amoeba.exact_schedule");
        else
          task->setAttr("amoeba.exact_schedule", attr);
        predictionEqual &= task->getAttrOfType<IntegerAttr>("compiled_ii").getInt() == expectedII->second;
      }
      if (productionScheduler) {
        (*selectedFunction)->removeAttr("joint_scheduling_exact_dispatch_order");
        (*selectedFunction)->setAttr("joint_scheduling_scheduler_backend",
            StringAttr::get(module.getContext(), "orchestrate-tasks-on-accelerators"));
        (*selectedFunction)->setAttr("joint_scheduling_production_dispatch_policy",
            StringAttr::get(module.getContext(), productionDispatchPolicy));
        (*selectedFunction)->setAttr("joint_scheduling_fixed_point_max_iterations",
            IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                             productionFixedPointIterations));
      } else
        (*selectedFunction)->setAttr("joint_scheduling_exact_dispatch_order", exactDispatch);
      (*selectedFunction)->setAttr("joint_scheduling_candidate_id", StringAttr::get(module.getContext(), candidateId));
      (*selectedFunction)->setAttr("joint_scheduling_prediction_mapper_equal", BoolAttr::get(module.getContext(), predictionEqual));
    }
    if (allUnitBaseline.getValue()) {
      (*selectedFunction)->removeAttr("joint_scheduling_exact_dispatch_order");
      (*selectedFunction)->setAttr(
          "joint_scheduling_scheduler_backend",
          StringAttr::get(module.getContext(),
                          "orchestrate-tasks-on-accelerators"));
      (*selectedFunction)->setAttr(
          "joint_scheduling_production_dispatch_policy",
          StringAttr::get(module.getContext(), productionDispatchPolicy));
      (*selectedFunction)->setAttr(
          "joint_scheduling_fixed_point_max_iterations",
          IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                           productionFixedPointIterations));
      (*selectedFunction)->setAttr(
          "joint_scheduling_mapper_duration_provenance",
          StringAttr::get(module.getContext(),
                          "source-owned-all-unit-mapped-duration"));
      (*selectedFunction)->setAttr(
          "joint_scheduling_ml_prediction_status",
          StringAttr::get(module.getContext(), "unknown"));
      (*selectedFunction)->setAttr(
          "joint_scheduling_candidate_id",
          StringAttr::get(module.getContext(), "candidate-0"));
    }
    if (parentProfileMode) {
      if (failed(writeParentProfile(module, *selectedFunction, tasks,
                                    parentProfileShapes)))
        return signalPassFailure();
      (*selectedFunction)->setAttr(
          "joint_scheduling_mapper_profile_export_only",
          UnitAttr::get(module.getContext()));
      (*selectedFunction)->setAttr(
          "joint_scheduling_mapper_profile_export_provenance",
          StringAttr::get(
              module.getContext(),
              "source-owned-common-parent-profile-only-mapped-duration-v1"));
    }
    (*selectedFunction)->setAttr("joint_scheduling_mapper_cache_hits", IntegerAttr::get(IntegerType::get(module.getContext(), 64), mappingCacheHits));
    (*selectedFunction)->setAttr("joint_scheduling_mapper_cache_misses", IntegerAttr::get(IntegerType::get(module.getContext(), 64), mappingCacheMisses));
    llvm::errs() << "[JointScheduling] mapper cache hits=" << mappingCacheHits << " misses=" << mappingCacheMisses << "\n";
    (*selectedFunction)
        ->setAttr("joint_scheduling_mapper_replay_completed",
                  UnitAttr::get(module.getContext()));
  }
};

} // namespace

namespace mlir::amoeba::neura {

std::unique_ptr<Pass> createMapJointSchedulingTasksPass() {
  return std::make_unique<MapJointSchedulingTasksPass>();
}

} // namespace mlir::amoeba::neura

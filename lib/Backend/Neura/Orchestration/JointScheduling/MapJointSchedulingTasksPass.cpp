//===- MapJointSchedulingTasksPass.cpp ----------------------------------===//
//
// Replays Neura mapping independently for each selected task rectangle.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "AnalyticalTaskCostCatalog.h"
#include "MapperCostAnalysis.h"
#include "SpatialTaskCandidateSpace.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "MapperCounterBounds.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"

#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/Architecture/Architecture.h"
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
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::taskflow;
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
  std::string productionDispatchPolicy;
  int64_t productionFixedPointIterations = 10;

  LogicalResult loadStartupCycles(ModuleOp module, StringRef function) {
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

  LogicalResult mapTask(TaskflowTaskOp task) {
    if (!task->hasAttr("amoeba.joint_shape_orientation_fixed"))
      return task.emitError("task has no fixed joint-scheduling shape");
    auto mapper_rows =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_rows");
    auto mapper_cols =
        task->getAttrOfType<IntegerAttr>("amoeba.selected_mapper_tile_cols");
    if (!mapper_rows || !mapper_cols || mapper_rows.getInt() <= 0 ||
        mapper_cols.getInt() <= 0 ||
        mapper_rows.getInt() > std::numeric_limits<int>::max() ||
        mapper_cols.getInt() > std::numeric_limits<int>::max())
      return task.emitError("selected mapper rectangle is missing or invalid");

    SmallVector<::mlir::neura::KernelOp> kernels;
    task.walk(
        [&](::mlir::neura::KernelOp kernel) { kernels.push_back(kernel); });
    if (kernels.size() != 1)
      return task.emitError("per-task mapper replay requires exactly one "
                            "neura.kernel");
    ::mlir::neura::KernelOp kernel = kernels.front();
    if (kernel.getBody().empty())
      return task.emitError("cannot map an empty neura.kernel");

    // Dataflow kernels can publish their results through neura.return_value
    // while ending the body with a void neura.yield. The temporary function
    // follows the yield contract; the original kernel keeps its result types.
    SmallVector<Type> wrapperResultTypes;
    bool sawYield = false;
    for (Block &block : kernel.getBody()) {
      auto yield = dyn_cast<::mlir::neura::YieldOp>(block.getTerminator());
      if (!yield)
        continue;
      SmallVector<Type> blockResultTypes;
      for (Value result : yield.getResults())
        blockResultTypes.push_back(result.getType());
      if (!sawYield) {
        wrapperResultTypes = std::move(blockResultTypes);
        sawYield = true;
      } else if (wrapperResultTypes != blockResultTypes) {
        return task.emitError("mapper replay kernel yields inconsistent "
                              "result types");
      }
    }
    if (!sawYield)
      return task.emitError("mapper replay kernel has no terminal neura.yield");

    MLIRContext *context = task.getContext();
    Location location = task.getLoc();
    ModuleOp temporary = ModuleOp::create(location);
    OpBuilder builder(context);
    builder.setInsertionPointToStart(temporary.getBody());
    SmallVector<Type> argument_types;
    for (BlockArgument argument : kernel.getBody().front().getArguments())
      argument_types.push_back(argument.getType());
    auto wrapper = builder.create<func::FuncOp>(
        location, "__joint_task_mapper_replay__",
        builder.getFunctionType(argument_types, wrapperResultTypes));
    wrapper->setAttr("accelerator", builder.getStringAttr("neura"));
    IRMapping mapping;
    kernel.getBody().cloneInto(&wrapper.getBody(), mapping);
    for (Block &block : wrapper.getBody()) {
      if (auto yield =
              dyn_cast<::mlir::neura::YieldOp>(block.getTerminator())) {
        builder.setInsertionPoint(yield);
        SmallVector<Value> returnValues;
        for (Value result : yield.getResults()) {
          // InsertDataMovPass only wraps Neura consumers. Give the mapper a
          // routed producer for an operand-bearing func.return as well.
          auto move = builder.create<::mlir::neura::DataMovOp>(
              location, result.getType(), result);
          returnValues.push_back(move.getResult());
        }
        builder.create<func::ReturnOp>(location, returnValues);
        yield.erase();
      }
    }

    if (failed(::mlir::amoeba::neura::joint_scheduling::prepareMapperCounterBounds(wrapper)))
      return task.emitError("invalid counter metadata or unprepared mapper body");

    if (allUnitBaseline.getValue()) {
      auto currentArchitecture = llvm::MemoryBuffer::getFile(
          ::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!currentArchitecture ||
          (*currentArchitecture)->getBuffer() != baselineArchitectureText)
        return task.emitError("mapper architecture changed during all-unit "
                              "baseline replay");
      double startup = 0.0;
      std::string startupError;
      if (!deriveStructuralStartupCycles(wrapper, startup, startupError))
        return task.emitError() << "cannot derive structural startup cycles: "
                                << startupError;
      if (!std::isfinite(startup) || startup <= 0.0 ||
          !startupCycles.try_emplace(task.getTaskName(), startup).second)
        return task.emitError("all-unit baseline structural startup is "
                              "invalid or duplicated");
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
      keyStream << "orbit-exact-task-mapper-cache-v1\nrows=" << mapper_rows.getInt()
                << " cols=" << mapper_cols.getInt() << "\n";
      wrapper.print(keyStream, OpPrintingFlags().useLocalScope());
      auto architecture = llvm::MemoryBuffer::getFile(::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!architecture) return task.emitError("cannot read mapper cache architecture");
      if (allUnitBaseline.getValue() &&
          (*architecture)->getBuffer() != baselineArchitectureText)
        return task.emitError("mapper architecture changed during all-unit "
                              "baseline replay");
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
    options.x_tiles = static_cast<int>(mapper_cols.getInt());
    options.y_tiles = static_cast<int>(mapper_rows.getInt());
    manager.addPass(::mlir::neura::createMapToAcceleratorPass(options));
    if (failed(manager.run(temporary)))
      return task.emitError("real per-task mapper replay failed");

      ++mappingCacheMisses;
    }
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
        !mapped_cols || mapped_rows.getInt() != mapper_rows.getInt() ||
        mapped_cols.getInt() != mapper_cols.getInt())
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

    task->setAttr("compiled_ii", compiled_ii);
    task->setAttr("amoeba.mapper_mapping_info", mapping_info);
    task->setAttr("amoeba.mapper_replay_verified", builder.getUnitAttr());
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
    SmallVector<NamedAttribute> profile;
    profile.push_back(builder.getNamedAttr(
        "duration", builder.getI64IntegerAttr(roundedDuration)));
    if (allUnitBaseline.getValue()) {
      auto sourceWork = baselineSourceWorkCounts.find(task.getTaskName());
      if (sourceWork == baselineSourceWorkCounts.end() ||
          sourceWork->second < tripCount.getInt())
        return task.emitError("all-unit baseline has no validated source work "
                              "count");
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
    task->setAttr("profile_info", builder.getDictionaryAttr(profile));
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
      std::string metadataError;
      FailureOr<llvm::SmallVector<
          ::mlir::amoeba::neura::joint_scheduling::TaskMetadata>> metadata =
          ::mlir::amoeba::neura::joint_scheduling::collectAnalyticalTaskMetadata(
              *selectedFunction, metadataError);
      if (failed(metadata)) {
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
            entry.tripCount <= 0 || entry.sourceIterationWorkCount < entry.tripCount) {
          module.emitError() << "all-unit baseline requires a complete, "
                                "source-owned iteration-domain certificate "
                                "and proven current count for task "
                             << entry.name;
          return signalPassFailure();
        }
        if (!selectedTrip || selectedTrip.getInt() != entry.tripCount) {
          module.emitError() << "all-unit baseline current Taskflow firing "
                                "count does not match its selected candidate "
                                "count for task "
                             << entry.name;
          return signalPassFailure();
        }
        baselineSourceWorkCounts[entry.name] = entry.sourceIterationWorkCount;
      }
      if (metadata->size() != tasks.size()) {
        module.emitError("all-unit baseline source count coverage differs "
                         "from the selected task set");
        return signalPassFailure();
      }
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> architecture =
          llvm::MemoryBuffer::getFile(
              ::mlir::amoeba::getNeuraArchitectureSpecFile());
      if (!architecture || (*architecture)->getBuffer().empty()) {
        module.emitError("all-unit baseline cannot read the current mapper "
                         "architecture");
        return signalPassFailure();
      }
      baselineArchitectureText = (*architecture)->getBuffer().str();
    }
    if (!allUnitBaseline.getValue() && startupCycles.size() != tasks.size()) {
      module.emitError("selected score task coverage does not match replay");
      return signalPassFailure();
    }
    for (TaskflowTaskOp task : tasks)
      if (failed(mapTask(task)))
        return signalPassFailure();
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

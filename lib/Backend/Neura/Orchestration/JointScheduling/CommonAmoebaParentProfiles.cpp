//===- CommonAmoebaParentProfiles.cpp -------------------------*- C++ -*-===//

#include "CommonAmoebaParentProfiles.h"

#include "CommonMapperReplayWrapper.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraOps.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "SpatialTaskCandidateSpace.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>

using namespace mlir;

namespace mlir::amoeba::neura::joint_scheduling {
namespace {

constexpr std::array<llvm::StringLiteral, 8> kProfileShapes = {
    "1x1", "1x2", "2x1", "1x3", "3x1", "2x2", "1x4", "4x1"};
constexpr llvm::StringLiteral kCommonProfileFormat = "amoeba-task-profile-v1";
constexpr llvm::StringLiteral kCommonCandidateId = "candidate-0";
constexpr llvm::StringLiteral kCommonStartupProvenance =
    "source-owned-cpp-route-expanded-structural-critical-path-v1";
constexpr llvm::StringLiteral kCommonMapperIIProvenance =
    "native-neura-map-to-accelerator-v1";
constexpr llvm::StringLiteral kCanonicalWitnessFormat =
    "mlir-generic-print-use-local-scope-v1";
constexpr llvm::StringLiteral
    kDiagnosticMinimumLegalProfileInitializationAttr =
        "amoeba.diagnostic_minimum_legal_profile_initialization";

static std::optional<int64_t> checkedFormulaDuration(double startup,
                                                     int64_t ii,
                                                     int64_t tripCount) {
  if (!std::isfinite(startup) || startup <= 0.0 || ii < 1 || tripCount < 1)
    return std::nullopt;
  const long double duration =
      static_cast<long double>(startup) +
      static_cast<long double>(ii) *
          static_cast<long double>(tripCount - 1);
  if (!std::isfinite(duration) || duration <= 0.0L ||
      duration > static_cast<long double>(
                     std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  return static_cast<int64_t>(std::ceil(duration));
}

static std::optional<int64_t> checkedStepsFormulaDuration(int64_t steps,
                                                          int64_t ii,
                                                          int64_t tripCount) {
  if (steps < 1 || ii < 1 || tripCount < 1 ||
      tripCount - 1 >
          (std::numeric_limits<int64_t>::max() - steps) / ii)
    return std::nullopt;
  return steps + ii * (tripCount - 1);
}

static std::optional<int64_t> jsonInteger(const llvm::json::Object &object,
                                          llvm::StringRef key) {
  return object.getInteger(key);
}

static std::optional<llvm::StringRef>
jsonString(const llvm::json::Object &object, llvm::StringRef key) {
  return object.getString(key);
}

static bool printGeneric(ModuleOp module, std::string &text) {
  if (!module)
    return false;
  llvm::raw_string_ostream stream(text);
  module.print(stream,
               OpPrintingFlags().printGenericOpForm().useLocalScope());
  stream.flush();
  return true;
}

static bool printGeneric(func::FuncOp function, std::string &text) {
  if (!function)
    return false;
  llvm::raw_string_ostream stream(text);
  function.print(stream,
                 OpPrintingFlags().printGenericOpForm().useLocalScope());
  stream.flush();
  return true;
}

// Only these attrs carry F45 scheduler output or source-domain certificates
// that are independently verified before semantic projection. Unknown attrs
// remain in the projection and therefore cause an exact-witness mismatch.
static void stripKnownSchedulerAnnotations(ModuleOp module) {
  constexpr std::array<llvm::StringLiteral, 8> functionAttrs = {
      "task_orchestration_summary",
      "amoeba.task_scheduler_task_schedule",
      "amoeba.task_scheduler_dispatch_order",
      "amoeba.task_scheduler_time_unit",
      "amoeba.task_scheduler_time_scale",
      "joint_scheduling_predicted_makespan",
      "joint_scheduling_dispatch_policy",
      "joint_scheduling_scheduler_backend"};
  constexpr std::array<llvm::StringLiteral, 23> taskAttrs = {
      "task_orchestration_info",
      "amoeba.task_scheduler_schedule_info",
      "profile_info",
      "compiled_ii",
      "active_replicas",
      "composed_cgra_shape",
      "composed_cgra_count",
      "cgra_shape",
      "cgra_count",
      "est_latency",
      "amoeba.joint_shape_orientation_fixed",
      "amoeba.selected_cgra_shape",
      "amoeba.selected_cgra_count",
      "amoeba.selected_mapper_tile_rows",
      "amoeba.selected_mapper_tile_cols",
      "amoeba.selected_trip_count",
      "amoeba.mapper_replay_verified",
      "amoeba.mapper_duration_provenance",
      "amoeba.mapper_structural_startup_cycles",
      "amoeba.mapper_actual_ii",
      "amoeba.mapper_ml_prediction_status",
      "amoeba.source_iteration_control_binding",
      "amoeba.source_iteration_source_control_binding"};

  for (func::FuncOp function : module.getOps<func::FuncOp>())
    for (llvm::StringRef name : functionAttrs)
      function->removeAttr(name);
  module.walk([&](taskflow::TaskflowTaskOp task) {
    for (llvm::StringRef name : taskAttrs)
      task->removeAttr(name);
  });
}

static bool parsePositiveField(const llvm::json::Object &object,
                               llvm::StringRef name, int64_t &out,
                               llvm::StringRef context, std::string &error) {
  auto value = jsonInteger(object, name);
  if (!value || *value < 1) {
    error = context.str() + " has an invalid positive " + name.str();
    return false;
  }
  out = *value;
  return true;
}

static bool parseProfileRow(
    const llvm::json::Object &row, const TaskMetadata &task,
    llvm::StringRef expectedShape, int64_t expectedCandidateIndex,
    int64_t expectedCount,
    int64_t expectedMapperRows, int64_t expectedMapperCols,
    int64_t runtimeIICeiling, CommonAmoebaMapperShapeProfile &parsed,
    std::string &error) {
  const std::string context = "common parent profile task=" + task.name +
                              " shape=" + expectedShape.str();
  auto shape = jsonString(row, "composed_cgra_shape");
  auto count = jsonInteger(row, "composed_cgra_count");
  auto mapperRows = jsonInteger(row, "mapper_tile_rows");
  auto mapperCols = jsonInteger(row, "mapper_tile_cols");
  auto tripCount = jsonInteger(row, "sample_trip_count");
  auto latency = jsonInteger(row, "estimated_latency");
  auto startup = row.getNumber("structural_startup_cycles");
  auto succeeded = row.getBoolean("mapper_succeeded");
  auto provenance = jsonString(row, "duration_provenance");
  auto formula = jsonString(row, "duration_formula");
  auto stepsLatency = jsonInteger(row, "steps_formula_estimated_latency");
  auto startupStepsMatch = row.getBoolean("startup_steps_match");
  auto sourceWork = jsonInteger(row, "source_iteration_work_count");
  auto domainStatus = jsonString(row, "source_iteration_domain_status");
  auto domainCertified = row.getBoolean("source_iteration_domain_certified");
  auto domainComplete = row.getBoolean("source_iteration_domain_complete");
  auto wrapperByteCount = jsonInteger(row, "pre_mapper_wrapper_byte_count");
  auto wrapperBytes = jsonString(row, "pre_mapper_wrapper_bytes");
  auto candidateIndex = jsonInteger(row, "candidate_index_in_task");
  if (!shape || *shape != expectedShape || !count || *count != expectedCount ||
      !mapperRows || *mapperRows != expectedMapperRows || !mapperCols ||
      *mapperCols != expectedMapperCols || !tripCount ||
      *tripCount != task.tripCount || !latency || *latency < 1 || !startup ||
      !std::isfinite(*startup) || *startup <= 0.0 || !succeeded ||
      !*succeeded || !provenance ||
      *provenance != kCommonAmoebaParentProfileProvenance || !formula ||
      *formula != kCommonAmoebaParentProfileDurationFormula ||
      !stepsLatency || *stepsLatency < 1 || !startupStepsMatch || !sourceWork ||
      *sourceWork != task.sourceIterationWorkCount || !domainStatus ||
      *domainStatus != task.sourceIterationDomainStatus || !domainCertified ||
      !*domainCertified || !domainComplete || !*domainComplete ||
      !wrapperByteCount || *wrapperByteCount < 1 || !wrapperBytes ||
      static_cast<uint64_t>(*wrapperByteCount) != wrapperBytes->size() ||
      !candidateIndex || *candidateIndex != expectedCandidateIndex ||
      !parsePositiveField(row, "compiled_ii", parsed.compiledII, context,
                          error) ||
      !parsePositiveField(row, "steps", parsed.steps, context, error) ||
      !parsePositiveField(row, "materialized_operation_count",
                          parsed.materializedOperationCount, context, error)) {
    if (error.empty())
      error = context + " has malformed mapper, domain, formula, or wrapper evidence";
    return false;
  }
  if (parsed.compiledII > runtimeIICeiling) {
    error = context + " compiled II exceeds the configured runtime ceiling";
    return false;
  }
  auto formulaLatency = checkedFormulaDuration(
      *startup, parsed.compiledII, *tripCount);
  auto expectedStepsLatency = checkedStepsFormulaDuration(
      parsed.steps, parsed.compiledII, *tripCount);
  if (!formulaLatency || *formulaLatency != *latency ||
      !expectedStepsLatency || *expectedStepsLatency != *stepsLatency ||
      *startupStepsMatch != (*latency == *stepsLatency)) {
    error = context + " duration or steps formula is inconsistent";
    return false;
  }

  parsed.shape = shape->str();
  parsed.cgraCount = *count;
  parsed.mapperTileRows = *mapperRows;
  parsed.mapperTileCols = *mapperCols;
  parsed.sampleTripCount = *tripCount;
  parsed.estimatedLatency = *latency;
  parsed.stepsFormulaEstimatedLatency = *stepsLatency;
  parsed.sourceIterationWorkCount = *sourceWork;
  parsed.sourceIterationDomainStatus = domainStatus->str();
  parsed.structuralStartupCycles = *startup;
  parsed.preMapperWrapperBytes = wrapperBytes->str();
  return true;
}

static bool normalizeSemanticProjection(ModuleOp module,
                                        std::string &projection) {
  if (!module)
    return false;
  stripKnownSchedulerAnnotations(module);
  return printGeneric(module, projection);
}

static bool validateDiagnosticMinimumLegalProfileInitializations(
    llvm::ArrayRef<TaskMetadata> currentTasks,
    llvm::ArrayRef<CommonAmoebaTaskProfiles> authenticatedProfiles,
    int64_t runtimeIICeiling, std::string &error) {
  if (currentTasks.size() != authenticatedProfiles.size()) {
    error = "diagnostic profile initialization task inventory is incomplete";
    return false;
  }

  for (auto [index, task] : llvm::enumerate(currentTasks)) {
    Attribute raw = task.op->getAttr(
        kDiagnosticMinimumLegalProfileInitializationAttr);
    if (!raw)
      continue;

    const std::string context =
        "common parent profile task=" + task.name +
        " diagnostic minimum legal profile initialization";
    auto fail = [&](llvm::StringRef reason) {
      error = context + ": " + reason.str();
      return false;
    };
    if (runtimeIICeiling != 23)
      return fail("requires the explicit diagnostic II ceiling 23");

    auto marker = dyn_cast<DictionaryAttr>(raw);
    if (!marker || marker.size() != 2)
      return fail(
          "must contain exactly composed_cgra_count and composed_cgra_shape");
    auto count = marker.getAs<IntegerAttr>("composed_cgra_count");
    auto shape = marker.getAs<StringAttr>("composed_cgra_shape");
    if (!count || !count.getType().isSignlessInteger(32) || !shape)
      return fail("has malformed count or shape fields");

    const CommonAmoebaTaskProfiles &profiles = authenticatedProfiles[index];
    if (profiles.task != task.name)
      return fail("authenticated mapper-profile task order differs");
    if (profiles.profilesByShape.count("1x1"))
      return fail("is present despite a successful 1-CGRA profile");

    const CommonAmoebaMapperShapeProfile *expected = nullptr;
    // The verifier authenticates profile rows and attempt records in this
    // canonical candidate order. Iterating it here preserves the profiler's
    // original order as the final tie breaker after count and latency.
    for (llvm::StringRef candidateShape : kProfileShapes) {
      auto found = profiles.profilesByShape.find(candidateShape.str());
      if (found == profiles.profilesByShape.end())
        continue;
      const CommonAmoebaMapperShapeProfile &candidate = found->second;
      if (!expected || candidate.cgraCount < expected->cgraCount ||
          (candidate.cgraCount == expected->cgraCount &&
           candidate.estimatedLatency < expected->estimatedLatency))
        expected = &candidate;
    }
    if (!expected || expected->cgraCount <= 1)
      return fail("has no authenticated multi-CGRA fallback profile");
    if (count.getInt() != expected->cgraCount ||
        shape.getValue() != expected->shape)
      return fail(
          "does not match the deterministic minimum legal mapper profile");
  }
  return true;
}

} // namespace

bool verifyCommonAmoebaParentProfiles(
    llvm::StringRef profilePath, llvm::StringRef canonicalModulePath,
    llvm::StringRef expectedFunction, ModuleOp currentModule,
    func::FuncOp currentFunction, llvm::ArrayRef<TaskMetadata> currentTasks,
    llvm::ArrayRef<CommonAmoebaF45TaskChoice> f45Choices,
    int64_t runtimeIICeiling, CommonAmoebaParentProfiles &result,
    std::string &error) {
  auto fail = [&](llvm::StringRef message) {
    error = message.str();
    return false;
  };
  if (profilePath.empty() || canonicalModulePath.empty() ||
      expectedFunction.empty() || !currentModule || !currentFunction ||
      currentTasks.empty() || currentTasks.size() != f45Choices.size() ||
      currentTasks.size() >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) /
              kProfileShapes.size() ||
      (runtimeIICeiling != 20 && runtimeIICeiling != 23))
    return fail("common parent profile verifier has incomplete inputs");

  auto profileBuffer = llvm::MemoryBuffer::getFile(profilePath);
  if (!profileBuffer)
    return fail("cannot read common parent profile " + profilePath.str() +
                ": " + profileBuffer.getError().message());
  auto parsedJson = llvm::json::parse((*profileBuffer)->getBuffer());
  if (!parsedJson)
    return fail("cannot parse common parent profile JSON: " +
                llvm::toString(parsedJson.takeError()));
  const llvm::json::Object *root = parsedJson->getAsObject();
  if (!root)
    return fail("common parent profile root is not an object");

  auto format = jsonString(*root, "format");
  auto functionName = jsonString(*root, "function");
  auto candidateId = jsonString(*root, "candidate_id");
  auto candidateScope = jsonString(*root, "candidate_scope");
  auto provenance = jsonString(*root, "profile_provenance");
  auto sourceCoverage = root->getBoolean(
      "source_iteration_domain_coverage_verified");
  auto startupProvenance = jsonString(*root, "startup_provenance");
  auto mapperIIProvenance = jsonString(*root, "mapper_ii_provenance");
  auto durationFormula = jsonString(*root, "duration_formula");
  auto architecturePath = jsonString(*root, "architecture_spec_path");
  auto architectureText = jsonString(*root, "architecture_spec_text");
  auto witnessFormat = jsonString(*root, "canonical_witness_format");
  auto moduleWitnessBytes = jsonString(*root, "canonical_module_witness_bytes");
  auto functionWitnessBytes =
      jsonString(*root, "canonical_function_witness_bytes");
  auto moduleWitnessSize =
      jsonInteger(*root, "canonical_module_witness_byte_count");
  auto functionWitnessSize =
      jsonInteger(*root, "canonical_function_witness_byte_count");
  auto wholeProgramScheduler =
      root->getBoolean("whole_program_scheduler_invoked");
  auto declaredTasks = jsonInteger(*root, "task_count");
  auto expectedAttempts = jsonInteger(*root, "expected_candidate_count");
  auto completedAttempts = jsonInteger(*root, "completed_candidate_count");
  const llvm::json::Array *shapeDomain = root->getArray("shape_domain");
  const llvm::json::Array *taskRecords = root->getArray("tasks");
  const llvm::json::Array *attemptRecords = root->getArray("candidate_attempts");
  if (!format || *format != kCommonProfileFormat || !functionName ||
      *functionName != expectedFunction || !candidateId ||
      *candidateId != kCommonCandidateId || !candidateScope ||
      *candidateScope != kSearchScope || !provenance ||
      *provenance != kCommonAmoebaParentProfileProvenance ||
      !sourceCoverage || !*sourceCoverage || !startupProvenance ||
      *startupProvenance != kCommonStartupProvenance ||
      !mapperIIProvenance || *mapperIIProvenance != kCommonMapperIIProvenance ||
      !durationFormula ||
      *durationFormula != kCommonAmoebaParentProfileDurationFormula ||
      !architecturePath || !architectureText || !witnessFormat ||
      *witnessFormat != kCanonicalWitnessFormat || !moduleWitnessBytes ||
      !functionWitnessBytes || !moduleWitnessSize ||
      *moduleWitnessSize < 1 ||
      static_cast<uint64_t>(*moduleWitnessSize) != moduleWitnessBytes->size() ||
      !functionWitnessSize || *functionWitnessSize < 1 ||
      static_cast<uint64_t>(*functionWitnessSize) !=
          functionWitnessBytes->size() ||
      !wholeProgramScheduler || *wholeProgramScheduler || !declaredTasks ||
      *declaredTasks != static_cast<int64_t>(currentTasks.size()) ||
      !expectedAttempts || !completedAttempts || !shapeDomain ||
      !taskRecords || !attemptRecords)
    return fail("common parent profile root schema or provenance is invalid");

  if (*expectedAttempts !=
          static_cast<int64_t>(currentTasks.size() * kProfileShapes.size()) ||
      *completedAttempts != *expectedAttempts ||
      attemptRecords->size() != static_cast<size_t>(*expectedAttempts) ||
      taskRecords->size() != currentTasks.size() ||
      shapeDomain->size() != kProfileShapes.size())
    return fail("common parent profile task or complete shape-attempt counts differ");
  for (auto [index, value] : llvm::enumerate(*shapeDomain)) {
    auto shape = value.getAsString();
    if (!shape || *shape != kProfileShapes[index])
      return fail("common parent profile shape domain is not the exact canonical 8-shape order");
  }

  const auto &architecture = ::mlir::neura::getArchitecture();
  const int64_t gridRows = architecture.getMultiCgraRows();
  const int64_t gridCols = architecture.getMultiCgraColumns();
  const int64_t perCgraRows = architecture.getPerCgraRows();
  const int64_t perCgraCols = architecture.getPerCgraColumns();
  if (gridRows != 4 || gridCols != 4 || perCgraRows < 1 || perCgraCols < 1)
    return fail("common parent profiles require the validated 4x4 architecture");
  if (perCgraRows > std::numeric_limits<int64_t>::max() / gridRows ||
      perCgraCols > std::numeric_limits<int64_t>::max() / gridCols)
    return fail("common parent profile mapper dimensions overflow int64");
  auto architectureBuffer =
      llvm::MemoryBuffer::getFile(
          ::mlir::amoeba::getNeuraArchitectureSpecFile());
  if (!architectureBuffer ||
      architecturePath->str() != ::mlir::amoeba::getNeuraArchitectureSpecFile() ||
      architectureText->str() != (*architectureBuffer)->getBuffer().str())
    return fail("common parent profile architecture path or exact text differs from runtime");
  const llvm::json::Object *hardware = root->getObject("hardware_coordinates");
  auto hardwareGridRows = hardware ? jsonInteger(*hardware, "multi_cgra_grid_rows")
                                   : std::nullopt;
  auto hardwareGridCols = hardware ? jsonInteger(*hardware, "multi_cgra_grid_cols")
                                   : std::nullopt;
  auto hardwareTileRows = hardware ? jsonInteger(*hardware, "per_cgra_tile_rows")
                                   : std::nullopt;
  auto hardwareTileCols = hardware ? jsonInteger(*hardware, "per_cgra_tile_cols")
                                   : std::nullopt;
  auto totalMapperRows = hardware ? jsonInteger(*hardware, "total_mapper_rows")
                                  : std::nullopt;
  auto totalMapperCols = hardware ? jsonInteger(*hardware, "total_mapper_cols")
                                  : std::nullopt;
  if (!hardware || !hardwareGridRows || *hardwareGridRows != gridRows ||
      !hardwareGridCols || *hardwareGridCols != gridCols || !hardwareTileRows ||
      *hardwareTileRows != perCgraRows || !hardwareTileCols ||
      *hardwareTileCols != perCgraCols || !totalMapperRows ||
      *totalMapperRows != gridRows * perCgraRows || !totalMapperCols ||
      *totalMapperCols != gridCols * perCgraCols)
    return fail("common parent profile hardware coordinates differ from runtime architecture");

  ParserConfig parserConfig(currentModule.getContext());
  auto canonicalBuffer = llvm::MemoryBuffer::getFile(canonicalModulePath);
  if (!canonicalBuffer)
    return fail("cannot read pre-F45 canonical module " +
                canonicalModulePath.str() + ": " +
                canonicalBuffer.getError().message());
  OwningOpRef<ModuleOp> canonicalModule = parseSourceString<ModuleOp>(
      (*canonicalBuffer)->getBuffer(), parserConfig,
      canonicalModulePath.str());
  if (!canonicalModule)
    return fail("pre-F45 canonical module witness cannot be parsed");
  std::string canonicalText;
  if (!printGeneric(*canonicalModule, canonicalText) ||
      canonicalText != *moduleWitnessBytes)
    return fail("pre-F45 canonical module file differs from the exact common profile witness");
  std::string canonicalFunctionText;
  std::string functionError;
  FailureOr<func::FuncOp> canonicalFunction = selectTaskFunction(
      *canonicalModule, expectedFunction, functionError);
  if (failed(canonicalFunction) ||
      !printGeneric(*canonicalFunction, canonicalFunctionText) ||
      canonicalFunctionText != *functionWitnessBytes)
    return fail("pre-F45 canonical function differs from the exact common profile witness");

  auto currentCandidate = currentFunction->getAttrOfType<StringAttr>(
      "joint_scheduling_candidate_id");
  auto currentScope = currentFunction->getAttrOfType<StringAttr>(
      "joint_scheduling_candidate_scope");
  if (!currentCandidate || currentCandidate.getValue() != *candidateId ||
      !currentScope || currentScope.getValue() != *candidateScope)
    return fail("current F45 function candidate identity or scope differs from common profile");

  result = CommonAmoebaParentProfiles{};
  result.profilePath = profilePath.str();
  result.canonicalModulePath = canonicalModulePath.str();
  result.function = expectedFunction.str();
  result.candidateId = candidateId->str();
  result.candidateScope = candidateScope->str();
  result.profileProvenance = provenance->str();
  result.canonicalModuleWitness = moduleWitnessBytes->str();
  result.canonicalFunctionWitness = functionWitnessBytes->str();
  result.architectureSpecPath = architecturePath->str();
  result.architectureSpecText = architectureText->str();
  result.gridRows = gridRows;
  result.gridCols = gridCols;
  result.perCgraTileRows = perCgraRows;
  result.perCgraTileCols = perCgraCols;
  result.tasks.reserve(currentTasks.size());

  for (auto [taskIndex, task] : llvm::enumerate(currentTasks)) {
    const CommonAmoebaF45TaskChoice &choice = f45Choices[taskIndex];
    const std::string context = "common parent profile task=" + task.name;
    auto taskValue = (*taskRecords)[taskIndex].getAsObject();
    auto recordTask = taskValue ? jsonString(*taskValue, "task") : std::nullopt;
    auto recordTrip = taskValue ? jsonInteger(*taskValue, "sample_trip_count")
                                : std::nullopt;
    auto recordSourceWork =
        taskValue ? jsonInteger(*taskValue, "source_iteration_work_count")
                  : std::nullopt;
    auto recordDomainStatus =
        taskValue ? jsonString(*taskValue, "source_iteration_domain_status")
                  : std::nullopt;
    auto domainCertified = taskValue
                               ? taskValue->getBoolean(
                                     "source_iteration_domain_certified")
                               : std::nullopt;
    auto domainComplete = taskValue
                              ? taskValue->getBoolean(
                                    "source_iteration_domain_complete")
                              : std::nullopt;
    const llvm::json::Array *profiles =
        taskValue ? taskValue->getArray("profiles") : nullptr;
    if (!taskValue || !recordTask || *recordTask != task.name || !recordTrip ||
        *recordTrip != task.tripCount || task.tripCount < 1 || !recordSourceWork ||
        *recordSourceWork != task.sourceIterationWorkCount ||
        !task.sourceIterationDomainCertified ||
        !task.sourceIterationDomainComplete || !recordDomainStatus ||
        *recordDomainStatus != task.sourceIterationDomainStatus ||
        !domainCertified || !*domainCertified || !domainComplete ||
        !*domainComplete || !profiles || choice.task != task.name ||
        choice.actualTripCount != task.tripCount || choice.activeReplicas < 1 ||
        choice.activeReplicas > 4 || choice.selectedCgraCount < 1 ||
        choice.mapperTileRows < 1 || choice.mapperTileCols < 1)
      return fail(context + " task/source-domain/F45-choice binding is incomplete or stale");

    CommonAmoebaTaskProfiles taskProfiles;
    taskProfiles.task = task.name;
    taskProfiles.sampleTripCount = *recordTrip;
    taskProfiles.sourceIterationWorkCount = *recordSourceWork;
    taskProfiles.sourceIterationDomainStatus = recordDomainStatus->str();
    int64_t previousShapeIndex = -1;
    for (const llvm::json::Value &rawProfile : *profiles) {
      const llvm::json::Object *profile = rawProfile.getAsObject();
      auto shape = profile ? jsonString(*profile, "composed_cgra_shape")
                           : std::nullopt;
      if (!profile || !shape)
        return fail(context + " has a malformed shape profile row");
      auto shapeIterator = llvm::find(kProfileShapes, *shape);
      if (shapeIterator == kProfileShapes.end())
        return fail(context + " contains a shape outside the exact profile domain");
      const int64_t shapeIndex =
          static_cast<int64_t>(std::distance(kProfileShapes.begin(), shapeIterator));
      if (shapeIndex <= previousShapeIndex)
        return fail(context + " successful shape rows are not in canonical order");
      previousShapeIndex = shapeIndex;
      int64_t rows = 0;
      int64_t cols = 0;
      if (shape->split('x').first.getAsInteger(10, rows) ||
          shape->split('x').second.getAsInteger(10, cols) || rows < 1 ||
          cols < 1)
        return fail(context + " has a malformed canonical shape spelling");
      CommonAmoebaMapperShapeProfile parsedProfile;
      if (!parseProfileRow(*profile, task, *shape, shapeIndex + 1,
                           rows * cols,
                           rows * perCgraRows, cols * perCgraCols,
                           runtimeIICeiling, parsedProfile, error))
        return false;
      if (!taskProfiles.profilesByShape
               .emplace(parsedProfile.shape, std::move(parsedProfile))
               .second)
        return fail(context + " contains a duplicate successful shape row");
    }

    const CommonAmoebaMapperShapeProfile *firstProfile = nullptr;
    for (const auto &entry : taskProfiles.profilesByShape) {
      if (!firstProfile)
        firstProfile = &entry.second;
      else if (entry.second.preMapperWrapperBytes !=
               firstProfile->preMapperWrapperBytes)
        return fail(context + " profile rows disagree on the pre-mapper wrapper bytes");
    }
    if (!firstProfile)
      return fail(context + " contains no successful common mapper shape profile");

    OwningOpRef<ModuleOp> wrapperOwner(ModuleOp::create(task.op->getLoc()));
    std::string wrapperError;
    func::FuncOp wrapper = buildCommonMapperReplayWrapper(
        task.op, *wrapperOwner, "__joint_task_mapper_replay__", wrapperError);
    if (!wrapper)
      return fail(context + " cannot rebuild the current source mapper wrapper: " +
                  wrapperError);
    const std::string currentWrapper = printCommonMapperReplayWrapper(wrapper);
    if (currentWrapper != firstProfile->preMapperWrapperBytes)
      return fail(context + " current source mapper wrapper differs byte-for-byte from common pre-mapper witness");

    const CommonAmoebaMapperShapeProfile *selected = nullptr;
    auto selectedIterator =
        taskProfiles.profilesByShape.find(choice.selectedShape);
    if (selectedIterator != taskProfiles.profilesByShape.end())
      selected = &selectedIterator->second;
    if (!selected || selected->cgraCount != choice.selectedCgraCount ||
        selected->mapperTileRows != choice.mapperTileRows ||
        selected->mapperTileCols != choice.mapperTileCols ||
        selected->sampleTripCount != choice.actualTripCount)
      return fail(context + " original F45 selected shape/count/dimensions has no matching successful common profile");

    // The unchanged F45 allocator consumes this common profile's structural
    // startup duration. Its old profiler used mapper steps for startup; those
    // two values can differ and must not be conflated in the common path.
    const int64_t expectedSourceDuration =
        selected->estimatedLatency / choice.activeReplicas +
        (selected->estimatedLatency % choice.activeReplicas != 0 ? 1 : 0);
    if (choice.compiledII != selected->compiledII ||
        choice.steps != selected->steps ||
        choice.materializedOperationCount != selected->materializedOperationCount ||
        choice.sourceSchedulerDuration != expectedSourceDuration)
      return fail(context + " original F45 profile_info differs from the selected common mapper profile and parent/N duration");

    result.taskIndices.emplace(task.name, static_cast<unsigned>(taskIndex));
    result.tasks.push_back(std::move(taskProfiles));
  }

  for (auto [attemptIndex, rawAttempt] : llvm::enumerate(*attemptRecords)) {
    const llvm::json::Object *attempt = rawAttempt.getAsObject();
    const size_t taskIndex = attemptIndex / kProfileShapes.size();
    const size_t shapeIndex = attemptIndex % kProfileShapes.size();
    const TaskMetadata &task = currentTasks[taskIndex];
    auto taskName = attempt ? jsonString(*attempt, "task") : std::nullopt;
    auto candidateIndex = attempt
                              ? jsonInteger(*attempt,
                                            "candidate_index_in_task")
                              : std::nullopt;
    auto shape = attempt ? jsonString(*attempt, "shape") : std::nullopt;
    auto count = attempt ? jsonInteger(*attempt, "composed_cgra_count")
                         : std::nullopt;
    auto profileCreated = attempt ? attempt->getBoolean("profile_created")
                                  : std::nullopt;
    auto mapperSucceeded = attempt ? attempt->getBoolean("mapper_succeeded")
                                   : std::nullopt;
    const std::string key(kProfileShapes[shapeIndex]);
    const bool hasProfile =
        result.tasks[taskIndex].profilesByShape.count(key) != 0;
    int64_t rows = 0;
    int64_t cols = 0;
    if (llvm::StringRef(key).split('x').first.getAsInteger(10, rows) ||
        llvm::StringRef(key).split('x').second.getAsInteger(10, cols))
      return fail("internal canonical profile shape parse failed");
    if (!attempt || !taskName || *taskName != task.name || !candidateIndex ||
        *candidateIndex != static_cast<int64_t>(shapeIndex + 1) || !shape ||
        *shape != key || !count || *count != rows * cols || !profileCreated ||
        !mapperSucceeded || *profileCreated != *mapperSucceeded ||
        *mapperSucceeded != hasProfile)
      return fail("common parent profile candidate-attempt inventory is incomplete or inconsistent");
  }

  // This F45-only diagnostic marker is meaningful only when it agrees with
  // the exact profile inventory authenticated above. Validate it before
  // projecting it out of the semantic comparison below.
  if (!validateDiagnosticMinimumLegalProfileInitializations(
          currentTasks, result.tasks, runtimeIICeiling, error))
    return false;

  OwningOpRef<ModuleOp> canonicalProjection(
      cast<ModuleOp>((*canonicalModule)->clone()));
  OwningOpRef<ModuleOp> currentProjection(
      cast<ModuleOp>(currentModule->clone()));
  std::string projectionError;
  FailureOr<func::FuncOp> projectedFunction = selectTaskFunction(
      *currentProjection, currentFunction.getSymName(), projectionError);
  if (failed(projectedFunction))
    return fail(
        "cannot select current function for common semantic projection: " +
        projectionError);
  projectedFunction->walk([&](taskflow::TaskflowTaskOp task) {
    task->removeAttr(kDiagnosticMinimumLegalProfileInitializationAttr);
  });
  std::string canonicalSemanticText;
  std::string currentSemanticText;
  if (!normalizeSemanticProjection(*canonicalProjection,
                                   canonicalSemanticText) ||
      !normalizeSemanticProjection(*currentProjection,
                                   currentSemanticText) ||
      canonicalSemanticText != currentSemanticText)
    return fail("current F45 module changes pre-F45 taskflow semantics outside the verified scheduler/source-certificate allowlist");

  return true;
}

bool refreshCommonAmoebaSourceIterationBindings(func::FuncOp function,
                                                std::string &error) {
  if (!function) {
    error = "common source-binding refresh requires a selected function";
    return false;
  }
  bool failedBinding = false;
  function.walk([&](taskflow::TaskflowTaskOp task) {
    if (failedBinding)
      return WalkResult::interrupt();
    auto failTask = [&](llvm::StringRef reason) {
      error = "common source-binding refresh task=" +
              task.getTaskName().str() + ": " + reason.str();
      failedBinding = true;
      return WalkResult::interrupt();
    };
    if (task->hasAttr(kSourceIterationCapturePendingAttr))
      return failTask("unresolved source capture cannot be refreshed");
    if (task->hasAttr(kSourceIterationPartitionProofAttr))
      return failTask("parent common-profile mode does not accept partition rewrites");
    auto domain = task->getAttrOfType<DictionaryAttr>(kSourceIterationDomainAttr);
    if (!domain)
      return failTask("exact source iteration-domain certificate is missing");
    auto complete = domain.getAs<BoolAttr>("complete");
    if (!complete || !complete.getValue())
      return failTask("source iteration-domain certificate is not complete");
    std::string parseError;
    FailureOr<SourceIterationDomainInfo> info =
        parseSourceIterationDomain(task, parseError);
    if (failed(info))
      return failTask(parseError);
    const std::string domainWitness =
        sourceIterationDomainCanonicalWitness(*info);
    const std::string expected =
        currentSourceIterationControlBinding(task, domainWitness);
    if (expected.empty())
      return failTask("cannot reconstruct exact current source-control binding");

    OpBuilder builder(task.getContext());
    const StringAttr binding = builder.getStringAttr(expected);
    task->setAttr(kSourceIterationControlBindingAttr, binding);
    task->setAttr(kSourceIterationSourceControlBindingAttr, binding);
    return WalkResult::advance();
  });
  return !failedBinding;
}

} // namespace mlir::amoeba::neura::joint_scheduling

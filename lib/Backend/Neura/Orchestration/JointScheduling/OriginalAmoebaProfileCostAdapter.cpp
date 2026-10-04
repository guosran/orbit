//===- OriginalAmoebaProfileCostAdapter.cpp -------------------*- C++ -*-===//
// Builds a selected-shape cost catalogue from corrected original-AMOEBA
// mapper-body and successful-profile evidence. This pass never searches
// shapes: it evaluates only each task's recorded original selected shape.
//===----------------------------------------------------------------------===//

#include "OriginalAmoebaProfileCostAdapter.h"

#include "AnalyticalMLPInference.h"
#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/Orchestration/JointScheduling/MapperFeatureExtractor.h"
#include "Conversion/NeuraConversionPasses.h"
#include "MapperCostAnalysis.h"
#include "OriginalAmoebaCtrlToDataFlowPass.h"
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
constexpr std::array<llvm::StringLiteral, 8> kExpectedProfileShapes = {
    "1x1", "1x2", "2x1", "1x3", "3x1", "1x4", "2x2", "4x1"};

struct SelectedProfile {
  std::string shape;
  int64_t cgraCount = 0;
  int64_t compiledII = 0;
  int64_t steps = 0;
  int64_t sampleTripCount = 0;
  int64_t materializedOperationCount = 0;
  int64_t estimatedLatency = 0;
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
  MLPEnsemblePrediction prediction;
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

static bool checkedProfileDuration(int64_t ii, int64_t trip, int64_t steps,
                                   int64_t &duration) {
  if (ii < 1 || trip < 1 || steps < 1 ||
      trip - 1 > (std::numeric_limits<int64_t>::max() - steps) / ii)
    return false;
  duration = ii * (trip - 1) + steps;
  return duration > 0;
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
  if (!profiles) {
    error = "profile file omits profile records for task " + taskName.str();
    return false;
  }

  const json::Object *selected = nullptr;
  std::set<std::string> uniqueShapes;
  for (const json::Value &rawProfile : *profiles) {
    const json::Object *candidate = rawProfile.getAsObject();
    std::optional<StringRef> shape =
        candidate ? candidate->getString("composed_cgra_shape") : std::nullopt;
    if (!candidate || !shape || shape->empty() ||
        !uniqueShapes.insert(shape->str()).second) {
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
  if (uniqueShapes.size() != kExpectedProfileShapes.size() ||
      !llvm::all_of(kExpectedProfileShapes, [&](StringRef expected) {
        return uniqueShapes.count(expected.str()) != 0;
      })) {
    error = "profile file does not contain the exact original four-CGRA "
            "shape domain for " +
            taskName.str();
    return false;
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
  metadata["iteration_domain_coverage_status"] = "pending";
  metadata["iteration_domain_coverage_verified"] = false;
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

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](StringRef message) {
      module.emitError() << message;
      signalPassFailure();
    };
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
    if (!profileTasks || profileTasks->size() != tasks.size()) {
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

    FormalMax4MLPEnsemble model;
    if (!model.load(ensembleFile, checkpointDirectory, architectureContract,
                    error)) {
      fail(error);
      return;
    }
    std::string architectureText;
    if (!readTextFile(architecturePath, architectureText, error)) {
      fail(error);
      return;
    }
    MLCostCacheResources modelResources =
        model.makeCacheResources(architectureText);

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
          !activeReplicas || activeReplicas.getInt() != 1 ||
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
                              item.profile, error)) {
        metadata.op.emitError() << error;
        return signalPassFailure();
      }
      auto irII = getInteger(profileInfo, "compiled_ii");
      auto irDuration = getInteger(profileInfo, "duration");
      auto irOperations =
          getInteger(profileInfo, "materialized_operation_count");
      auto irTrip = getInteger(profileInfo, "sample_trip_count");
      auto irSteps = getInteger(profileInfo, "steps");
      int64_t formulaDuration = 0;
      if (!irII || !irDuration || !irOperations || !irTrip || !irSteps ||
          *irII != item.profile.compiledII ||
          *irDuration != item.profile.estimatedLatency ||
          *irOperations != item.profile.materializedOperationCount ||
          *irTrip != metadata.tripCount ||
          *irTrip != item.profile.sampleTripCount ||
          *irSteps != item.profile.steps ||
          !checkedProfileDuration(*irII, *irTrip, *irSteps, formulaDuration) ||
          formulaDuration != *irDuration) {
        metadata.op.emitError()
            << "selected profile JSON, original profile_info, "
               "and compiler-inferred trip count do not "
               "match exactly";
        return signalPassFailure();
      }
      if (!fillFeatures(item, module.getContext(), model, error)) {
        metadata.op.emitError() << error;
        return signalPassFailure();
      }
      selectedTasks.push_back(std::move(item));
    }

    if (sourceRepository.empty() || !isSourceCommit(sourceCommit)) {
      fail("source repository and valid Git commit provenance are required");
      return;
    }
    auto binding =
        makeBinding(bodyExport, profileFile, graph.getValue(), architecturePath,
                    ensembleFile, checkpointDirectory, sourceRepository,
                    sourceCommit, model, modelResources, selectedTasks);
    if (!writeAtomically(
            outputFile,
            [&](raw_ostream &os) {
              return writeCostCatalog(
                  os, function, modelNamespace, sourceRepository, sourceCommit,
                  architecturePath, architectureContract, graph.getValue(),
                  model, selectedTasks, std::move(binding));
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

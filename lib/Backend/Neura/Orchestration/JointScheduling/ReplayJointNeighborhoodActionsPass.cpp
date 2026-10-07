//===- ReplayJointNeighborhoodActionsPass.cpp ----------------------------===//
//
// Replay one caller-supplied typed action path through the source-owned
// neighborhood materializers. This pass validates provenance and legality;
// it does not enumerate a frontier, predict cost, score, or choose a winner.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/ReplicaOutputCoordinateProof.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProveStaticActiveTransferShapesPass.h"
#include "JointNeighborhoodActions.h"
#include "Backend/Neura/Orchestration/JointScheduling/SourceIterationDomainPartitionProof.h"
#include "TaskflowFissionSourceReplay.h"
#include "Backend/Neura/Orchestration/SourceIterationDomain.h"
#include "Backend/Neura/Transforms/Optimizations/TaskflowFission.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::amoeba::neura::joint_scheduling;
namespace json = llvm::json;

namespace {

constexpr StringLiteral kActionSchema =
    "orbit-joint-neighborhood-typed-actions-v1";
constexpr StringLiteral kPartitionLineage =
    "amoeba.neighborhood.partition_lineage.v1";

struct ReplayInputs {
  std::string canonicalPath;
  std::string candidatePath;
  std::string function;
  std::string stage;
  unsigned maxPartitionFactor = 8;
  uint64_t maxFissionActionsPerTask = 64;
  std::vector<NeighborhoodAction> actions;
  std::vector<NeighborhoodShape> initialShapes;
  std::vector<NeighborhoodShape> requestedInitialShapes;
  std::vector<NeighborhoodAction> canonicalFissionActions;
  std::vector<std::string> fissionDiagnostics;
  bool hasRequestedInitialShapes = false;
  std::string actionBytes;
  std::string canonicalBytes;
  std::string candidateBytes;
  std::string preparedSourcePath;
  std::string preparedSourceBytes;
};

static bool readFile(StringRef path, std::string &bytes, std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read " + path.str() + ": " + buffer.getError().message();
    return false;
  }
  bytes = (*buffer)->getBuffer().str();
  return true;
}

static std::string jsonText(json::Value value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value << "\n";
  stream.flush();
  return text;
}

static std::string moduleText(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, OpPrintingFlags().printGenericOpForm());
  stream << "\n";
  stream.flush();
  return text;
}

static std::string candidateComparisonText(ModuleOp module,
                                           StringRef functionName,
                                           std::string &error) {
  OwningOpRef<ModuleOp> normalized = module.clone();
  if (!normalized) {
    error = "cannot clone candidate module for exact replay comparison";
    return {};
  }
  FailureOr<func::FuncOp> function =
      selectTaskFunction(normalized.get(), functionName, error);
  if (failed(function))
    return {};
  // Search adds this derived identifier while preparing the graph-fact key.
  // It does not change the candidate program. Preserve every other operation
  // and attribute so source, body, fission, and proof mismatches fail closed.
  (*function)->removeAttr("amoeba.graph_variant_id");
  return moduleText(normalized.get());
}

static bool parseBoundModule(StringRef text, MLIRContext *context,
                             StringRef label, OwningOpRef<ModuleOp> &module,
                             std::string &error) {
  module = parseSourceString<ModuleOp>(text, context);
  if (!module) {
    error = "cannot parse " + label.str() + " as a ModuleOp";
    return false;
  }
  if (failed(verify(module->getOperation()))) {
    error = label.str() + " does not verify";
    return false;
  }
  return true;
}

static bool parsePrimitive(const json::Object &object,
                           NeighborhoodPrimitive &primitive) {
  auto kind = object.getString("kind");
  auto firstTask = object.getString("firstTask");
  auto secondTask = object.getString("secondTask");
  auto mode = object.getString("mode");
  auto axis = object.getInteger("axis");
  auto factor = object.getInteger("factor");
  const json::Array *leftNodes = object.getArray("leftNodes");
  if (!kind || !firstTask || !secondTask || !mode || !axis || !factor)
    return false;
  if (object.get("leftNodes") && !leftNodes)
    return false;
  primitive.kind = kind->str();
  primitive.firstTask = firstTask->str();
  primitive.secondTask = secondTask->str();
  primitive.mode = mode->str();
  primitive.axis = *axis;
  primitive.factor = *factor;
  if (leftNodes)
    for (const json::Value &value : *leftNodes) {
      auto node = value.getAsInteger();
      if (!node || *node < 0 ||
          static_cast<uint64_t>(*node) >
              std::numeric_limits<unsigned>::max())
        return false;
      primitive.leftNodes.push_back(static_cast<unsigned>(*node));
    }
  const bool fission = primitive.kind == "fission";
  if ((fission && primitive.leftNodes.empty()) ||
      (!fission && !primitive.leftNodes.empty()))
    return false;
  for (size_t index = 1; index < primitive.leftNodes.size(); ++index)
    if (primitive.leftNodes[index - 1] >= primitive.leftNodes[index])
      return false;
  return true;
}

static bool parseAction(const json::Object &object,
                        NeighborhoodAction &action) {
  auto family = object.getString("family");
  auto label = object.getString("label");
  auto shapeTask = object.getString("shapeTask");
  auto shapeRows = object.getInteger("shapeRows");
  auto shapeCols = object.getInteger("shapeCols");
  auto primitives = object.getArray("primitives");
  if (!family || !label || !shapeTask || !shapeRows || !shapeCols ||
      !primitives)
    return false;
  action.family = family->str();
  action.label = label->str();
  action.shapeTask = shapeTask->str();
  action.shapeRows = *shapeRows;
  action.shapeCols = *shapeCols;
  if (const json::Value *reset = object.get("canonicalReset")) {
    auto parsedReset = reset->getAsBoolean();
    if (!parsedReset)
      return false;
    action.canonicalReset = *parsedReset;
  }
  for (const json::Value &value : *primitives) {
    const json::Object *primitiveObject = value.getAsObject();
    if (!primitiveObject)
      return false;
    NeighborhoodPrimitive primitive;
    if (!parsePrimitive(*primitiveObject, primitive))
      return false;
    action.primitives.push_back(std::move(primitive));
  }
  if (action.family == "fission" &&
      (action.primitives.size() != 1 || action.canonicalReset ||
       !action.shapeTask.empty() || action.shapeRows != 0 ||
       action.shapeCols != 0 || action.primitives.front().kind != "fission"))
    return false;
  if (action.family != "fission" &&
      llvm::any_of(action.primitives, [](const NeighborhoodPrimitive &p) {
        return p.kind == "fission";
      }))
    return false;
  return !action.family.empty() && !action.label.empty();
}

static bool isKnownFamily(StringRef family) {
  return family == "identity" || family == "canonical-reset" ||
         family == "lineage-replacement" || family == "shape" ||
         family == "replica" || family == "tiling" || family == "k-tiling" ||
         family == "producer-consumer-co-tiling" ||
         family == "producer-consumer-co-k-tiling" || family == "fusion" ||
         family == "fission" ||
         family == "fusion-plus-shape" || family == "fusion-plus-tiling" ||
         family == "tiling-plus-fusion" || family == "sibling-fusion";
}

static unsigned familyStage(StringRef family) {
  if (family == "replica")
    return 3;
  if (family == "tiling" || family == "k-tiling" ||
      family == "producer-consumer-co-tiling" ||
      family == "producer-consumer-co-k-tiling")
    return 4;
  if (family == "fusion" || family == "fusion-plus-shape" ||
      family == "fusion-plus-tiling" || family == "tiling-plus-fusion" ||
      family == "sibling-fusion")
    return 5;
  if (family == "fission")
    return 6;
  return 1;
}

static unsigned stageNumber(StringRef stage) {
  if (stage == "s1" || stage == "shape" || stage == "shape-only")
    return 1;
  if (stage == "s2" || stage == "shape-temporal")
    return 2;
  if (stage == "s3" || stage == "shape-temporal-replica")
    return 3;
  if (stage == "s4" || stage == "shape-temporal-replica-tiling")
    return 4;
  if (stage == "s5" || stage == "full-joint" ||
      stage == "shape-temporal-replica-tiling-fusion")
    return 5;
  if (stage == "s6" || stage == "full-joint-fission" ||
      stage == "shape-temporal-replica-tiling-fusion-fission")
    return 6;
  return 0;
}

static bool supportedShape(int64_t rows, int64_t cols) {
  return (rows == 1 && (cols == 1 || cols == 2 || cols == 3 || cols == 4)) ||
         (cols == 1 && (rows == 2 || rows == 3 || rows == 4)) ||
         (rows == 2 && cols == 2);
}

static bool actionEqual(const NeighborhoodAction &lhs,
                        const NeighborhoodAction &rhs) {
  if (lhs.family != rhs.family || lhs.label != rhs.label ||
      lhs.shapeTask != rhs.shapeTask || lhs.shapeRows != rhs.shapeRows ||
      lhs.shapeCols != rhs.shapeCols ||
      lhs.canonicalReset != rhs.canonicalReset ||
      lhs.primitives.size() != rhs.primitives.size())
    return false;
  for (size_t index = 0; index < lhs.primitives.size(); ++index) {
    const NeighborhoodPrimitive &left = lhs.primitives[index];
    const NeighborhoodPrimitive &right = rhs.primitives[index];
    if (left.kind != right.kind || left.firstTask != right.firstTask ||
        left.secondTask != right.secondTask || left.mode != right.mode ||
        left.axis != right.axis || left.factor != right.factor ||
        left.leftNodes != right.leftNodes)
      return false;
  }
  return true;
}

static bool readInputs(const json::Object &root, StringRef canonicalInput,
                       StringRef candidateInput, StringRef functionOption,
                       StringRef stageOption, unsigned factorOption,
                       StringRef preparedSourceOption,
                       uint64_t maxFissionActionsPerTaskOption,
                       MLIRContext *context, ModuleOp inputModule,
                       ReplayInputs &inputs, OwningOpRef<ModuleOp> &canonical,
                       OwningOpRef<ModuleOp> &preparedSource,
                       std::string &error) {
  auto schema = root.getString("schema");
  auto canonicalPath = root.getString("canonicalInput");
  auto candidatePath = root.getString("candidateInput");
  auto function = root.getString("function");
  auto stage = root.getString("stage");
  auto factor = root.getInteger("maxPartitionFactor");
  auto fissionCap = root.getInteger("maxFissionActionsPerTask");
  auto preparedSourcePath = root.getString("preparedSourceInput");
  auto preparedSourceExactBytes = root.getString("preparedSourceExactBytes");
  auto actions = root.getArray("actions");
  if (!schema || *schema != kActionSchema || !canonicalPath || !candidatePath ||
      !function || !stage || !factor || !actions) {
    error = "action file has missing fields or an unsupported schema";
    return false;
  }
  if (*canonicalPath != canonicalInput || *candidatePath != candidateInput) {
    error = "action-file input paths differ from the pass input bindings";
    return false;
  }
  if (*function != functionOption || *stage != stageOption ||
      *factor != static_cast<int64_t>(factorOption)) {
    error = "action-file function, stage, or cumulative cap differs from pass "
            "options";
    return false;
  }
  unsigned enabledStage = stageNumber(*stage);
  if (!enabledStage) {
    error = "unknown neighborhood replay stage";
    return false;
  }
  if (factorOption != 1 && factorOption != 2 && factorOption != 4 &&
      factorOption != 8) {
    error = "max-partition-factor must be one of {1, 2, 4, 8}";
    return false;
  }
  inputs.canonicalPath = canonicalPath->str();
  inputs.candidatePath = candidatePath->str();
  inputs.function = function->str();
  inputs.stage = stage->str();
  inputs.maxPartitionFactor = factorOption;
  inputs.maxFissionActionsPerTask = maxFissionActionsPerTaskOption;
  const bool fissionStage = enabledStage == 6;
  if (fissionStage) {
    if (preparedSourceOption.empty() || !preparedSourcePath ||
        !preparedSourceExactBytes || !fissionCap ||
        *preparedSourcePath != preparedSourceOption ||
        *fissionCap != static_cast<int64_t>(maxFissionActionsPerTaskOption) ||
        maxFissionActionsPerTaskOption == 0) {
      error = "full-joint-fission replay requires exact prepared-source-file and cut-cap bindings";
      return false;
    }
    inputs.preparedSourcePath = preparedSourceOption.str();
    if (!readFile(preparedSourceOption, inputs.preparedSourceBytes, error))
      return false;
    if (inputs.preparedSourceBytes != *preparedSourceExactBytes) {
      error = "prepared-source-file bytes differ from the action-file source binding";
      return false;
    }
    preparedSource = parseSourceString<ModuleOp>(inputs.preparedSourceBytes,
                                                 context);
    if (!preparedSource || failed(verify(preparedSource->getOperation()))) {
      error = "prepared-source-file is not a verified ModuleOp";
      return false;
    }
    FailureOr<std::vector<NeighborhoodAction>> fissionActions =
        enumerateTaskflowFissionNeighborhoodActions(
            preparedSource.get(), inputs.function,
            maxFissionActionsPerTaskOption, inputs.fissionDiagnostics,
            error);
    if (failed(fissionActions))
      return false;
    inputs.canonicalFissionActions = std::move(*fissionActions);
  } else if (!preparedSourceOption.empty() || preparedSourcePath ||
             preparedSourceExactBytes || fissionCap) {
    error = "prepared source and fission cut cap are valid only at full-joint-fission";
    return false;
  }

  bool sawNonFissionAction = false;
  std::set<std::string> fissionTargets;
  for (const json::Value &value : *actions) {
    const json::Object *actionObject = value.getAsObject();
    NeighborhoodAction action;
    if (!actionObject || !parseAction(*actionObject, action)) {
      error = "action list contains a malformed typed action";
      return false;
    }
    if (!isKnownFamily(action.family)) {
      error = "unknown_neighborhood_action_family:" + action.family;
      return false;
    }
    if (familyStage(action.family) > enabledStage) {
      error = "action_disabled_at_stage:" + action.label;
      return false;
    }
    if (action.family == "fission") {
      if (!fissionStage || sawNonFissionAction ||
          action.primitives.size() != 1 ||
          !fissionTargets
               .insert(action.primitives.front().firstTask)
               .second ||
          !llvm::any_of(inputs.canonicalFissionActions,
                        [&](const NeighborhoodAction &legal) {
            return actionEqual(action, legal);
          })) {
        error = "fission action must be a unique complete-source enumerated prefix action";
        return false;
      }
    } else {
      sawNonFissionAction = true;
    }
    if (action.family == "lineage-replacement") {
      error = "lineage_replacement_requires_an_explicit_canonical_path:" +
              action.label;
      return false;
    }
    for (const NeighborhoodPrimitive &primitive : action.primitives) {
      if (primitive.factor != 1 && primitive.factor != 2 &&
          primitive.factor != 4 && primitive.factor != 8) {
        error = "unsupported_partition_factor:" + action.label;
        return false;
      }
      if (static_cast<uint64_t>(primitive.factor) > factorOption) {
        error = "action_factor_exceeds_cumulative_cap:" + action.label;
        return false;
      }
    }
    inputs.actions.push_back(std::move(action));
  }

  // actionBytes is the exact buffer parsed by the caller. Do not reread the
  // path here: that could bind the emitted witness to bytes different from
  // the actions actually replayed if the file changes concurrently.
  if (!readFile(canonicalInput, inputs.canonicalBytes, error) ||
      !readFile(candidateInput, inputs.candidateBytes, error))
    return false;
  OwningOpRef<ModuleOp> parsedCandidate;
  if (!parseBoundModule(inputs.candidateBytes, context, "candidate input",
                        parsedCandidate, error))
    return false;
  if (moduleText(inputModule) != moduleText(*parsedCandidate)) {
    error = "pass module differs from the exact candidate-input file";
    return false;
  }
  if (!parseBoundModule(inputs.canonicalBytes, context, "canonical input",
                        canonical, error))
    return false;
  if (fissionStage &&
      failed(verifyTaskflowFissionCanonicalLowering(
          preparedSource.get(), canonical.get(), error)))
    return false;

  if (fissionStage && !fissionTargets.empty() &&
      inputs.candidateBytes != inputs.canonicalBytes) {
    error = "source fission replay must start from the exact canonical candidate input";
    return false;
  }

  auto canonicalFunction =
      selectTaskFunction(*canonical, inputs.function, error);
  auto candidateFunction =
      selectTaskFunction(inputModule, inputs.function, error);
  if (failed(canonicalFunction) || failed(candidateFunction)) {
    if (error.empty())
      error = "cannot select canonical and candidate Taskflow functions";
    return false;
  }
  if (root.get("initialShapes") && !root.getArray("initialShapes")) {
    error = "initialShapes must be a JSON array when present";
    return false;
  }
  auto actionsInitialShapes = root.getArray("initialShapes");
  std::map<std::string, NeighborhoodShape> byName;
  if (actionsInitialShapes) {
    for (const json::Value &value : *actionsInitialShapes) {
      const json::Object *shapeObject = value.getAsObject();
      if (!shapeObject) {
        error = "initialShapes contains a non-object value";
        return false;
      }
      auto name = shapeObject->getString("task");
      auto rows = shapeObject->getInteger("rows");
      auto cols = shapeObject->getInteger("cols");
      if (!name || !rows || !cols || !supportedShape(*rows, *cols) ||
          !byName
               .emplace(name->str(),
                        NeighborhoodShape{name->str(), *rows, *cols})
               .second) {
        error = "initialShapes contains a malformed or duplicate task shape";
        return false;
      }
    }
  }
  inputs.hasRequestedInitialShapes = actionsInitialShapes != nullptr;
  if (!fissionTargets.empty()) {
    for (auto &[name, shape] : byName)
      inputs.requestedInitialShapes.push_back(shape);
  } else {
    candidateFunction->walk([&](taskflow::TaskflowTaskOp task) {
      std::string name = task.getTaskName().str();
      auto found = byName.find(name);
      inputs.initialShapes.push_back(
          found == byName.end() ? NeighborhoodShape{name, 1, 1} : found->second);
    });
    if (actionsInitialShapes && byName.size() != inputs.initialShapes.size()) {
      error = "initialShapes does not bind exactly the current candidate tasks";
      return false;
    }
  }
  // Preserve source task order so both the initial history and the selected
  // shapes can be authenticated by the complete-program search controller.
  return true;
}

static void suspendSourceCertificates(ModuleOp module) {
  module.walk([&](taskflow::TaskflowTaskOp task) {
    task->removeAttr(kSourceIterationDomainAttr);
    task->removeAttr(kSourceIterationControlBindingAttr);
    task->removeAttr(kSourceIterationSourceControlBindingAttr);
    task->removeAttr(kSourceIterationPartitionProofAttr);
    task->removeAttr(kSourceIterationCapturePendingAttr);
  });
}

static bool verifySourceState(func::FuncOp canonical, func::FuncOp candidate,
                              std::string &error) {
  return succeeded(
      verifySourceIterationDomainPartition(canonical, candidate, error));
}

static bool checkCumulativeCap(ModuleOp module, StringRef functionName,
                               unsigned cap, std::string &error) {
  auto function = selectTaskFunction(module, functionName, error);
  if (failed(function))
    return false;
  bool valid = true;
  function->walk([&](taskflow::TaskflowTaskOp task) {
    if (!valid)
      return;
    Attribute rawLedger = task->getAttr(kPartitionLineage);
    if (!rawLedger)
      return;
    auto ledger = dyn_cast<ArrayAttr>(rawLedger);
    if (!ledger) {
      error = "malformed partition lineage on task " + task.getTaskName().str();
      valid = false;
      return;
    }
    for (Attribute rootAttr : ledger) {
      auto root = dyn_cast<DictionaryAttr>(rootAttr);
      auto rootName = root ? root.getAs<StringAttr>("root") : StringAttr{};
      auto steps = root ? root.getAs<ArrayAttr>("steps") : ArrayAttr{};
      if (!rootName || rootName.getValue().empty() || !steps) {
        error = "malformed source partition lineage on task " +
                task.getTaskName().str();
        valid = false;
        return;
      }
      std::map<std::string, uint64_t> products;
      for (Attribute stepAttr : steps) {
        auto step = dyn_cast<DictionaryAttr>(stepAttr);
        auto family = step ? step.getAs<StringAttr>("family") : StringAttr{};
        auto factor = step ? step.getAs<IntegerAttr>("factor") : IntegerAttr{};
        if (!family || !factor ||
            (family.getValue() != "tiling" && family.getValue() != "replica") ||
            factor.getInt() < 2 || factor.getInt() > 8) {
          error = "malformed source partition step on task " +
                  task.getTaskName().str();
          valid = false;
          return;
        }
        uint64_t &product = products[family.getValue().str()];
        if (!product)
          product = 1;
        uint64_t selectedFactor = static_cast<uint64_t>(factor.getInt());
        if (selectedFactor > cap || product > cap / selectedFactor) {
          error = "cumulative_original_" + family.getValue().str() +
                  "_factor_exceeds_cap:task=" + task.getTaskName().str() +
                  ":root=" + rootName.getValue().str();
          valid = false;
          return;
        }
        product *= selectedFactor;
      }
    }
  });
  return valid;
}

static json::Array shapeFacts(ArrayRef<NeighborhoodShape> shapes) {
  json::Array facts;
  for (const NeighborhoodShape &shape : shapes)
    facts.push_back(json::Object{
        {"task", shape.task}, {"rows", shape.rows}, {"cols", shape.cols}});
  return facts;
}

static bool sourceFacts(func::FuncOp function, json::Array &facts,
                        std::string &error) {
  bool valid = true;
  function.walk([&](taskflow::TaskflowTaskOp task) {
    if (!valid)
      return;
    FailureOr<SourceIterationDomainInfo> info =
        parseSourceIterationDomain(task, error);
    if (failed(info)) {
      valid = false;
      return;
    }
    json::Array axes;
    for (const SourceIterationAxis &axis : info->axes)
      axes.push_back(json::Object{
          {"kind", axis.representedByTaskflow ? "taskflow-counter"
                                              : "internal-carried-loop"},
          {"expanded_inside_mapper_firing", axis.expandedInsideMapperFiring},
          {"ordinal", static_cast<int64_t>(axis.ordinal)},
          {"lower", axis.lower},
          {"upper", axis.upper},
          {"step", axis.step},
          {"extent", axis.extent},
          {"parent_counter_ordinal", axis.parentCounterOrdinal},
          {"carried_values", axis.carriedValues},
          {"result_uses", axis.resultUses}});
    std::string currentDomainError;
    FailureOr<TaskIterationDomainMetadata> currentDomain =
        resolveTaskIterationDomain(task, currentDomainError);
    json::Value currentFiringCount(nullptr);
    json::Value currentSourceWorkCount(nullptr);
    std::string currentDomainStatus = "unsupported-or-unproven";
    std::string currentDomainReason = currentDomainError;
    if (succeeded(currentDomain)) {
      currentDomainStatus = currentDomain->status;
      currentDomainReason = currentDomain->reason;
      if (currentDomain->countKnown) {
        currentFiringCount =
            json::Value(currentDomain->effectiveMapperFiringCount);
        currentSourceWorkCount =
            json::Value(currentDomain->sourceIterationWorkCount);
      }
    }

    int64_t requestedAxis = -1;
    auto outputAxis =
        task->getAttrOfType<IntegerAttr>("amoeba.replica.output_shard_axis");
    auto counterAxis =
        task->getAttrOfType<IntegerAttr>("amoeba.replica.shard_axis");
    bool invalidAxisMetadata =
        (task->hasAttr("amoeba.replica.output_shard_axis") && !outputAxis) ||
        (task->hasAttr("amoeba.replica.shard_axis") && !counterAxis) ||
        (outputAxis && !counterAxis);
    if (outputAxis)
      requestedAxis = outputAxis.getInt();
    else if (counterAxis)
      requestedAxis = counterAxis.getInt();
    ReplicaOutputCoordinateProof coordinateProof =
        analyzeReplicaOutputCoordinates(task, requestedAxis);
    if (invalidAxisMetadata) {
      coordinateProof.proven = false;
      coordinateProof.selectedAxisIndependent = false;
      coordinateProof.reason = "replica shard-axis metadata is incomplete";
    }
    if (coordinateProof.proven && outputAxis && counterAxis &&
        (requestedAxis < 0 ||
         requestedAxis >=
             static_cast<int64_t>(coordinateProof.outputCounterAxes.size()) ||
         coordinateProof.outputCounterAxes[requestedAxis] ==
             ReplicaOutputCoordinateProof::kConstantAxis ||
         coordinateProof.outputCounterAxes[requestedAxis] !=
             static_cast<unsigned>(counterAxis.getInt()))) {
      coordinateProof.proven = false;
      coordinateProof.selectedAxisIndependent = false;
      coordinateProof.reason =
          "replica output shard axis does not map to its counter axis";
    }
    json::Value currentCounterAxes(nullptr);
    if (!coordinateProof.taskLowers.empty() &&
        coordinateProof.taskLowers.size() ==
            coordinateProof.taskUppers.size()) {
      json::Array counterAxes;
      for (auto [ordinal, lower] :
           llvm::enumerate(coordinateProof.taskLowers)) {
        json::Object counterFact{{"ordinal", static_cast<int64_t>(ordinal)},
                                 {"lower", lower},
                                 {"upper", coordinateProof.taskUppers[ordinal]},
                                 {"step", 1},
                                 {"bound_status", "proven"}};
        if (coordinateProof.proven) {
          json::Array outputAxes;
          for (auto [outputOrdinal, mappedCounter] :
               llvm::enumerate(coordinateProof.outputCounterAxes))
            if (mappedCounter == ordinal)
              outputAxes.push_back(static_cast<int64_t>(outputOrdinal));
          counterFact["output_axes"] = std::move(outputAxes);
        } else {
          counterFact["output_axes"] = nullptr;
        }
        counterAxes.push_back(std::move(counterFact));
      }
      currentCounterAxes = std::move(counterAxes);
    }
    auto stringAttr = [&](StringRef name) -> std::string {
      auto attr = task->getAttrOfType<StringAttr>(name);
      return attr ? attr.getValue().str() : std::string();
    };
    facts.push_back(json::Object{
        {"task", task.getTaskName().str()},
        {"complete", info->complete},
        {"represented_multiplicity", info->representedMultiplicity},
        {"internal_multiplicity", info->internalMultiplicity},
        {"source_multiplicity", info->sourceMultiplicity},
        {"canonical_witness", sourceIterationDomainCanonicalWitness(*info)},
        {"source_control_binding",
         stringAttr(kSourceIterationSourceControlBindingAttr)},
        {"current_control_binding",
         stringAttr(kSourceIterationControlBindingAttr)},
        {"partition_proof", stringAttr(kSourceIterationPartitionProofAttr)},
        {"current_domain_status", currentDomainStatus},
        {"current_domain_reason", currentDomainReason},
        {"current_taskflow_firing_count", std::move(currentFiringCount)},
        {"current_source_work_count", std::move(currentSourceWorkCount)},
        {"current_output_coordinate_status",
         coordinateProof.proven ? "proven" : "unknown"},
        {"current_output_coordinate_reason", coordinateProof.reason},
        {"current_counter_axes", std::move(currentCounterAxes)},
        {"axes", std::move(axes)}});
  });
  return valid;
}

static bool writeText(StringRef path, StringRef text, std::string &error) {
  return writeAtomically(
      path,
      [&](llvm::raw_ostream &stream) {
        stream.write(text.data(), text.size());
        return true;
      },
      error);
}

static bool publishReplay(StringRef outputDir, StringRef canonicalBytes,
                          StringRef candidateBytes, StringRef actionBytes,
                          StringRef preparedSourceBytes,
                          StringRef expectedCandidateBytes,
                          StringRef candidateText, StringRef factsText,
                          std::string &error) {
  if (llvm::sys::fs::exists(outputDir)) {
    error = "replay output directory already exists: " + outputDir.str();
    return false;
  }
  SmallString<256> parent(llvm::sys::path::parent_path(outputDir));
  if (parent.empty())
    parent = ".";
  std::error_code ec = llvm::sys::fs::create_directories(parent);
  if (ec) {
    error = "cannot create replay output parent: " + ec.message();
    return false;
  }
  SmallString<256> prefix(parent);
  llvm::sys::path::append(
      prefix,
      (Twine(".") + llvm::sys::path::filename(outputDir) + ".partial").str());
  SmallString<256> temporary;
  ec = llvm::sys::fs::createUniqueDirectory(prefix, temporary);
  if (ec) {
    error = "cannot create replay staging directory: " + ec.message();
    return false;
  }
  auto cleanup = [&] { llvm::sys::fs::remove_directories(temporary); };
  SmallString<256> path(temporary);
  llvm::sys::path::append(path, "candidate.mlir");
  if (!writeText(path, candidateText, error)) {
    cleanup();
    return false;
  }
  path = temporary;
  llvm::sys::path::append(path, "source-facts.json");
  if (!writeText(path, factsText, error)) {
    cleanup();
    return false;
  }
  path = temporary;
  llvm::sys::path::append(path, "canonical-input.mlir");
  if (!writeText(path, canonicalBytes, error)) {
    cleanup();
    return false;
  }
  path = temporary;
  llvm::sys::path::append(path, "candidate-input.mlir");
  if (!writeText(path, candidateBytes, error)) {
    cleanup();
    return false;
  }
  path = temporary;
  llvm::sys::path::append(path, "actions.json");
  if (!writeText(path, actionBytes, error)) {
    cleanup();
    return false;
  }
  if (!preparedSourceBytes.empty()) {
    path = temporary;
    llvm::sys::path::append(path, "prepared-source.mlir");
    if (!writeText(path, preparedSourceBytes, error)) {
      cleanup();
      return false;
    }
  }
  if (!expectedCandidateBytes.empty()) {
    path = temporary;
    llvm::sys::path::append(path, "expected-candidate.mlir");
    if (!writeText(path, expectedCandidateBytes, error)) {
      cleanup();
      return false;
    }
  }
  ec = llvm::sys::fs::rename(temporary, outputDir);
  if (ec) {
    error = "cannot publish complete replay output directory: " + ec.message();
    cleanup();
    return false;
  }
  return true;
}

class ReplayJointNeighborhoodActionsPass
    : public PassWrapper<ReplayJointNeighborhoodActionsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ReplayJointNeighborhoodActionsPass)

  ReplayJointNeighborhoodActionsPass() = default;
  ReplayJointNeighborhoodActionsPass(
      const ReplayJointNeighborhoodActionsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "replay-joint-neighborhood-actions";
  }
  StringRef getDescription() const override {
    return "Replay one source-proved typed neighborhood action path";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, taskflow::TaskflowDialect>();
    registerTaskflowFissionCanonicalLoweringDependencies(registry);
    mlir::amoeba::neura::createMaterializeJointTaskReplicasPass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraJointRewritePass()
        ->getDependentDialects(registry);
    mlir::amoeba::neura::createMaterializeNeuraKReductionPass()
        ->getDependentDialects(registry);
  }

  Option<std::string> actionFile{*this, "action-file", llvm::cl::init("")};
  Option<std::string> canonicalInput{*this, "canonical-input",
                                     llvm::cl::init("")};
  Option<std::string> candidateInput{*this, "candidate-input",
                                     llvm::cl::init("")};
  Option<std::string> functionName{*this, "function", llvm::cl::init("")};
  Option<std::string> stage{*this, "stage", llvm::cl::init("full-joint")};
  Option<int64_t> maxPartitionFactor{*this, "max-partition-factor",
                                     llvm::cl::init(8)};
  Option<std::string> preparedSourceFile{*this, "prepared-source-file",
                                         llvm::cl::init("")};
  Option<int64_t> maxFissionActionsPerTask{
      *this, "max-fission-actions-per-task", llvm::cl::init(64)};
  Option<std::string> expectedCandidateFile{
      *this, "expected-candidate-file", llvm::cl::init("")};
  Option<std::string> activeTransferArgumentsText{
      *this, "active-transfer-arguments", llvm::cl::init("")};
  Option<bool> activeTransferRequireProven{
      *this, "active-transfer-require-proven", llvm::cl::init(false)};
  Option<std::string> outputDir{*this, "output-dir", llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp inputModule = getOperation();
    auto fail = [&](StringRef message) {
      inputModule.emitError() << message;
      signalPassFailure();
    };
    if (actionFile.empty() || canonicalInput.empty() ||
        candidateInput.empty() || functionName.empty() || outputDir.empty()) {
      fail("replay-joint-neighborhood-actions requires action-file, "
           "canonical-input, candidate-input, function, and output-dir");
      return;
    }
    if (maxPartitionFactor != 1 && maxPartitionFactor != 2 &&
        maxPartitionFactor != 4 && maxPartitionFactor != 8) {
      fail("max-partition-factor must be one of {1, 2, 4, 8}");
      return;
    }
    if (maxFissionActionsPerTask <= 0) {
      fail("max-fission-actions-per-task must be positive");
      return;
    }
    std::string actionBytes;
    std::string error;
    if (!readFile(actionFile, actionBytes, error)) {
      fail(error);
      return;
    }
    auto parsed = json::parse(actionBytes);
    if (!parsed) {
      fail("action file is not JSON: " + llvm::toString(parsed.takeError()));
      return;
    }
    const json::Object *root = parsed->getAsObject();
    if (!root) {
      fail("action file root must be a JSON object");
      return;
    }
    SmallVector<unsigned> activeArguments;
    const json::Array *boundActiveArguments =
        root->getArray("activeTransferArguments");
    auto boundRequireProven = root->getBoolean("activeTransferRequireProven");
    if (!activeTransferArgumentsText.empty()) {
      if (stageNumber(stage) != 6 || !boundActiveArguments ||
          !boundRequireProven ||
          *boundRequireProven != activeTransferRequireProven) {
        fail("active-transfer replay requires exact stage-6 action-file binding");
        return;
      }
      SmallVector<StringRef> tokens;
      StringRef(activeTransferArgumentsText.getValue()).split(tokens, ',');
      for (StringRef token : tokens) {
        unsigned index = 0;
        if (token.empty() || !llvm::all_of(token, llvm::isDigit) ||
            token.getAsInteger(10, index) ||
            (!activeArguments.empty() && index <= activeArguments.back())) {
          fail("active-transfer-arguments must be strictly increasing unsigned indices");
          return;
        }
        activeArguments.push_back(index);
      }
      if (boundActiveArguments->size() != activeArguments.size()) {
        fail("active-transfer action-file argument count differs from CLI");
        return;
      }
      for (auto [index, value] : llvm::enumerate(*boundActiveArguments)) {
        auto argument = value.getAsInteger();
        if (!argument || *argument < 0 ||
            static_cast<uint64_t>(*argument) != activeArguments[index]) {
          fail("active-transfer action-file indices differ from CLI");
          return;
        }
      }
    } else if (root->get("activeTransferArguments") ||
               root->get("activeTransferRequireProven") ||
               activeTransferRequireProven) {
      fail("active-transfer action-file binding requires nonempty CLI arguments");
      return;
    }
    ReplayInputs inputs;
    inputs.actionBytes = std::move(actionBytes);
    OwningOpRef<ModuleOp> canonical;
    OwningOpRef<ModuleOp> preparedSource;
    if (!readInputs(*root, canonicalInput, candidateInput, functionName, stage,
                    static_cast<unsigned>(maxPartitionFactor),
                    preparedSourceFile,
                    static_cast<uint64_t>(maxFissionActionsPerTask.getValue()),
                    inputModule.getContext(), inputModule, inputs, canonical,
                    preparedSource,
                    error)) {
      fail(error);
      return;
    }
    std::string expectedCandidateBytes;
    OwningOpRef<ModuleOp> expectedCandidate;
    bool expectedCandidateMatched = false;
    if (!expectedCandidateFile.empty()) {
      if (stageNumber(inputs.stage) != 6) {
        fail("expected-candidate-file is valid only for full-joint-fission replay");
        return;
      }
      if (!readFile(expectedCandidateFile, expectedCandidateBytes, error)) {
        fail(error);
        return;
      }
      if (!parseBoundModule(expectedCandidateBytes, inputModule.getContext(),
                            "expected candidate", expectedCandidate, error)) {
        fail(error);
        return;
      }
    }
    if (llvm::sys::fs::exists(outputDir)) {
      fail("replay output directory already exists: " + outputDir.getValue());
      return;
    }

    auto canonicalFunction =
        selectTaskFunction(*canonical, functionName, error);
    auto currentFunction = selectTaskFunction(inputModule, functionName, error);
    if (failed(canonicalFunction) || failed(currentFunction)) {
      fail(error.empty()
               ? "cannot select canonical or current Taskflow function"
               : error);
      return;
    }
    if (!verifySourceState(*canonicalFunction, *currentFunction, error)) {
      fail("source_iteration_domain_previous_candidate_unproven:" + error);
      return;
    }
    if (!checkCumulativeCap(inputModule, functionName,
                            static_cast<unsigned>(maxPartitionFactor), error)) {
      fail("cumulative_partition_cap_rejected_before_replay:" + error);
      return;
    }
    if (inputs.stage.empty() || stageNumber(inputs.stage) == 0) {
      fail("unknown cumulative ablation stage");
      return;
    }
    json::Array stepFacts;
    const unsigned apiCap = inputs.maxPartitionFactor <= 4 ? 4 : 8;
    OwningOpRef<ModuleOp> effectiveCanonical = canonical->clone();
    OwningOpRef<ModuleOp> working = inputModule.clone();
    std::vector<NeighborhoodShape> shapes = inputs.initialShapes;
    size_t fissionActionCount = 0;
    while (fissionActionCount < inputs.actions.size() &&
           inputs.actions[fissionActionCount].family == "fission")
      ++fissionActionCount;
    if (fissionActionCount != 0) {
      SmallVector<mlir::amoeba::neura::TaskflowFissionAction> sourceActions;
      for (size_t index = 0; index < fissionActionCount; ++index) {
        const NeighborhoodAction &action = inputs.actions[index];
        const NeighborhoodPrimitive &primitive = action.primitives.front();
        mlir::amoeba::neura::TaskflowFissionAction sourceAction;
        sourceAction.function = functionName.getValue();
        sourceAction.task = primitive.firstTask;
        sourceAction.leftNodes.append(primitive.leftNodes.begin(),
                                      primitive.leftNodes.end());
        sourceActions.push_back(std::move(sourceAction));
        stepFacts.push_back(json::Object{
            {"index", static_cast<int64_t>(index)},
            {"label", action.label}, {"family", action.family},
            {"status", "exact-source-replay-verified"},
            {"counter_domain_policy",
             "retained-per-operation-not-disjoint-firing-partitions"}});
      }
      OwningOpRef<ModuleOp> fissionedSource = preparedSource->clone();
      for (const auto &action : sourceActions) {
        if (failed(mlir::amoeba::neura::materializeTaskflowFission(
                fissionedSource.get(), action.function, action.task,
                action.leftNodes, error))) {
          fail("typed source fission materialization failed:" + error);
          return;
        }
      }
      if (!fissionedSource ||
          failed(mlir::amoeba::neura::verifyTaskflowFissionReplay(
              preparedSource.get(), fissionedSource.get(), sourceActions,
              error))) {
        fail("typed source fission replay failed:" + error);
        return;
      }
      if (failed(lowerPreparedTaskflowFissionSource(
              fissionedSource.get(), effectiveCanonical, error))) {
        fail("fixed canonical lowering of fissioned source failed:" + error);
        return;
      }
      working = effectiveCanonical->clone();
      auto splitFunction = selectTaskFunction(*working, functionName, error);
      if (failed(splitFunction)) {
        fail("fissioned canonical Taskflow function is missing:" + error);
        return;
      }
      shapes.clear();
      splitFunction->walk([&](taskflow::TaskflowTaskOp task) {
        std::string name = task.getTaskName().str();
        NeighborhoodShape shape{name, 1, 1};
        auto requested = llvm::find_if(
            inputs.requestedInitialShapes,
            [&](const NeighborhoodShape &candidate) {
              return candidate.task == name;
            });
        if (requested != inputs.requestedInitialShapes.end())
          shape = *requested;
        shapes.push_back(std::move(shape));
      });
      if (inputs.hasRequestedInitialShapes &&
          inputs.requestedInitialShapes.size() != shapes.size()) {
        fail("initialShapes does not bind exactly the fissioned canonical tasks");
        return;
      }
      auto splitCanonicalFunction =
          selectTaskFunction(*effectiveCanonical, functionName, error);
      auto splitCurrentFunction =
          selectTaskFunction(*working, functionName, error);
      if (failed(splitCanonicalFunction) || failed(splitCurrentFunction) ||
          !verifySourceState(*splitCanonicalFunction, *splitCurrentFunction,
                             error)) {
        fail("fissioned canonical source-domain proof failed:" + error);
        return;
      }
      if (!checkCumulativeCap(*working, functionName,
                              inputs.maxPartitionFactor, error)) {
        fail("fissioned canonical cumulative partition cap rejected:" + error);
        return;
      }
    }
    std::vector<NeighborhoodShape> initialShapes = shapes;
    json::Array initialShapeFacts = shapeFacts(initialShapes);
    auto prepareActiveTransfers = [&](ModuleOp module, bool trustedRewrite) {
      if (activeArguments.empty())
        return true;
      auto function = selectTaskFunction(module, functionName, error);
      if (failed(function))
        return false;
      for (unsigned requested : activeArguments)
        if (requested >= function->getNumArguments() ||
            !isa<MemRefType>(function->getArgument(requested).getType())) {
          error = "active-transfer replay argument is out of range or not a memref";
          return false;
        }
      for (unsigned index = 0; index < function->getNumArguments(); ++index) {
        if (!isa<MemRefType>(function->getArgument(index).getType()))
          continue;
        bool requested = llvm::is_contained(activeArguments, index);
        bool supplied = hasStaticActiveTransferShapeFacts(*function, index);
        if (!requested && !supplied)
          continue;
        if (trustedRewrite || !supplied) {
          if (failed(rederiveAndStoreStaticActiveTransferShapeProof(
                  *function, index, &error)))
            return false;
        } else if (failed(verifyStaticActiveTransferShapeProof(
                       *function, index, &error))) {
          return false;
        }
        if (failed(verifyStaticActiveTransferShapeProof(*function, index, &error)))
          return false;
        if (requested && activeTransferRequireProven &&
            !analyzeStaticActiveTransferShape(*function, index).proven) {
          error = "required active-transfer proof is unknown after replay";
          return false;
        }
      }
      return true;
    };
    // Source replay owns the new split baseline. Imported seed proofs are
    // checked before use; only a trusted materialization may refresh them.
    if (!prepareActiveTransfers(effectiveCanonical.get(), false) ||
        !prepareActiveTransfers(working.get(), false)) {
      fail("active_transfer_seed_replay_unproven:" + error);
      return;
    }
    auto activeCanonicalFunction =
        selectTaskFunction(*effectiveCanonical, functionName, error);
    if (failed(activeCanonicalFunction)) {
      fail("cannot select effective canonical task function:" + error);
      return;
    }
    for (size_t index = fissionActionCount; index < inputs.actions.size(); ++index) {
      const NeighborhoodAction &action = inputs.actions[index];
      auto current = selectTaskFunction(*working, functionName, error);
      if (failed(current) ||
          !verifySourceState(*activeCanonicalFunction, *current, error)) {
        fail("source_iteration_domain_action_predecessor_unproven:" + error);
        return;
      }
      std::string enumerationError;
      std::vector<NeighborhoodAction> available = enumerateNeighborhoodActions(
          *working, functionName, shapes, stage, /*round=*/1, apiCap,
          preparedSource.get(), inputs.maxFissionActionsPerTask,
          &enumerationError, nullptr, inputs.canonicalFissionActions);
      if (!enumerationError.empty()) {
        fail("fission action enumeration failed:" + enumerationError);
        return;
      }
      if (!llvm::any_of(available, [&](const NeighborhoodAction &candidate) {
            return actionEqual(action, candidate);
          })) {
        fail("action_not_enumerated_for_current_source_graph:" + action.label);
        return;
      }

      OwningOpRef<ModuleOp> trial = working->clone();
      OwningOpRef<ModuleOp> actionCanonical = effectiveCanonical->clone();
      suspendSourceCertificates(*trial);
      suspendSourceCertificates(*actionCanonical);
      std::vector<NeighborhoodShape> nextShapes = shapes;
      std::string reason;
      std::string diagnostic;
      if (!applyNeighborhoodAction(*trial, *actionCanonical, functionName,
                                   action, nextShapes, reason, diagnostic,
                                   apiCap)) {
        std::string message =
            "action_application_failed:" + action.label + ":" + reason;
        if (!diagnostic.empty())
          message += ":" + diagnostic;
        fail(message);
        return;
      }
      current = selectTaskFunction(*trial, functionName, error);
      if (failed(current) ||
          failed(proveAndRefreshSourceIterationDomainPartition(
              *activeCanonicalFunction, *current, error)) ||
          failed(verifySourceIterationDomainPartition(*activeCanonicalFunction,
                                                      *current, error))) {
        fail("source_iteration_domain_action_proof_failed:" + action.label +
             ":" + error);
        return;
      }
      if (failed(verify(trial->getOperation()))) {
        fail("action_output_verifier_failed:" + action.label);
        return;
      }
      if (!prepareActiveTransfers(trial.get(), true)) {
        fail("active_transfer_action_replay_unproven:" + action.label + ":" + error);
        return;
      }
      if (!checkCumulativeCap(*trial, functionName, inputs.maxPartitionFactor,
                              error)) {
        fail("cumulative_partition_cap_rejected:" + action.label + ":" + error);
        return;
      }
      size_t taskCount = 0;
      current->walk([&](taskflow::TaskflowTaskOp) { ++taskCount; });
      stepFacts.push_back(
          json::Object{{"index", static_cast<int64_t>(index)},
                       {"label", action.label},
                       {"family", action.family},
                       {"status", "applied"},
                       {"task_count", static_cast<int64_t>(taskCount)},
                       {"source_partition_proof", "verified-and-refreshed"}});
      working = std::move(trial);
      shapes = std::move(nextShapes);
    }

    currentFunction = selectTaskFunction(*working, functionName, error);
    if (failed(currentFunction) ||
        !verifySourceState(*activeCanonicalFunction, *currentFunction, error)) {
      fail("source_iteration_domain_final_candidate_unproven:" + error);
      return;
    }
    if (!expectedCandidateFile.empty()) {
      std::string replayComparable =
          candidateComparisonText(working.get(), functionName, error);
      if (replayComparable.empty()) {
        fail("cannot normalize reconstructed candidate for exact comparison:" +
             error);
        return;
      }
      std::string expectedComparable =
          candidateComparisonText(expectedCandidate.get(), functionName, error);
      if (expectedComparable.empty()) {
        fail("cannot normalize expected candidate for exact comparison:" +
             error);
        return;
      }
      if (replayComparable != expectedComparable) {
        fail("expected_candidate_exact_replay_mismatch");
        return;
      }
      expectedCandidateMatched = true;
    }
    json::Array taskFacts;
    if (!sourceFacts(*currentFunction, taskFacts, error)) {
      fail("cannot extract final source-domain facts:" + error);
      return;
    }
    json::Object facts{
        {"schema", "orbit-joint-neighborhood-action-replay-facts-v1"},
        {"status", "complete"},
        {"canonical_input_path", inputs.canonicalPath},
        {"candidate_input_path", inputs.candidatePath},
        {"action_file_path", actionFile.getValue()},
        {"prepared_source_input_path", inputs.preparedSourcePath},
        {"canonical_input_witness", "canonical-input.mlir"},
        {"candidate_input_witness", "candidate-input.mlir"},
        {"prepared_source_input_witness",
         inputs.preparedSourceBytes.empty() ? "" : "prepared-source.mlir"},
        {"action_file_witness", "actions.json"},
        {"candidate_module", "candidate.mlir"},
        {"function", inputs.function},
        {"stage", inputs.stage},
        {"max_partition_factor",
         static_cast<int64_t>(inputs.maxPartitionFactor)},
        {"max_fission_actions_per_task",
         static_cast<int64_t>(inputs.maxFissionActionsPerTask)},
        {"fission_source_replay_verified", fissionActionCount != 0},
        {"actions_applied", static_cast<int64_t>(stepFacts.size())},
        {"steps", std::move(stepFacts)},
        {"initial_shapes", std::move(initialShapeFacts)},
        {"selected_shapes", shapeFacts(shapes)},
        {"source_iteration_domain_verified", true},
        {"tasks", std::move(taskFacts)},
        {"cost_catalog_read", false},
        {"prediction_run", false},
        {"native_mapping_run", false},
        {"ranking_performed", false}};
    if (!expectedCandidateFile.empty()) {
      facts["expected_candidate_path"] = expectedCandidateFile.getValue();
      facts["expected_candidate_witness"] = "expected-candidate.mlir";
      facts["expected_candidate_exact_replay_match"] =
          expectedCandidateMatched;
      facts["expected_candidate_comparison"] =
          "exact-generic-module-after-removing-only-function-graph-variant-id";
    }
    if (!activeArguments.empty()) {
      json::Array arguments;
      for (unsigned index : activeArguments)
        arguments.push_back(static_cast<int64_t>(index));
      facts["active_transfer_replay_verified"] = true;
      facts["active_transfer_arguments"] = std::move(arguments);
      facts["active_transfer_require_proven"] = activeTransferRequireProven.getValue();
    }
    // A path replayed from the exact canonical input is also a source-owned
    // seed manifest. The search still replays its history, re-proves the full
    // candidate, and charges its score to the normal budget before admitting
    // it to the archive. This record contains no predicted or measured score.
    if (inputs.candidateBytes == inputs.canonicalBytes) {
      json::Array path;
      for (const NeighborhoodAction &action : inputs.actions)
        path.push_back(action.label);
      auto actionDocument = json::parse(inputs.actionBytes);
      const auto *actionObject = actionDocument ? actionDocument->getAsObject()
                                                : nullptr;
      const auto *actions = actionObject ? actionObject->getArray("actions")
                                         : nullptr;
      if (!actions) {
        if (!actionDocument)
          llvm::consumeError(actionDocument.takeError());
        fail("cannot preserve authenticated action seed history");
        return;
      }
      SmallString<256> candidatePath(outputDir.getValue());
      llvm::sys::path::append(candidatePath, "candidate.mlir");
      facts["record_type"] = "selection";
      facts["candidate_path"] = candidatePath.str().str();
      facts["shapes"] = shapeFacts(shapes);
      facts["action_path"] = std::move(path);
      json::Array fissionActionRows, neighborhoodActionRows;
      for (const json::Value &value : *actions) {
        const json::Object *action = value.getAsObject();
        auto family = action ? action->getString("family") : std::nullopt;
        if (!family) {
          fail("authenticated action list lost its family field");
          return;
        }
        if (*family == "fission")
          fissionActionRows.push_back(value);
        else
          neighborhoodActionRows.push_back(value);
      }
      facts["action_history"] = json::Object{
          {"schema", "orbit-joint-neighborhood-typed-actions-v1"},
          {"known", true},
          {"canonicalFactKey", fissionActionCount
                                   ? "source-replay-pending"
                                   : "graph-0"},
          {"initialShapes", shapeFacts(initialShapes)},
          {"actions", std::move(neighborhoodActionRows)},
          {"fissionActions", std::move(fissionActionRows)},
          {"unknownReason", ""}};
    }
    std::string factsText = jsonText(json::Value(std::move(facts)));
    std::string candidateText = moduleText(*working);
    if (!publishReplay(outputDir, inputs.canonicalBytes, inputs.candidateBytes,
                       inputs.actionBytes, inputs.preparedSourceBytes,
                       expectedCandidateBytes, candidateText, factsText,
                       error)) {
      fail(error);
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::amoeba::neura::createReplayJointNeighborhoodActionsPass() {
  return std::make_unique<ReplayJointNeighborhoodActionsPass>();
}

#include "ProgramGraphAmoeba.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/JointScheduling/InterTaskNetwork.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProgramGraphCommunication.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "ProductionSchedulerAdapter.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace mlir::amoeba::neura::joint_scheduling {
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;
std::string json(const J &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
  return s;
}
bool write(StringRef path, StringRef text, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    error = ec.message();
    return false;
  }
  os << text;
  return true;
}
bool save(StringRef path, const J &v, std::string &error) {
  return write(path, json(v) + "\n", error);
}
std::string fileDigest(StringRef path) {
  auto b = llvm::MemoryBuffer::getFile(path);
  if (!b)
    return {};
  llvm::SHA256 h;
  h.update((*b)->getBuffer());
  return llvm::toHex(h.final(), true);
}
std::string quote(StringRef s) {
  std::string r = "\"";
  for (char c : s) {
    if (c == '\\' || c == '\"')
      r += '\\';
    r += c;
  }
  return r + '"';
}
bool dump(StringRef path, ModuleOp module, std::string &error) {
  std::string s;
  llvm::raw_string_ostream os(s);
  module.print(os);
  return write(path, s, error);
}
func::FuncOp functionOf(ModuleOp module, StringRef name) {
  return module.lookupSymbol<func::FuncOp>(name);
}
bool lower(ModuleOp module, std::string &error) {
  // Exact existing mapper-preparation passes; no replacement arithmetic body.
  const char *pipeline =
      "func.func(construct-hyperblock-from-task),classify-task-and-counter,"
      "convert-taskflow-to-neura,cse,lower-affine,convert-scf-to-cf,convert-cf-"
      "to-llvm,assign-accelerator,lower-memref-to-neura,lower-arith-to-neura,"
      "lower-builtin-to-neura,lower-llvm-to-neura,promote-input-arg-to-const,"
      "fold-constant,canonicalize-return,canonicalize-live-in,leverage-"
      "predicated-value,transform-ctrl-to-data-flow,fold-constant,insert-data-"
      "mov,bind-source-iteration-domain";
  PassManager pm(module.getContext(), ModuleOp::getOperationName(),
                 OpPassManager::Nesting::Implicit);
  llvm::raw_string_ostream errors(error);
  if (failed(parsePassPipeline(pipeline, pm, errors)) ||
      failed(pm.run(module))) {
    if (error.empty())
      error = "original generic-to-Neura pipeline failed";
    return false;
  }
  return true;
}
J placementsJSON(const ProductionScheduledResult &schedule,
                 ArrayRef<TaskShapeChoice> choices) {
  A out;
  for (const auto &p : schedule.placements)
    out.push_back(O{{"task", choices[p.task].task},
                    {"row", p.row},
                    {"col", p.col},
                    {"rows", choices[p.task].shape.rows},
                    {"cols", choices[p.task].shape.cols},
                    {"start_cycle", p.start},
                    {"end_cycle", p.end},
                    {"idle_cycles", p.idleCycles}});
  return out;
}
// Rectangular blocking changes lexical iteration order. For every storage
// written by this task, prove both source axes identify disjoint slices for
// every read and write, including reads through a different body argument.
// Constant/shifted/indirect accesses fail closed rather than assume
// independence.
bool proveTileIndependence(const GenericTask &task, std::string &error) {
  if (task.loops.size() != 2) {
    error = "Amoeba tiling independence requires two source loops";
    return false;
  }
  auto outer = dyn_cast<affine::AffineForOp>(task.loops[0].operation);
  auto inner = dyn_cast<affine::AffineForOp>(task.loops[1].operation);
  if (!outer || !inner || inner->getParentOp() != outer.getOperation() ||
      outer->getParentOp() != task.operation) {
    error = "Amoeba tiling requires a strict perfect two-loop task nest";
    return false;
  }
  for (Operation &op : outer.getBody()->getOperations())
    if (&op != inner.getOperation() && !op.hasTrait<OpTrait::IsTerminator>()) {
      error = "Amoeba tiling requires no executable operations between the "
              "tiled loops";
      return false;
    }
  for (Operation &op : task.operation->getRegion(0).front().getOperations())
    if (&op != outer.getOperation() && !op.hasTrait<OpTrait::IsTerminator>() &&
        op.getName().getStringRef() != "arith.constant") {
      error = "Amoeba tiling permits only constant invariants outside the "
              "perfect loop nest";
      return false;
    }
  std::set<size_t> written;
  for (const auto &access : task.accesses) {
    if (!inner->isAncestor(access.operation)) {
      error = "Amoeba tiling requires all memory effects inside the innermost "
              "tiled loop";
      return false;
    }
    if (access.write)
      written.insert(access.storage);
  }
  for (size_t storage : written) {
    int commonAxes[2] = {-1, -1};
    for (const auto &access : task.accesses)
      if (access.storage == storage) {
        SmallVector<Value> indices;
        AffineMap map;
        if (auto op = dyn_cast<affine::AffineLoadOp>(access.operation)) {
          map = op.getAffineMap();
          indices.append(op.getMapOperands().begin(),
                         op.getMapOperands().end());
        } else if (auto op =
                       dyn_cast<affine::AffineStoreOp>(access.operation)) {
          map = op.getAffineMap();
          indices.append(op.getMapOperands().begin(),
                         op.getMapOperands().end());
        } else if (auto op = dyn_cast<memref::LoadOp>(access.operation))
          indices.append(op.getIndices().begin(), op.getIndices().end());
        else if (auto op = dyn_cast<memref::StoreOp>(access.operation))
          indices.append(op.getIndices().begin(), op.getIndices().end());
        else {
          error =
              "unsupported memory effect in Amoeba tiling independence proof";
          return false;
        }
        for (size_t loop = 0; loop < 2; ++loop) {
          int axis = -1;
          if (map) {
            for (auto result : llvm::enumerate(map.getResults()))
              if (auto dim = dyn_cast<AffineDimExpr>(result.value()))
                if (indices[dim.getPosition()] == task.loops[loop].induction)
                  axis = result.index();
          } else
            for (auto index : llvm::enumerate(indices))
              if (index.value() == task.loops[loop].induction)
                axis = index.index();
          if (axis < 0 || (commonAxes[loop] >= 0 && commonAxes[loop] != axis)) {
            error = "Amoeba rectangular source tiling lacks a disjoint "
                    "read/write slice proof";
            return false;
          }
          commonAxes[loop] = axis;
        }
        if (commonAxes[0] == commonAxes[1]) {
          error = "Amoeba tiling source axes do not identify independent "
                  "storage slices";
          return false;
        }
      }
  }
  return true;
}
bool tileSource(ModuleOp module, ArrayRef<int64_t> extents,
                std::string &error) {
  if (extents.size() != 2 || extents[0] <= 0 || extents[1] <= 0 ||
      module->hasAttr("orbit.amoeba.tile_sizes")) {
    error = "Amoeba source tiling requires two positive extents and an untiled "
            "source";
    return false;
  }
  GenericAnalysis checked;
  if (!analyzeGenericProgram(module, checked, error))
    return false;
  for (const auto &task : checked.tasks)
    if (!proveTileIndependence(task, error))
      return false;
  SmallVector<::mlir::taskflow::TaskflowTaskOp> tasks;
  module.walk(
      [&](::mlir::taskflow::TaskflowTaskOp task) { tasks.push_back(task); });
  size_t count = 0;
  bool changed = false;
  for (auto task : tasks) {
    SmallVector<affine::AffineForOp> loops;
    task.walk<WalkOrder::PreOrder>(
        [&](affine::AffineForOp loop) { loops.push_back(loop); });
    if (loops.size() != 2 || !loops[0]->isAncestor(loops[1]) ||
        task.getWillWrites().empty() ||
        task.getDoneWrites().size() != task.getWillWrites().size() ||
        !task.getDoneReads().empty() || !task.getValueOutputs().empty()) {
      error =
          "Amoeba source tiling supports a two-loop memory-output task only";
      return false;
    }
    for (auto loop : loops)
      if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound() ||
          loop.getStepAsInt() != 1 || loop.getNumIterOperands()) {
        error = "Amoeba source tiling requires constant unit-step independent "
                "loop bounds";
        return false;
      }
    ::mlir::taskflow::TaskflowTaskOp previous;
    size_t ordinal = 0;
    for (int64_t i = loops[0].getConstantLowerBound();
         i < loops[0].getConstantUpperBound(); i += extents[0])
      for (int64_t j = loops[1].getConstantLowerBound();
           j < loops[1].getConstantUpperBound(); j += extents[1]) {
        if (++count > 64) {
          error = "Amoeba source tile task expansion exceeds64";
          return false;
        }
        IRMapping map;
        auto clone = cast<::mlir::taskflow::TaskflowTaskOp>(task->clone(map));
        OpBuilder insertion(task);
        insertion.insert(clone.getOperation());
        clone->setAttr("task_name",
                       StringAttr::get(module.getContext(),
                                       task.getTaskName().str() + ".tile." +
                                           std::to_string(ordinal++)));
        if (previous)
          for (size_t w = 0; w < task.getWillWrites().size(); ++w)
            clone->setOperand(clone.getWillReads().size() + w,
                              previous.getDoneWrites()[w]);
        SmallVector<affine::AffineForOp> copied;
        clone.walk<WalkOrder::PreOrder>(
            [&](affine::AffineForOp loop) { copied.push_back(loop); });
        copied[0].setConstantLowerBound(i);
        copied[0].setConstantUpperBound(
            std::min(i + extents[0], loops[0].getConstantUpperBound()));
        copied[1].setConstantLowerBound(j);
        copied[1].setConstantUpperBound(
            std::min(j + extents[1], loops[1].getConstantUpperBound()));
        previous = clone;
      }
    changed |= ordinal > 1;
    task->replaceAllUsesWith(previous->getResults());
    task.erase();
  }
  if (!changed) {
    error = "Amoeba tiling is an identity at these extents";
    return false;
  }
  module->setAttr("orbit.amoeba.tile_sizes",
                  DenseI64ArrayAttr::get(module.getContext(), extents));
  return succeeded(verify(module));
}
J evaluate(StringRef text, const J &configuration, StringRef output,
           const ProgramGraphAmoebaOptions &options, std::string &error) {
  auto unsupported = [&]() {
    return J(O{{"status", "unsupported"}, {"reason", error}});
  };
  auto source = parseSourceString<ModuleOp>(text, options.context);
  GenericAnalysis analysis;
  if (!source || !analyzeGenericProgram(*source, analysis, error))
    return unsupported();
  ProgramMappingChoice mapping;
  if (!readNativeMappingChoice(*source, mapping, error))
    return unsupported();
  if (mapping.dataOrganization != "phase-local" || mapping.storage >= 0 ||
      mapping.reuseLoop >= 0 || mapping.panelLoop >= 0 ||
      (*source)->hasAttr("orbit.air_tile_sizes") ||
      (*source)->hasAttr("orbit.program_partition_extent")) {
    error = "Amoeba public adapter does not support AIR tiling/partition or "
            "retained/shared mapping choices";
    return unsupported();
  }
  auto impl = (*source)->getAttrOfType<StringAttr>("orbit.implementation");
  auto lanes = (*source)->getAttrOfType<IntegerAttr>("orbit.vector_lanes");
  if (((*source)->hasAttr("orbit.implementation") &&
       (!impl || impl.getValue() != "scalar")) ||
      ((*source)->hasAttr("orbit.vector_lanes") &&
       (!lanes || lanes.getInt() != 1))) {
    error = "Amoeba public adapter does not support AIR-specific "
            "implementation/lane choices";
    return unsupported();
  }
  if (analysis.function.getName() != options.function) {
    error = "Amoeba selected function differs from common source";
    return unsupported();
  }
  const auto &architecture = ::mlir::neura::getArchitecture();
  AmoebaTargetContract target;
  target.fabricRows = architecture.getMultiCgraRows();
  target.fabricColumns = architecture.getMultiCgraColumns();
  target.peRows = architecture.getPerCgraRows();
  target.peColumns = architecture.getPerCgraColumns();
  int contexts = 0;
  if (!::mlir::taskflow::getProductionContextSlots(contexts, error))
    return unsupported();
  target.contextSlots = contexts;
  target.architectureIdentity = fileDigest(options.architecture);
  StringRef networkPath = getInterTaskNetworkSpecOverridePath();
  if (networkPath.empty())
    networkPath = options.architecture;
  target.networkIdentity = fileDigest(networkPath);
  auto network = loadInterTaskNetworkSpec(options.architecture, true, error);
  if (failed(network) || !network->has_value() ||
      !network->value().validateForGrid(target.fabricRows, target.fabricColumns,
                                        error))
    return unsupported();
  if (fileDigest(::mlir::amoeba::getNeuraArchitectureSpecFile()) !=
      target.architectureIdentity) {
    error = "Amoeba command architecture differs from backend architecture "
            "contract";
    return unsupported();
  }
  const auto *cfg = configuration.getAsObject();
  const auto *backend = cfg ? cfg->getObject("amoeba") : nullptr;
  bool differential =
      backend && backend->getBoolean("legacy_differential").value_or(false);
  AmoebaTaskShape shape;
  if (backend)
    if (auto defaults = backend->getArray("default_shape")) {
      if (defaults->size() != 2 || !(*defaults)[0].getAsInteger() ||
          !(*defaults)[1].getAsInteger()) {
        error = "amoeba.default_shape needs two integer CGRA dimensions";
        return unsupported();
      }
      shape.rows = *(*defaults)[0].getAsInteger();
      shape.columns = *(*defaults)[1].getAsInteger();
    }
  if (auto selected =
          (*source)->getAttrOfType<DenseI64ArrayAttr>("orbit.amoeba.shape")) {
    if (selected.size() != 2) {
      error = "invalid materialized Amoeba shape";
      return unsupported();
    }
    shape.rows = selected[0];
    shape.columns = selected[1];
  }
  std::vector<AmoebaTaskShape> shapes(analysis.tasks.size(), shape);
  ProgramGraph graph;
  if (!buildAmoebaProgramGraph(analysis, target, shapes, graph, error) ||
      !checkAmoebaProgramGraph(analysis, target, shapes, graph, error))
    return unsupported();
  std::string graphId = programGraphIdentity(graph),
              candidate = staticFingerprint(J(text));
  if (auto ec = llvm::sys::fs::create_directories(output)) {
    error = ec.message();
    return unsupported();
  }
  if (!save(output.str() + "/target-contract.json",
            serializeAmoebaTarget(target), error))
    return unsupported();
  if (!save(output.str() + "/program-graph.json", serializeProgramGraph(graph),
            error) ||
      !write(output.str() + "/program-graph.dot", renderProgramGraphDOT(graph),
             error))
    return unsupported();
  OwningOpRef<ModuleOp> lowered = source->clone();
  auto function = functionOf(*lowered, options.function);
  Builder builder(options.context);
  function->setAttr("orbit.program_graph", builder.getStringAttr(graphId));
  SmallVector<::mlir::taskflow::TaskflowTaskOp> tasks;
  function.walk(
      [&](::mlir::taskflow::TaskflowTaskOp task) { tasks.push_back(task); });
  if (tasks.size() != analysis.tasks.size()) {
    error = "generic clone changed public source task count";
    return unsupported();
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (tasks[i].getTaskName() != analysis.tasks[i].id) {
      error = "generic clone changed source task lineage";
      return unsupported();
    }
    tasks[i]->setAttr("orbit.program_graph_task", builder.getI64IntegerAttr(i));
  }
  // The original source-domain binder must observe the final lineage metadata.
  // After conversion we verify correspondence without altering its bound body.
  if (!lower(*lowered, error))
    return unsupported();
  function = functionOf(*lowered, options.function);
  tasks.clear();
  function.walk(
      [&](::mlir::taskflow::TaskflowTaskOp task) { tasks.push_back(task); });
  if (tasks.size() != analysis.tasks.size()) {
    error = "original Neura conversion changed public source task count";
    return unsupported();
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto lineage =
        tasks[i]->getAttrOfType<IntegerAttr>("orbit.program_graph_task");
    if (tasks[i].getTaskName() != analysis.tasks[i].id || !lineage ||
        lineage.getInt() != static_cast<int64_t>(i)) {
      error = "original Neura conversion changed source task lineage";
      return unsupported();
    }
  }
  auto projected = buildTaskEdgeGraphFromProgramGraph(function, graph, error);
  if (failed(projected))
    return unsupported();
  A publicEdges;
  std::vector<int64_t> projectedTransfers, projectedOrders;
  for (auto edge : projected->getEdges()) {
    A lower, upper;
    for (auto v : edge.transfer_region_lower)
      lower.push_back(v);
    for (auto v : edge.transfer_region_upper)
      upper.push_back(v);
    publicEdges.push_back(
        O{{"producer", edge.producer.getTaskName()},
          {"consumer", edge.consumer.getTaskName()},
          {"kind", edge.kind == TaskEdgeKind::Raw ? "raw" : "control"},
          {"payload_bits", edge.payload_bits
                               ? J(static_cast<int64_t>(*edge.payload_bits))
                               : J(nullptr)},
          {"program_graph_event", edge.programGraphEvent},
          {"program_graph_order", edge.programGraphOrder},
          {"region_lower", std::move(lower)},
          {"region_upper", std::move(upper)}});
    if (edge.programGraphEvent >= 0)
      projectedTransfers.push_back(edge.programGraphEvent);
    if (edge.programGraphOrder >= 0)
      projectedOrders.push_back(edge.programGraphOrder);
  }
  if (!dump(output.str() + "/neura-source.mlir", *lowered, error))
    return unsupported();
  std::vector<TaskShapeChoice> choices;
  std::vector<TaskShapeCost> costs;
  std::vector<int64_t> durations;
  if (!predictProgramGraphTaskCosts(
          *lowered, function, options, graphId, output, shapes, target.peRows,
          target.peColumns, choices, costs, durations, error))
    return unsupported();
  A taskCosts, dispatch;
  for (size_t i = 0; i < choices.size(); ++i)
    taskCosts.push_back(O{{"task", choices[i].task},
                          {"predicted_ii", costs[i].predictedII},
                          {"startup_cycles", costs[i].startupCycles},
                          {"trip_count", choices[i].tripCount},
                          {"duration_cycles", durations[i]},
                          {"cgra_rows", choices[i].shape.rows},
                          {"cgra_cols", choices[i].shape.cols},
                          {"mapper_rows", choices[i].shape.mapperRows},
                          {"mapper_cols", choices[i].shape.mapperCols}});
  ProductionScheduledResult schedule;
  {
    ScopedProgramGraphCommunication active(&graph);
    if (!scheduleWithAmoebaProductionPass(*lowered, function, choices,
                                          durations, candidate, false, schedule,
                                          error))
      return unsupported();
  }
  if (schedule.programGraphId != graphId ||
      schedule.programGraphTransfers != projectedTransfers ||
      schedule.programGraphOrders != projectedOrders) {
    error = "production scheduler did not consume the checked public edge "
            "projection";
    return unsupported();
  }
  for (unsigned id : schedule.dispatch)
    dispatch.push_back(choices[id].task);
  J placements = placementsJSON(schedule, choices);
  O legacy{{"requested", differential}, {"status", "not_requested"}};
  if (differential) {
    ProductionScheduledResult old;
    ScopedProgramGraphCommunication inactive(nullptr);
    if (!scheduleWithAmoebaProductionPass(*lowered, function, choices,
                                          durations, candidate, false, old,
                                          error))
      return unsupported();
    bool same = old.makespan == schedule.makespan &&
                json(placementsJSON(old, choices)) == json(placements);
    legacy["status"] = same ? "identical" : "different";
    legacy["makespan_cycles"] = old.makespan;
    legacy["task_schedule"] = placementsJSON(old, choices);
    legacy["public_task_schedule"] = placements;
    legacy["production_communication_trace_mlir"] = old.communicationTraceMLIR;
    legacy["production_dependency_trace_mlir"] = old.dependencyTraceMLIR;
    if (!same && !(*source)->hasAttr("orbit.amoeba.tile_sizes")) {
      error =
          "baseline common/legacy production scheduling differential failed";
      return unsupported();
    }
    legacy["scope"] =
        "same actual lowered body/model durations/shapes; partial-region "
        "differences may correct legacy whole-value overcharge";
    if (!save(output.str() + "/legacy-differential.json", J(O(legacy)), error))
      return unsupported();
  }
  // Existing exact-score reader is the unchanged mapper replay protocol.
  std::string scoreFile = output.str() + "/mapper-score.jsonl";
  J header = O{{"record_type", "header"},
               {"schema", "amoeba-exact-joint-task-scores-v1"},
               {"function", options.function},
               {"schedule_space", "production-scheduler"},
               {"dispatch_policy", "critical-path"},
               {"start_policy", "production-spatial-temporal-scheduler"},
               {"temporal_search_scope", "production-task-scheduler-only"},
               {"production_fixed_point_max_iterations", 64}};
  J score = O{{"record_type", "score"},
              {"schema", "amoeba-exact-joint-task-scores-v1"},
              {"candidate_id", candidate},
              {"shape_candidate_id", candidate},
              {"valid", true},
              {"task_costs", J(A(taskCosts))},
              {"task_schedule", placements},
              {"dispatch_order", J(A(dispatch))}};
  J footer = O{{"record_type", "footer"},
               {"schema", "amoeba-exact-joint-task-scores-v1"},
               {"scored_count", 1}};
  if (!write(scoreFile,
             json(header) + "\n" + json(score) + "\n" + json(footer) + "\n",
             error))
    return unsupported();
  function->setAttr("joint_scheduling_candidate_id",
                    builder.getStringAttr(candidate));
  function->setAttr("joint_scheduling_candidate_scope",
                    builder.getStringAttr(kSearchScope));
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i]->setAttr("cgra_count", builder.getI32IntegerAttr(
                                        shapes[i].rows * shapes[i].columns));
    tasks[i]->setAttr(
        "cgra_shape",
        builder.getStringAttr(choices[i].shape.toCgraShapeAttrValue()));
    tasks[i]->setAttr("amoeba.joint_shape_orientation_fixed",
                      builder.getUnitAttr());
    tasks[i]->setAttr(
        "amoeba.selected_cgra_count",
        builder.getI64IntegerAttr(shapes[i].rows * shapes[i].columns));
    tasks[i]->setAttr("amoeba.selected_trip_count",
                      builder.getI64IntegerAttr(choices[i].tripCount));
    tasks[i]->setAttr(
        "amoeba.selected_cgra_shape",
        builder.getStringAttr(choices[i].shape.toCgraShapeAttrValue()));
    tasks[i]->setAttr("amoeba.selected_mapper_tile_rows",
                      builder.getI64IntegerAttr(choices[i].shape.mapperRows));
    tasks[i]->setAttr("amoeba.selected_mapper_tile_cols",
                      builder.getI64IntegerAttr(choices[i].shape.mapperCols));
  }
  auto mapper = ::mlir::amoeba::neura::createMapJointSchedulingTasksPass();
  std::string mapperOptions =
      "function=" + quote(options.function) + " scores=" + quote(scoreFile) +
      " candidate-id=" + quote(candidate) +
      " mapping-cache-dir=" + quote(output.str() + "/mapper-cache");
  if (failed(mapper->initializeOptions(mapperOptions,
                                       [&](const llvm::Twine &message) {
                                         error = message.str();
                                         return failure();
                                       })))
    return unsupported();
  {
    ScopedProgramGraphCommunication active(&graph);
    PassManager pm(options.context);
    pm.addPass(std::move(mapper));
    if (failed(pm.run(*lowered))) {
      if (error.empty())
        error = "actual original mapper replay failed";
      return unsupported();
    }
  }
  function = functionOf(*lowered, options.function);
  if (!function->hasAttr("joint_scheduling_mapper_replay_completed")) {
    error = "original mapper did not produce replay-completed evidence";
    return unsupported();
  }
  if (!dump(output.str() + "/mapped.mlir", *lowered, error))
    return unsupported();
  std::map<std::string, int64_t> compiled;
  bool validActualII = true;
  A actualII;
  function.walk([&](::mlir::taskflow::TaskflowTaskOp task) {
    auto ii = task->getAttrOfType<IntegerAttr>("compiled_ii");
    validActualII &= ii && ii.getInt() > 0;
    if (ii)
      validActualII &=
          compiled.emplace(task.getTaskName().str(), ii.getInt()).second;
    actualII.push_back(O{{"task", task.getTaskName()},
                         {"compiled_ii", ii ? J(ii.getInt()) : J(nullptr)}});
  });
  if (!validActualII || compiled.size() != choices.size()) {
    error = "actual mapper has missing/nonpositive compiled II";
    return unsupported();
  }
  std::vector<int64_t> actualDurations;
  A actualDurationFacts;
  for (size_t i = 0; i < choices.size(); ++i) {
    auto found = compiled.find(choices[i].task);
    if (found == compiled.end()) {
      error = "actual mapper task lineage differs from checked choices";
      return unsupported();
    }
    long double duration =
        static_cast<long double>(costs[i].startupCycles) +
        static_cast<long double>(found->second) *
            static_cast<long double>(choices[i].tripCount - 1);
    if (!std::isfinite(duration) || duration <= 0 ||
        duration >
            static_cast<long double>(std::numeric_limits<int64_t>::max())) {
      error = "actual compiled II/startup duration is outside scheduler range";
      return unsupported();
    }
    actualDurations.push_back(
        std::max<int64_t>(1, static_cast<int64_t>(std::ceil(duration))));
    actualDurationFacts.push_back(
        O{{"task", choices[i].task},
          {"compiled_ii", found->second},
          {"startup_cycles", costs[i].startupCycles},
          {"trip_count", choices[i].tripCount},
          {"duration_cycles", actualDurations.back()}});
  }
  ProductionScheduledResult actualSchedule;
  {
    ScopedProgramGraphCommunication active(&graph);
    if (!scheduleWithAmoebaProductionPass(*lowered, function, choices,
                                          actualDurations, candidate, false,
                                          actualSchedule, error))
      return unsupported();
  }
  if (actualSchedule.programGraphId != graphId ||
      actualSchedule.programGraphTransfers != projectedTransfers ||
      actualSchedule.programGraphOrders != projectedOrders) {
    error = "actual-duration scheduler did not consume the checked public edge "
            "projection";
    return unsupported();
  }
  J actualPlacements = placementsJSON(actualSchedule, choices);
  A actualDispatch;
  for (unsigned id : actualSchedule.dispatch)
    actualDispatch.push_back(choices[id].task);
  O actualLegacy{{"requested", differential}, {"status", "not_requested"}};
  if (differential) {
    ProductionScheduledResult old;
    ScopedProgramGraphCommunication inactive(nullptr);
    if (!scheduleWithAmoebaProductionPass(*lowered, function, choices,
                                          actualDurations, candidate, false,
                                          old, error))
      return unsupported();
    bool same = old.makespan == actualSchedule.makespan &&
                json(placementsJSON(old, choices)) == json(actualPlacements);
    actualLegacy["status"] = same ? "identical" : "different";
    actualLegacy["makespan_cycles"] = old.makespan;
    actualLegacy["task_schedule"] = placementsJSON(old, choices);
    actualLegacy["public_task_schedule"] = actualPlacements;
    actualLegacy["production_communication_trace_mlir"] =
        old.communicationTraceMLIR;
    actualLegacy["production_dependency_trace_mlir"] = old.dependencyTraceMLIR;
    actualLegacy["scope"] =
        "same actual mapped body/compiled-II durations/shapes; partial regions "
        "may correct legacy whole-value overcharge";
    if (!same && !(*source)->hasAttr("orbit.amoeba.tile_sizes")) {
      error = "actual-duration common/legacy production scheduling "
              "differential failed";
      return unsupported();
    }
  }
  J receipt =
      O{{"status", "passed"},
        {"graph_id", graphId},
        {"public_edge_projection", std::move(publicEdges)},
        {"production_graph_projection_verified", true},
        {"task_costs", std::move(taskCosts)},
        {"task_schedule", placements},
        {"dispatch_order", std::move(dispatch)},
        {"predicted_makespan_cycles", schedule.makespan},
        {"mapper_replay_completed", true},
        {"actual_ii", std::move(actualII)},
        {"legacy_differential", std::move(legacy)},
        {"external_io_cost_status",
         "unpriced; unchanged original inter-task scheduler scope"},
        {"memory_capacity_status",
         "unknown; abstract per-task availability is not physical SRAM"},
        {"schedule_policy", "dependency-order-witness graph; physical "
                            "placement/cycles checked by original backend"}};
  (*receipt.getAsObject())["target_contract"] = serializeAmoebaTarget(target);
  (*receipt.getAsObject())["actual_makespan_cycles"] = actualSchedule.makespan;
  (*receipt.getAsObject())["actual_task_schedule"] = actualPlacements;
  (*receipt.getAsObject())["actual_dispatch_order"] = std::move(actualDispatch);
  (*receipt.getAsObject())["actual_duration_facts"] =
      std::move(actualDurationFacts);
  (*receipt.getAsObject())["actual_legacy_differential"] =
      std::move(actualLegacy);
  (*receipt.getAsObject())["actual_production_communication_trace_mlir"] =
      actualSchedule.communicationTraceMLIR;
  (*receipt.getAsObject())["actual_production_dependency_trace_mlir"] =
      actualSchedule.dependencyTraceMLIR;
  (*receipt.getAsObject())["actual_schedule_scope"] =
      "unchanged production scheduler on actual mapped body; "
      "ceil(startup_cycles + compiled_ii * (source_trip_count - 1)); not "
      "execution timing";
  (*receipt.getAsObject())["resource_binding"] =
      O{{"public_resource", "intertask.network"},
        {"public_capacity_meaning", "one abstract routing provider; not a "
                                    "global mutually-exclusive DMA engine"},
        {"backend_resources",
         "actual directed links and local channels reserved and validated by "
         "the unchanged production communication oracle"},
        {"network_identity", target.networkIdentity},
        {"architecture_identity", target.architectureIdentity},
        {"physical_capacity_claim", false}};
  (*receipt.getAsObject())["production_communication_trace_mlir"] =
      schedule.communicationTraceMLIR;
  (*receipt.getAsObject())["production_dependency_trace_mlir"] =
      schedule.dependencyTraceMLIR;
  (*receipt.getAsObject())["copy_lifetime_scope"] =
      "public event-ordinal lifetimes are dependency witnesses, not backend "
      "cycle-buffer capacity proofs; actual readiness/reservations are in "
      "production traces";
  if (!save(output.str() + "/backend-receipt.json", receipt, error))
    return unsupported();
  std::string planId =
      staticFingerprint(O{{"graph_id", graphId},
                          {"predicted_cycles", schedule.makespan},
                          {"placements", placements},
                          {"actual_cycles", actualSchedule.makespan},
                          {"actual_placements", actualPlacements},
                          {"model", fileDigest(options.model)}});
  return O{{"status", "passed"},
           {"program_graph", serializeProgramGraph(graph)},
           {"program_graph_id", graphId},
           {"program_graph_dot", renderProgramGraphDOT(graph)},
           {"best_plan", O{{"plan_id", planId},
                           {"makespan_cycles", schedule.makespan},
                           {"cost_status", "conditional-original-cgra-ml"},
                           {"backend_receipt", receipt}}}};
}
} // namespace

bool configureProgramGraphAmoebaSearch(ModuleOp source, SearchOptions &search,
                                       const ProgramGraphAmoebaOptions &options,
                                       std::string &error) {
  if (options.function.empty() || options.architecture.empty() ||
      options.model.empty() || options.cache.empty() ||
      options.sourceRepository.empty() || options.sourceCommit.empty() ||
      options.architectureContract.empty()) {
    error = "Amoeba public evaluation requires original "
            "model/cache/architecture/source provenance";
    return false;
  }
  GenericAnalysis analysis;
  if (!analyzeGenericProgram(source, analysis, error))
    return false;
  llvm::raw_string_ostream stream(search.nativeSourceMLIR);
  source.print(stream);
  stream.flush();
  // Bind checkpoint bytes as well as the ensemble index. The cache is output,
  // not model identity. Conservatively include all regular model-directory
  // files.
  std::map<std::string, std::string> files;
  std::error_code ec;
  auto parent = std::filesystem::path(options.model).parent_path();
  for (std::filesystem::recursive_directory_iterator it(parent, ec), end;
       !ec && it != end; it.increment(ec))
    if (it->is_regular_file())
      files[it->path().string()] = fileDigest(it->path().string());
  if (ec) {
    error = "cannot bind model checkpoint directory: " + ec.message();
    return false;
  }
  O identities;
  for (const auto &[path, digest] : files)
    identities[path] = digest;
  search.nativeBackendIdentity = staticFingerprint(
      O{{"protocol", "amoeba-program-graph-v12"},
        {"architecture", fileDigest(options.architecture)},
        {"network", fileDigest(getInterTaskNetworkSpecOverridePath().empty()
                                   ? StringRef(options.architecture)
                                   : getInterTaskNetworkSpecOverridePath())},
        {"transfer_catalog",
         options.architectureTransferCatalog.empty()
             ? ""
             : fileDigest(options.architectureTransferCatalog)},
        {"models", std::move(identities)},
        {"contract", options.architectureContract},
        {"source_repository", options.sourceRepository},
        {"source_commit", options.sourceCommit}});
  search.nativeEvaluator = [options](StringRef text, const J &cfg,
                                     StringRef output, std::string &reason) {
    return evaluate(text, cfg, output, options, reason);
  };
  search.nativeApplyAction = [](ModuleOp module, const SearchAction &action,
                                std::string &reason) {
    if (action.backendAction == "tile")
      return tileSource(module, action.backendParameters, reason);
    if (action.backendAction == "shape" &&
        action.backendParameters.size() == 2) {
      module->setAttr("orbit.amoeba.shape",
                      DenseI64ArrayAttr::get(module.getContext(),
                                             action.backendParameters));
      return true;
    }
    reason = "unknown Amoeba public source action";
    return false;
  };
  if (!analysis.tasks.empty() && analysis.tasks.front().loops.size() == 2) {
    SearchAction tile;
    tile.name = "amoeba-source-rectangular-tiling";
    tile.backendAction = "tile";
    for (const auto &loop : analysis.tasks.front().loops)
      tile.backendParameters.push_back(
          std::max<int64_t>(1, (loop.upper - loop.lower + 1) / 2));
    search.nativeActions.push_back(tile);
  }
  const auto &architecture = ::mlir::neura::getArchitecture();
  for (auto shape : std::vector<std::pair<int64_t, int64_t>>{{1, 2}, {2, 1}})
    if (shape.first <= architecture.getMultiCgraRows() &&
        shape.second <= architecture.getMultiCgraColumns()) {
      SearchAction action;
      action.name = "amoeba-cgra-shape-" + std::to_string(shape.first) + "x" +
                    std::to_string(shape.second);
      action.backendAction = "shape";
      action.backendParameters = {shape.first, shape.second};
      search.nativeActions.push_back(action);
    }
  return true;
}
} // namespace mlir::amoeba::neura::joint_scheduling

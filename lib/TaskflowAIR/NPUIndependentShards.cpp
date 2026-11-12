#include "Taskflow/NPUIndependentShards.h"
#include "Taskflow/AIRBackend.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NPUConstraintSchedule.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramPartition.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &e, llvm::StringRef s) {
  e = "independent NPU shards: " + s.str();
  return false;
}
bool named(Operation *o, llvm::StringRef n) {
  return o->getName().getStringRef() == n;
}
Operation *unique(Operation *root, llvm::StringRef name) {
  Operation *answer = nullptr;
  bool multiple = false;
  root->walk([&](Operation *o) {
    if (named(o, name)) {
      multiple |= answer != nullptr;
      answer = o;
    }
  });
  return multiple ? nullptr : answer;
}
Block *body(Operation *o) {
  return o && o->getNumRegions() == 1 && llvm::hasSingleElement(o->getRegion(0))
             ? &o->getRegion(0).front()
             : nullptr;
}
bool restore(Operation *o, unsigned leading, ArrayRef<Type> types,
             std::string &e) {
  auto *b = body(o);
  if (!b || b->getNumArguments() != leading + types.size())
    return fail(e, "hierarchy ABI mismatch");
  for (size_t i = 0; i < types.size(); ++i)
    b->getArgument(leading + i).setType(types[i]);
  return true;
}
bool shiftDMA(Operation *herd, int64_t delta, std::string &e) {
  auto *b = body(herd);
  bool bad = false;
  herd->walk([&](Operation *op) {
    if (!named(op, "air.dma_memcpy_nd"))
      return;
    auto segments = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (!segments || segments.size() != 9) {
      bad = true;
      return;
    }
    auto lengths = segments.asArrayRef();
    SmallVector<unsigned> starts;
    unsigned total = 0;
    for (auto n : lengths) {
      if (n < 0) {
        bad = true;
        return;
      }
      starts.push_back(total);
      total += n;
    }
    if (total != op->getNumOperands() || lengths[1] != 1 || lengths[5] != 1) {
      bad = true;
      return;
    }
    unsigned external = 0;
    for (unsigned segment : {1u, 5u}) {
      auto arg = dyn_cast<BlockArgument>(op->getOperand(starts[segment]));
      if (!arg || arg.getOwner() != b || arg.getArgNumber() < 4)
        continue;
      ++external;
      const auto key =
          segment == 1 ? "static_dst_offsets" : "static_src_offsets";
      auto offsets = op->getAttrOfType<DenseI64ArrayAttr>(key);
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type || type.getRank() < 1 || !offsets ||
          offsets.size() != type.getRank()) {
        bad = true;
        return;
      }
      auto values = llvm::to_vector(offsets.asArrayRef());
      OpBuilder builder(op);
      if (values[0] == ShapedType::kDynamic) {
        if (lengths[segment + 1] < 1) {
          bad = true;
          return;
        }
        auto old = op->getOperand(starts[segment + 1]);
        auto c = builder.create<arith::ConstantIndexOp>(op->getLoc(), delta);
        auto shifted = builder.create<arith::AddIOp>(op->getLoc(), old, c);
        op->setOperand(starts[segment + 1], shifted);
      } else {
        if ((delta > 0 && values[0] > INT64_MAX - delta) ||
            (delta < 0 && values[0] < -delta)) {
          bad = true;
          return;
        }
        values[0] += delta;
        op->setAttr(key, builder.getDenseI64ArrayAttr(values));
      }
    }
    if (external != 1)
      bad = true;
  });
  return bad ? fail(e, "unsupported host DMA layout or rebase") : true;
}
std::string printModule(ModuleOp m) {
  std::string s;
  llvm::raw_string_ostream out(s);
  m.print(out, OpPrintingFlags().printGenericOpForm());
  return s;
}
} // namespace
bool orbit::taskflow::mergeIndependentAIRShards(
    ArrayRef<ModuleOp> shards, ArrayRef<NPUShardBinding> bindings,
    ArrayRef<Type> types, int64_t rows, OwningOpRef<ModuleOp> &result,
    std::string &e) {
  if (shards.empty() || shards.size() != bindings.size() || rows <= 0)
    return fail(e, "invalid shard inputs");
  std::vector<std::pair<int64_t, int64_t>> ranges;
  std::set<std::pair<int64_t, int64_t>> coordinates;
  for (auto b : bindings) {
    if (b.rowOffset < 0 || b.rowCount <= 0 || b.rowOffset > rows ||
        b.rowCount > rows - b.rowOffset || b.column < 0 || b.column > 3 ||
        b.row != 2 || !coordinates.emplace(b.column, b.row).second)
      return fail(e, "invalid/duplicate shard placement");
    ranges.emplace_back(b.rowOffset, b.rowOffset + b.rowCount);
  }
  std::sort(ranges.begin(), ranges.end());
  int64_t next = 0;
  for (auto r : ranges) {
    if (r.first != next)
      return fail(e, "shards overlap or leave a gap");
    next = r.second;
  }
  if (next != rows)
    return fail(e, "incomplete output partition");
  std::vector<func::FuncOp> functions;
  std::vector<Operation *> launches, segments;
  for (auto m : shards) {
    auto functionsInModule = m.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(functionsInModule))
      return fail(e, "one shard function required");
    auto f = *functionsInModule.begin();
    auto *l = unique(m, "air.launch"), *s = unique(m, "air.segment"),
         *h = unique(m, "air.herd");
    if (f.isExternal() || f.getNumArguments() != types.size() ||
        f.getNumResults() || !l || !s || !h || s->getParentOp() != l ||
        h->getParentOp() != s || l->getNumResults() || s->getNumResults() ||
        h->getNumResults() || !body(l) || !body(s) || !body(h))
      return fail(e, "one synchronous launch/segment/herd per shard required");
    functions.push_back(f);
    launches.push_back(l);
    segments.push_back(s);
  }
  OwningOpRef<ModuleOp> merged(cast<ModuleOp>(shards.front()->clone()));
  auto f = *merged->getOps<func::FuncOp>().begin();
  f.setType(FunctionType::get(merged->getContext(), types, {}));
  for (size_t i = 0; i < types.size(); ++i)
    f.getArgument(i).setType(types[i]);
  auto *launch = unique(*merged, "air.launch"),
       *segment = unique(*merged, "air.segment");
  if (!restore(launch, 4, types, e) || !restore(segment, 0, types, e))
    return false;
  auto *destination = body(segment);
  while (!destination->empty())
    destination->back().erase();
  OpBuilder b(destination, destination->end());
  for (size_t i = 0; i < shards.size(); ++i) {
    IRMapping mapping;
    auto *source = body(segments[i]);
    for (auto pair : llvm::zip(functions[i].getArguments(), f.getArguments()))
      mapping.map(std::get<0>(pair), std::get<1>(pair));
    for (auto pair : llvm::zip(body(launches[i])->getArguments(),
                               body(launch)->getArguments()))
      mapping.map(std::get<0>(pair), std::get<1>(pair));
    for (auto pair :
         llvm::zip(source->getArguments(), destination->getArguments()))
      mapping.map(std::get<0>(pair), std::get<1>(pair));
    for (auto &op : *source) {
      if (named(&op, "air.segment_terminator"))
        continue;
      bool bad = false;
      op.walk([&](Operation *nested) {
        for (auto value : nested->getOperands()) {
          if (mapping.contains(value))
            continue;
          if (auto arg = dyn_cast<BlockArgument>(value)) {
            auto *parent = arg.getOwner()->getParentOp();
            if (parent == &op || op.isProperAncestor(parent))
              continue;
            bad = true;
            continue;
          }
          auto *definition = value.getDefiningOp();
          if (definition == &op || op.isProperAncestor(definition) ||
              segments[i]->isProperAncestor(definition))
            continue;
          if (definition && named(definition, "arith.constant"))
            b.clone(*definition, mapping);
          else
            bad = true;
        }
      });
      if (bad)
        return fail(e, "unproved capture between independent shards");
      auto *copy = b.clone(op, mapping);
      if (!named(copy, "air.herd"))
        continue;
      if (!restore(copy, 4, types, e))
        return false;
      copy->setAttr("x_loc", b.getI32IntegerAttr(bindings[i].column));
      copy->setAttr("y_loc", b.getI32IntegerAttr(bindings[i].row));
      copy->setAttr("orbit.spatial_owner",
                    b.getStringAttr("npu.tile(" +
                                    std::to_string(bindings[i].column) + "," +
                                    std::to_string(bindings[i].row) + ")"));
      copy->setAttr("sym_name",
                    b.getStringAttr("orbit_shard_" + std::to_string(i)));
      copy->setAttr("orbit.shard", b.getI64IntegerAttr(i));
      copy->setAttr("orbit.row_offset",
                    b.getI64IntegerAttr(bindings[i].rowOffset));
      copy->setAttr("orbit.row_count",
                    b.getI64IntegerAttr(bindings[i].rowCount));
      if (!shiftDMA(copy, bindings[i].rowOffset, e))
        return false;
    }
  }
  OperationState terminator(segment->getLoc(), "air.segment_terminator");
  b.create(terminator);
  if (failed(verify(*merged)))
    return fail(e, "merged AIR failed SSA/IR verification");
  // Child plan/graph identities describe normalized local templates, not the
  // merged physical program. Preserve them only in per-shard manifest records.
  for (auto key :
       {"orbit.mapping_plan", "orbit.program_graph", "orbit.source_fingerprint",
        "orbit.graph_schedule_realization"})
    merged->getOperation()->removeAttr(key);
  merged->getOperation()->setAttr(
      "orbit.multicore_realization",
      b.getStringAttr(
          "independent row shards, one segment, disjoint static local pools"));
  result = std::move(merged);
  return true;
}
bool orbit::taskflow::lowerNPUIndependentProgram(
    ModuleOp source, const j::Value &config, int64_t count,
    OwningOpRef<ModuleOp> &output, j::Value &manifest, std::string &e) {
  auto start = std::chrono::steady_clock::now();
  GenericAnalysis original;
  if (!analyzeGenericProgram(source, original, e))
    return false;
  if (count < 1 || count > 4 || original.tasks.empty())
    return fail(e, "shards must be in [1,4]");
  auto rows = original.tasks.front().loops.empty()
                  ? 0
                  : original.tasks.front().loops.front().upper;
  if (rows < count)
    return fail(e, "more shards than independent rows");
  int64_t chunk = (rows + count - 1) / count;
  ProgramPartitionPlan partition;
  if (!planProgramPartition(original, chunk, partition, e))
    return false;
  for (auto axis : partition.storageAxes)
    if (axis != 0)
      return fail(e, "first multicore subset requires every storage "
                     "partitioned on axis zero");
  std::vector<OwningOpRef<ModuleOp>> sources, airs;
  std::vector<GenericAnalysis> analyses;
  std::vector<GenericMappingPlan> plans;
  NPUScheduleProblem problem;
  for (int64_t i = 0; i < count; ++i) {
    problem.dmaResources.push_back({2 * i, 4});
    problem.dmaResources.push_back({2 * i + 1, 4});
    problem.tiles.push_back({i, i, 2, 24576, 2 * i, 2 * i + 1, 0, 0});
  }
  j::Array sourceRecords;
  int64_t offset = 0;
  while (offset < rows) {
    auto extent = std::min(chunk, rows - offset);
    OwningOpRef<ModuleOp> child, air;
    if (!materializeProgramPartition(source, partition, extent, child, e))
      return false;
    GenericAnalysis analysis;
    if (!analyzeGenericProgram(*child, analysis, e))
      return false;
    // A single full-task group keeps internal copies on one physical shard.
    // Source-derived external ports and all local versions still pass the
    // planner.
    ProgramMappingChoice choice;
    choice.residency = "tiled-phase-local";
    choice.grouping = "tile-major";
    choice.dataOrganization = "retain-across-users";
    ProgramResidencyGroup group;
    for (size_t t = 0; t < analysis.tasks.size(); ++t)
      group.tasks.push_back(t);
    for (size_t s = 0; s < analysis.storages.size(); ++s) {
      int writers = 0, readers = 0;
      for (const auto &task : analysis.tasks) {
        bool w = false, r = false;
        for (const auto &x : task.accesses)
          if (x.storage == s) {
            w |= x.write;
            r |= x.read;
          }
        writers += w;
        readers += r;
      }
      if ((writers == 1 && readers) || (writers == 0 && readers > 1))
        group.retainedStorages.push_back(s);
    }
    if (group.tasks.size() > 1)
      choice.residencyGroups.push_back(group);
    else {
      choice.residency = "tiled-phase-local";
      choice.grouping = "task-shape-stages";
      choice.dataOrganization = "phase-local";
    }
    if (!setNativeAIRTileSizes(*child, extent,
                               analysis.tasks.front().loops.size() > 1
                                   ? analysis.tasks.front().loops[1].upper
                                   : 1,
                               1, e) ||
        !setNativeMappingChoice(*child, choice, e) ||
        !analyzeGenericProgram(*child, analysis, e))
      return false;
    GenericMappingPlan plan;
    if (!planGenericMapping(analysis, config, plan, e) ||
        !lowerGenericToAIR(analysis, plan, air, e))
      return false;
    if (plan.communication.executionGroups.size() != 1 ||
        plan.moduleWideChannelConversion)
      return fail(
          e,
          "each spatial shard must lower to one synchronous direct-DMA group");
    NPUIndependentShard shard;
    shard.id = problem.shards.size();
    shard.staticBytes = plan.residentBytes;
    shard.computeNs = 200;
    shard.inputNs = 500;
    shard.outputNs = 500;
    std::set<size_t> inputs, outputs;
    for (auto &p : plan.phases)
      for (auto &bounds : p.loops)
        shard.computeNs +=
            std::max<int64_t>(1, bounds.upper - bounds.lower) * 10;
    for (auto &transfer : plan.transfers) {
      if (transfer.direction == "host_to_l1") {
        shard.inputNs += transfer.bytes;
        inputs.insert(transfer.storage);
      } else {
        shard.outputNs += transfer.bytes;
        outputs.insert(transfer.storage);
      }
    }
    shard.inputBdCount = inputs.size();
    shard.outputBdCount = outputs.size();
    for (size_t t = 0; t < problem.tiles.size(); ++t)
      shard.candidateTileIndices.push_back(t);
    problem.shards.push_back(shard);
    sourceRecords.push_back(
        j::Object{{"row_offset", offset},
                  {"row_count", extent},
                  {"source_mlir", printModule(*child)},
                  {"source_fingerprint", analysis.bodyFingerprint},
                  {"plan", serializeGenericMapping(analysis, plan)}});
    sources.push_back(std::move(child));
    airs.push_back(std::move(air));
    analyses.push_back(std::move(analysis));
    plans.push_back(std::move(plan));
    offset += extent;
  }
  ConstraintSearchOptions options;
  options.wallSeconds = 1;
  auto solved = solveNPUConstraintSchedule(problem, options);
  if (solved.status != ConstraintSearchStatus::Optimal &&
      solved.status != ConstraintSearchStatus::Feasible)
    return fail(e, "CP-SAT returned no physical placement: " + solved.message);
  std::vector<NPUShardBinding> bindings(problem.shards.size());
  j::Array placement;
  for (auto s : solved.shards) {
    auto &b = bindings[s.shardIndex];
    auto &record = *sourceRecords[s.shardIndex].getAsObject();
    b.rowOffset = *record.getInteger("row_offset");
    b.rowCount = *record.getInteger("row_count");
    b.column = s.column;
    b.row = s.row;
    placement.push_back(j::Object{{"shard", s.shardIndex},
                                  {"column", s.column},
                                  {"row", s.row},
                                  {"input_channel", s.inputChannel},
                                  {"output_channel", s.outputChannel},
                                  {"input_start_ns", s.inputStartNs},
                                  {"compute_start_ns", s.computeStartNs},
                                  {"output_start_ns", s.outputStartNs}});
  }
  SmallVector<ModuleOp> modules;
  for (auto &m : airs)
    modules.push_back(*m);
  SmallVector<Type> types(original.function.getArgumentTypes());
  if (!mergeIndependentAIRShards(modules, bindings, types, rows, output, e))
    return false;
  output->getOperation()->setAttr(
      "orbit.source_fingerprint",
      StringAttr::get(source.getContext(), original.bodyFingerprint));
  output->getOperation()->setAttr(
      "orbit.graph_schedule_realization",
      StringAttr::get(source.getContext(),
                      "independent physical shards; per-shard normalized graph "
                      "and concrete placement in manifest"));
  manifest = j::Object{
      {"schema", "orbit.npu-independent-shards.v22"},
      {"source_fingerprint", original.bodyFingerprint},
      {"partition", serializeProgramPartition(original, partition)},
      {"shards", std::move(sourceRecords)},
      {"placement", std::move(placement)},
      {"solver_status", constraintSearchStatusName(solved.status)},
      {"solver_model_build_ms", solved.telemetry.modelBuildMs},
      {"solver_ms", solved.telemetry.solveMs},
      {"solver_variables", solved.telemetry.variables},
      {"solver_constraints", solved.telemetry.constraints},
      {"model_makespan_ns", solved.makespanNs},
      {"model_is_assumed", true},
      {"schedule_realization", "independent resources; event readiness and DMA "
                               "arbitration, no timed dispatch guarantee"},
      {"total_lower_ms", std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count()},
      {"final_aie_checked", false}};
  return true;
}

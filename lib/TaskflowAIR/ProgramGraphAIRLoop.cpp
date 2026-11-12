#include "Taskflow/ProgramGraphAIRLoop.h"
#include "Taskflow/ProgramMatrixImplementation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <set>
using namespace orbit::taskflow;
using namespace mlir;
namespace {
bool fail(std::string &error, const char *text) {
  error = std::string("selected AIR loop: ") + text;
  return false;
}
llvm::json::Array ids(ArrayRef<size_t> values) {
  llvm::json::Array result;
  for (auto value : values)
    result.push_back(int64_t(value));
  return result;
}
bool affineOffsets(const std::vector<std::vector<int64_t>> &values,
                   int64_t rows, int64_t columns,
                   ProgramGraphAIRLoopOffset &out) {
  if (values.empty() || int64_t(values.size()) != rows * columns)
    return false;
  out.base = values.front();
  out.row.assign(out.base.size(), 0);
  out.column = out.row;
  for (size_t d = 0; d < out.base.size(); ++d) {
    if (rows > 1)
      out.row[d] = values[columns][d] - out.base[d];
    if (columns > 1)
      out.column[d] = values[1][d] - out.base[d];
  }
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < columns; ++c) {
      const auto &actual = values[r * columns + c];
      if (actual.size() != out.base.size())
        return false;
      for (size_t d = 0; d < out.base.size(); ++d)
        if (actual[d] != out.base[d] + r * out.row[d] + c * out.column[d])
          return false;
    }
  return true;
}
bool sourceTranslationInvariant(const ProgramGraph &g, size_t task) {
  bool valid = true;
  g.source.tasks[task].operation->walk([&](affine::AffineForOp loop) {
    if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1)
      valid = false;
    for (auto &use : loop.getInductionVar().getUses())
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(use.getOwner()))
        valid = false;
  });
  return valid;
}
} // namespace
bool orbit::taskflow::buildProgramGraphAIRLoopPlan(const ProgramGraph &g,
                                                   ProgramGraphAIRLoopPlan &out,
                                                   std::string &error) {
  out = {};
  if (!checkProgramGraph(g, error) ||
      !checkProgramGraphSource(g.source, g, error))
    return false;
  if (g.executionGroups.empty())
    return fail(error, "no selected execution groups");
  std::map<size_t, const MappingTaskInstance *> instances;
  for (const auto &instance : g.instances) {
    if (instance.resourceCount != 1 ||
        (!instance.resourceShape.empty() &&
         instance.resourceShape != std::vector<int64_t>{1}) ||
        !instances.emplace(instance.event, &instance).second)
      return fail(error,
                  "resource replication or duplicate instance is unsupported");
    if (!sourceTranslationInvariant(g, instance.task))
      return fail(error, "source induction variable has non-address semantics");
    if (instance.implementation == programMatrixImplementation) {
      ProgramMatrixInstanceContract contract;
      if (!analyzeProgramMatrixInstance(g, instance, contract, error))
        return false;
    } else if (instance.implementation != "scalar")
      return fail(error, "unproved implementation family");
  }
  std::set<size_t> allEvents;
  std::string owner;
  llvm::json::Array witnesses;
  for (const auto &selected : g.executionGroups) {
    if (owner.empty())
      owner = selected.spatialOwner;
    if (selected.spatialOwner != owner)
      return fail(error, "loop lowering currently requires one physical owner");
    std::vector<size_t> events = selected.memberEvents;
    if (events.empty())
      return fail(error, "selected group event coverage is absent");
    llvm::sort(events);
    std::set<int64_t> rowSet, columnSet;
    for (auto event : events) {
      if (event >= g.events.size() || !allEvents.insert(event).second)
        return fail(error, "group events repeat or are invalid");
      auto found = instances.find(event);
      if (found == instances.end())
        continue;
      const auto &domain = found->second->domain;
      if (domain.size() < 2 || domain[0].loop != 0 || domain[1].loop != 1 ||
          domain[0].step != 1 || domain[1].step != 1)
        return fail(error, "requires identity M/N source domains");
      rowSet.insert(domain[0].lower);
      columnSet.insert(domain[1].lower);
    }
    if (rowSet.empty() || columnSet.empty())
      return fail(error, "group lacks source compute domains");
    ProgramGraphAIRLoopGroup group;
    group.rows = rowSet.size();
    group.columns = columnSet.size();
    int64_t repeats = group.rows * group.columns;
    if (int64_t(events.size()) % repeats)
      return fail(error, "event count does not form a complete tile grid");
    size_t period = events.size() / repeats;
    std::vector<int64_t> rows(rowSet.begin(), rowSet.end()),
        columns(columnSet.begin(), columnSet.end());
    for (int64_t iteration = 0; iteration < repeats; ++iteration)
      group.iterations.emplace_back(events.begin() + iteration * period,
                                    events.begin() + (iteration + 1) * period);
    group.events = group.iterations.front();
    for (size_t q = 0; q < period; ++q) {
      const auto &reference = g.events[group.events[q]];
      std::vector<std::vector<int64_t>> transferOffsets, domainOffsets;
      for (int64_t iteration = 0; iteration < repeats; ++iteration) {
        size_t event = group.iterations[iteration][q];
        const auto &actual = g.events[event];
        if (actual.kind != reference.kind || actual.task != reference.task ||
            actual.resource != reference.resource ||
            actual.link != reference.link ||
            actual.memory != reference.memory ||
            actual.reads.size() != reference.reads.size() ||
            actual.writes.size() != reference.writes.size())
          return fail(error,
                      "event template changes selected operation or resource");
        if (actual.kind == "transfer") {
          if (!actual.transferRegion || !reference.transferRegion ||
              actual.transferBytes != reference.transferBytes ||
              actual.transferRegion->storage !=
                  reference.transferRegion->storage ||
              actual.transferRegion->sizes != reference.transferRegion->sizes)
            return fail(error,
                        "transfer template changes selected rectangle shape");
          transferOffsets.push_back(actual.transferRegion->offsets);
        } else {
          auto found = instances.find(event),
               ref = instances.find(group.events[q]);
          if (found == instances.end() || ref == instances.end())
            return fail(error, "compute template lacks source instance");
          const auto &a = *found->second, &b = *ref->second;
          if (a.task != b.task || a.domainTask != b.domainTask ||
              a.kind != b.kind || a.implementation != b.implementation ||
              a.lanes != b.lanes || a.parallelLoop != b.parallelLoop ||
              a.domain.size() != b.domain.size())
            return fail(error, "source implementation template changes");
          std::vector<int64_t> lower;
          for (size_t d = 0; d < a.domain.size(); ++d) {
            if (a.domain[d].loop != b.domain[d].loop ||
                a.domain[d].step != b.domain[d].step ||
                a.domain[d].upper - a.domain[d].lower !=
                    b.domain[d].upper - b.domain[d].lower)
              return fail(error, "source tile extents or loop order change");
            lower.push_back(a.domain[d].lower);
          }
          if (lower[0] != rows[iteration / group.columns] ||
              lower[1] != columns[iteration % group.columns])
            return fail(error,
                        "source instances are not a complete lexical M/N grid");
          domainOffsets.push_back(std::move(lower));
        }
      }
      ProgramGraphAIRLoopOffset pattern;
      if (reference.kind == "transfer") {
        if (!affineOffsets(transferOffsets, group.rows, group.columns, pattern))
          return fail(error, "host offsets do not follow affine tile grid");
        group.hostOffsets[group.events[q]] = std::move(pattern);
      } else if (!affineOffsets(domainOffsets, group.rows, group.columns,
                                pattern))
        return fail(error, "source domains do not follow affine tile grid");
      for (bool write : {false, true}) {
        const auto &referenceCopies =
            write ? reference.writes : reference.reads;
        for (size_t c = 0; c < referenceCopies.size(); ++c) {
          auto referenceCopy = referenceCopies[c];
          const auto &copy = g.copies[referenceCopy];
          const auto &region = g.regions[g.versions[copy.version].region];
          std::vector<std::vector<int64_t>> offsets;
          for (int64_t iteration = 0; iteration < repeats; ++iteration) {
            const auto &event = g.events[group.iterations[iteration][q]];
            auto ci = (write ? event.writes : event.reads)[c];
            const auto &actualCopy = g.copies[ci];
            const auto &actualRegion =
                g.regions[g.versions[actualCopy.version].region];
            if (actualCopy.allocation != copy.allocation ||
                actualCopy.layout != copy.layout ||
                actualRegion.storage != region.storage ||
                actualRegion.sizes != region.sizes)
              return fail(error, "copy version template changes physical slot, "
                                 "layout or exact region shape");
            offsets.push_back(actualRegion.offsets);
          }
          if (!affineOffsets(offsets, group.rows, group.columns, pattern))
            return fail(error,
                        "copy version regions do not follow source tile grid");
        }
      }
    }
    llvm::json::Array iterations;
    for (const auto &iteration : group.iterations) {
      llvm::json::Array versions;
      for (auto event : iteration) {
        llvm::json::Array reads, writes;
        for (auto copy : g.events[event].reads)
          reads.push_back(llvm::json::Object{
              {"copy", int64_t(copy)},
              {"version", int64_t(g.copies[copy].version)},
              {"allocation", int64_t(g.copies[copy].allocation)}});
        for (auto copy : g.events[event].writes)
          writes.push_back(llvm::json::Object{
              {"copy", int64_t(copy)},
              {"version", int64_t(g.copies[copy].version)},
              {"allocation", int64_t(g.copies[copy].allocation)}});
        versions.push_back(llvm::json::Object{{"event", int64_t(event)},
                                              {"reads", std::move(reads)},
                                              {"writes", std::move(writes)}});
      }
      iterations.push_back(llvm::json::Object{
          {"events", ids(iteration)}, {"copy_versions", std::move(versions)}});
    }
    witnesses.push_back(
        llvm::json::Object{{"rows", group.rows},
                           {"columns", group.columns},
                           {"template_events", ids(group.events)},
                           {"iterations", std::move(iterations)}});
    out.groups.push_back(std::move(group));
  }
  if (allEvents.size() != g.events.size())
    return fail(error, "selected groups do not cover the complete graph");
  std::vector<size_t> lexical;
  for (const auto &group : out.groups)
    for (const auto &iteration : group.iterations)
      llvm::append_range(lexical, iteration);
  std::vector<size_t> position(g.events.size());
  for (size_t i = 0; i < lexical.size(); ++i)
    position[lexical[i]] = i;
  for (size_t event = 0; event < g.events.size(); ++event)
    for (auto dep : g.events[event].dependencies)
      if (position[dep] >= position[event])
        return fail(error,
                    "compacted lexical order violates a selected dependency");
  out.graphId = programGraphIdentity(g);
  out.witness = llvm::json::Object{
      {"schema", "orbit.selected-air-loop-witness.v32"},
      {"program_graph_id", out.graphId},
      {"source_body_fingerprint", g.source.bodyFingerprint},
      {"groups", std::move(witnesses)},
      {"expanded_event_count", int64_t(g.events.size())},
      {"source_translation_proven", true},
      {"scope",
       "finite selected events, exact copy versions and physical slots; "
       "dynamic host offsets only; no timing or kernel numeric certificate"}};
  return true;
}
bool orbit::taskflow::emitProgramGraphAIRLoopEvents(
    OpBuilder &b, const ProgramGraphAIRLoopPlan &plan,
    ProgramGraphAIRLoopEmitter emit, std::string &error) {
  auto loc = b.getUnknownLoc();
  for (const auto &group : plan.groups) {
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0),
         one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto rows = b.create<arith::ConstantIndexOp>(loc, group.rows),
         cols = b.create<arith::ConstantIndexOp>(loc, group.columns);
    auto row = b.create<scf::ForOp>(loc, zero, rows, one);
    row->setAttr("orbit.selected_loop_repetitions",
                 b.getI64IntegerAttr(group.rows * group.columns));
    b.setInsertionPointToStart(row.getBody());
    auto column = b.create<scf::ForOp>(loc, zero, cols, one);
    b.setInsertionPointToStart(column.getBody());
    for (auto event : group.events) {
      SmallVector<Value> offsets;
      auto pattern = group.hostOffsets.find(event);
      if (pattern != group.hostOffsets.end()) {
        const auto &p = pattern->second;
        for (size_t d = 0; d < p.base.size(); ++d) {
          Value offset = b.create<arith::ConstantIndexOp>(loc, p.base[d]);
          for (auto term :
               {std::make_pair(Value(row.getInductionVar()), p.row[d]),
                std::make_pair(Value(column.getInductionVar()), p.column[d])}) {
            if (term.second) {
              auto factor = b.create<arith::ConstantIndexOp>(loc, term.second);
              auto product = b.create<arith::MulIOp>(loc, term.first, factor);
              offset = b.create<arith::AddIOp>(loc, offset, product);
            }
          }
          offsets.push_back(offset);
        }
      }
      if (!emit(event, offsets))
        return false;
    }
    b.setInsertionPointAfter(row);
  }
  return true;
}

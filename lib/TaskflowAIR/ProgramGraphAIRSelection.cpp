#include "Taskflow/ProgramGraphAIRSelection.h"
#include "Taskflow/ProgramGraphAIR.h"
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const char *text) {
  error = std::string("selected AIR configuration: ") + text;
  return false;
}
// Supported-subset restriction after the r7 hardware counterexample: repeated
// equal-address inputs can make installed AIR combine core lock lifetimes
// across distinct output tiles. Isolate occurrences at already checked local
// copy frontiers, preserving all source events and DMA transfers. Different K
// panels remain legal. This does not establish a general core lock proof.
using InputRectangle =
    std::tuple<size_t, std::vector<int64_t>, std::vector<int64_t>, std::string,
               std::string>;
std::optional<InputRectangle> inputRectangle(const ProgramGraph &g,
                                             size_t event) {
  const auto &value = g.events[event];
  if (value.kind != "transfer")
    return {};
  const auto &source = g.allocations[g.copies[value.reads.front()].allocation];
  auto sourceMemory = llvm::find_if(g.memories, [&](const auto &memory) {
    return memory.id == source.memory;
  });
  if (sourceMemory->kind != "system-bo")
    return {};
  const auto &target = g.allocations[g.copies[value.writes.front()].allocation];
  const auto &region = *value.transferRegion;
  return InputRectangle{region.storage, region.offsets, region.sizes,
                        target.memory, target.spatialOwner};
}
bool hostOutput(const ProgramGraph &g, size_t event) {
  const auto &value = g.events[event];
  if (value.kind != "transfer")
    return false;
  const auto &target = g.allocations[g.copies[value.writes.front()].allocation];
  auto memory = llvm::find_if(
      g.memories, [&](const auto &entry) { return entry.id == target.memory; });
  return memory->kind == "system-bo";
}
bool checkTransferIsolation(const ProgramGraph &g, size_t begin, size_t end,
                            std::string &error) {
  std::set<InputRectangle> seen;
  size_t outputs = 0;
  for (size_t event = begin; event < end; ++event) {
    if (hostOutput(g, event) && ++outputs > 1)
      return fail(error, "multiple host output transfers require separate "
                         "configurations; core output lock lifetime is "
                         "not certified");
    if (auto rectangle = inputRectangle(g, event))
      if (!seen.insert(*rectangle).second)
        return fail(error, "repeated host input rectangle requires a separate "
                           "configuration; installed core lock reuse is "
                           "not certified");
  }
  return true;
}
bool frontiers(const ProgramGraph &g, std::vector<bool> &safe,
               std::string &error) {
  if (!checkProgramGraph(g, error) ||
      !checkProgramGraphSource(g.source, g, error))
    return false;
  safe.assign(g.events.size() + 1, true);
  for (size_t event = 0; event < g.events.size(); ++event)
    for (auto dep : g.events[event].dependencies)
      if (dep >= event)
        return fail(error,
                    "event indices are not a topological execution witness");
  for (const auto &copy : g.copies) {
    const auto &allocation = g.allocations[copy.allocation];
    auto memory = llvm::find_if(
        g.memories, [&](const auto &m) { return m.id == allocation.memory; });
    if (memory == g.memories.end())
      return fail(error, "copy memory is absent");
    if (memory->kind == "system-bo")
      continue;
    if (memory->kind != "local-sram" || copy.createdBy < 0)
      return fail(error, "only checked initialized local SRAM copies support "
                         "configuration cuts");
    size_t first = copy.createdBy, last = first;
    for (auto user : copy.users)
      last = std::max(last, user);
    for (auto user : copy.lastUsers)
      last = std::max(last, user);
    // Both explicit ledgers and the actual event operands must agree about
    // consumers; checking the latter prevents a cached user table hiding one.
    for (size_t event = 0; event < g.events.size(); ++event)
      if (llvm::is_contained(g.events[event].reads,
                             size_t(&copy - g.copies.data())))
        last = std::max(last, event);
    for (size_t cut = first + 1; cut <= last; ++cut)
      safe[cut] = false;
  }
  return true;
}
} // namespace
bool orbit::taskflow::checkProgramGraphAIRSelection(
    const ProgramGraph &g, const ProgramGraphAIRSelection &selection,
    std::string &error) {
  std::vector<bool> safe;
  if (!frontiers(g, safe, error))
    return false;
  if (selection.graphId != programGraphIdentity(g) ||
      selection.begin >= selection.end || selection.end > g.events.size())
    return fail(error, "selection identity or event range is invalid");
  if (!safe[selection.begin] || !safe[selection.end])
    return fail(error, "selection boundary splits a live local copy");
  return checkTransferIsolation(g, selection.begin, selection.end, error);
}
bool orbit::taskflow::checkProgramGraphAIRSelectionUnion(
    const ProgramGraph &g, llvm::ArrayRef<ProgramGraphAIRSelection> selections,
    llvm::json::Value &receipt, std::string &error) {
  std::vector<bool> safe;
  if (!frontiers(g, safe, error))
    return false;
  if (selections.empty())
    return fail(error, "empty configuration sequence");
  size_t cursor = 0, transfers = 0;
  llvm::json::Array records;
  const auto graphId = programGraphIdentity(g);
  for (size_t i = 0; i < selections.size(); ++i) {
    const auto &selection = selections[i];
    if (selection.graphId != graphId || selection.begin != cursor ||
        selection.end <= selection.begin || selection.end > g.events.size() ||
        !safe[selection.begin] || !safe[selection.end])
      return fail(error, "configuration union has a gap, overlap, reordered "
                         "range or live-copy cut");
    if (!checkTransferIsolation(g, selection.begin, selection.end, error))
      return false;
    llvm::json::Array events, inputs, outputs;
    for (size_t event = selection.begin; event < selection.end; ++event) {
      events.push_back(int64_t(event));
      if (g.events[event].kind == "transfer") {
        ++transfers;
        auto copy = g.events[event].reads.front();
        auto memory = llvm::find_if(g.memories, [&](const auto &m) {
          return m.id == g.allocations[g.copies[copy].allocation].memory;
        });
        (memory->kind == "system-bo" ? inputs : outputs)
            .push_back(int64_t(event));
      }
    }
    records.push_back(
        llvm::json::Object{{"index", int64_t(i)},
                           {"begin_event", int64_t(selection.begin)},
                           {"end_event_exclusive", int64_t(selection.end)},
                           {"events", std::move(events)},
                           {"input_transfers", std::move(inputs)},
                           {"output_transfers", std::move(outputs)}});
    cursor = selection.end;
  }
  if (cursor != g.events.size())
    return fail(error, "configuration union omits selected source events");
  receipt = llvm::json::Object{
      {"schema", "orbit.selected-air-configurations.v32"},
      {"program_graph_id", graphId},
      {"source_body_fingerprint", g.source.bodyFingerprint},
      {"configurations", std::move(records)},
      {"physical_invocation_count", int64_t(selections.size())},
      {"covered_event_count", int64_t(cursor)},
      {"covered_transfer_count", int64_t(transfers)},
      {"local_copies_crossing_boundaries", 0},
      {"additional_host_transfer_bytes", 0},
      {"runtime_requires_serial_wait2", true},
      {"input_sync_policy", "once before first invocation"},
      {"output_sync_policy", "once after last invocation"},
      {"private_intermediate_sync_to_forbidden", true},
      {"configuration_costs_charged", false},
      {"repeated_host_input_rectangles_per_configuration", false},
      {"maximum_host_output_transfers_per_configuration", 1},
      {"configuration_boundary_policy", "empty local-copy frontier, isolated "
                                        "repeated input and one host output; "
                                        "no selected event or DMA removed"},
      {"scope", "source-bound exact event/DMA union and empty-local-copy "
                "frontiers; separate ELF invocations must execute serially; no "
                "final AIE or numerical certificate"}};
  return true;
}
bool orbit::taskflow::buildProgramGraphAIRSelections(
    const ProgramGraph &g, int64_t maximumTransfers,
    std::vector<ProgramGraphAIRSelection> &selections,
    llvm::json::Value &receipt, std::string &error) {
  selections.clear();
  if (maximumTransfers < 1 || maximumTransfers > 8)
    return fail(error, "conservative static transfer budget must be in [1,8]");
  std::vector<bool> safe;
  if (!frontiers(g, safe, error))
    return false;
  const auto graphId = programGraphIdentity(g);
  for (size_t begin = 0; begin < g.events.size();) {
    size_t count = 0, outputs = 0, best = begin;
    std::set<InputRectangle> inputs;
    for (size_t end = begin + 1; end <= g.events.size(); ++end) {
      if (auto rectangle = inputRectangle(g, end - 1))
        if (!inputs.insert(*rectangle).second)
          break;
      if (hostOutput(g, end - 1) && ++outputs > 1)
        break;
      count += g.events[end - 1].kind == "transfer";
      if (count > uint64_t(maximumTransfers))
        break;
      if (safe[end])
        best = end;
    }
    if (best == begin) {
      selections.clear();
      return fail(error,
                  "no empty-local-copy frontier meets static transfer budget");
    }
    selections.push_back({graphId, begin, best});
    begin = best;
  }
  return checkProgramGraphAIRSelectionUnion(g, selections, receipt, error);
}
bool orbit::taskflow::lowerProgramGraphAIRSelection(
    const ProgramGraph &g, const ProgramGraphAIRSelection &selection,
    mlir::OwningOpRef<mlir::ModuleOp> &output, llvm::json::Value &manifest,
    std::string &error) {
  return lowerProgramGraphToAIR(g, output, manifest, error, nullptr,
                                &selection);
}

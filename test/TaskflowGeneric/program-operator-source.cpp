#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramOperatorSource.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
using namespace orbit::taskflow;
namespace j = llvm::json;
int main() {
  auto parsed = j::parse(
      R"JSON({"schema":"orbit.program-operator-source.v1","source_id":"0000000000000000000000000000000000000000000000000000000000000000","numerical_contract":"structural-fixture-not-arithmetic-proof","bindings":[{"id":"fixture","file":"fixture.cpp","sha256":"0000000000000000000000000000000000000000000000000000000000000000","symbol":"fixture"}],"values":[{"id":"x","dtype":"f32","shape":[4],"role":"input","layout":"row_major"},{"id":"raw","dtype":"f32","shape":[8],"role":"workspace","layout":"row_major","initialization":"uninitialized"},{"id":"a","dtype":"f32","shape":[4],"role":"temporary","layout":"row_major","allocation_view":{"base":"raw","offsets":[0],"strides":[1]}},{"id":"b","dtype":"f32","shape":[4],"role":"temporary","layout":"row_major","allocation_view":{"base":"raw","offsets":[4],"strides":[1]}},{"id":"sum","dtype":"f32","shape":[4],"role":"output","layout":"row_major"}],"operators":[{"id":"write-a","kind":"copy","binding":"fixture","source_ordinal":0,"inputs":["x"],"outputs":["a"],"dependencies":[],"attributes":{}},{"id":"write-b","kind":"copy","binding":"fixture","source_ordinal":1,"inputs":["x"],"outputs":["b"],"dependencies":[],"attributes":{}},{"id":"sum","kind":"add","binding":"fixture","source_ordinal":2,"inputs":["a","b"],"outputs":["sum"],"dependencies":[],"attributes":{}}]})JSON");
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return 1;
  }
  std::string error;
  unsigned checks = 0;
  auto require = [&](bool value, const char *message) {
    ++checks;
    if (!value)
      llvm::errs() << message << ": " << error << "\n";
    return value;
  };
  std::shared_ptr<ProgramOperatorSource> source;
  if (!require(parseProgramOperatorSource(*parsed, source, error),
               "workspace source parses"))
    return 1;
  StaticConfig target;
  StaticResourceDomain system;
  system.id = "system";
  system.kind = "system";
  StaticResourceDomain device;
  device.id = "device";
  device.kind = "device";
  device.parent = "system";
  target.domains = {system, device};
  StaticResource cpu;
  cpu.id = "cpu";
  cpu.kind = "compute";
  cpu.device = "cpu";
  cpu.capacity = 1;
  cpu.domain = "system";
  StaticResource npu = cpu;
  npu.id = "npu";
  npu.device = "npu";
  npu.domain = "device";
  target.resources = {cpu, npu};
  StaticMemory memory;
  memory.id = "abi";
  memory.kind = "system-bo";
  memory.domain = "system";
  memory.capacityBytes = INT64_MAX;
  memory.capacityKnown = false;
  memory.accessibleResources = {"cpu", "npu"};
  target.memories = {memory};
  ProgramOperatorSelection selection;
  for (size_t o = 0; o < 3; ++o) {
    ProgramOperatorImplementation impl;
    impl.id = "impl" + std::to_string(o);
    impl.sourceId = source->sourceId;
    impl.artifactIdentity = "fixture-artifact";
    impl.entrypoint = impl.id;
    impl.executable = true;
    impl.device = o ? "cpu" : "npu";
    impl.resource = o ? "cpu" : "npu";
    impl.memory = "abi";
    impl.operators = {o};
    impl.inputs = source->operators[o].inputs;
    impl.outputs = source->operators[o].outputs;
    impl.wallMs = .001;
    impl.launchCount = o ? 0 : 1;
    impl.visibilityPolicy = o ? "host-materialized" : "device-resident";
    selection.regions.push_back(impl);
  }
  ProgramGraph graph;
  if (!require(materializeProgramOperatorGraph(source, selection, target, "abi",
                                               graph, error),
               "actual cover materializes shared workspace"))
    return 1;
  if (!require(checkProgramGraph(graph, error),
               "public checker accepts first-class external source"))
    return 1;
  if (!require(graph.allocations.size() == 3 && graph.copies.size() == 4 &&
                   std::none_of(graph.regions.begin(), graph.regions.end(),
                                [](const auto &r) { return r.storage == 1; }),
               "uninitialized workspace has allocation and no fabricated "
               "initial copy"))
    return 1;
  if (!require(graph.regionEvents[0].kind == "submit" &&
                   graph.regionEvents[1].kind == "complete",
               "resident implementation has no fake host synchronization"))
    return 1;
  auto broken = graph;
  broken.dataEdges[0].version = broken.versions.size();
  if (!require(!checkProgramGraph(broken, error), "wrong ABI version rejected"))
    return 1;
  broken = graph;
  broken.regionEvents.push_back({"visible-to-host", 0, "npu", {}, {}});
  if (!require(!checkProgramGraph(broken, error),
               "fabricated host readback rejected"))
    return 1;
  GenericAnalysis native;
  if (!require(!checkProgramGraphSource(native, graph, error),
               "native source consumer explicitly rejects external source"))
    return 1;
  auto oldShape = source->values[0].shape;
  source->values[0].shape = {5};
  if (!require(!checkProgramOperatorSource(*source, error),
               "typed mutation cannot retain registered document identity"))
    return 1;
  source->values[0].shape = oldShape;
  auto unknown = selection;
  unknown.regions[0].wallMs.reset();
  ProgramGraph unpriced;
  if (!require(materializeProgramOperatorGraph(source, unknown, target, "abi",
                                               unpriced, error) &&
                   summarizeProgramOperatorGraph(unpriced)
                       .getAsObject()
                       ->get("estimated_ms")
                       ->getAsNull(),
               "missing local price remains unknown"))
    return 1;
  auto badPrice = selection;
  badPrice.regions[0].wallMs = std::numeric_limits<double>::quiet_NaN();
  if (!require(!materializeProgramOperatorGraph(source, badPrice, target, "abi",
                                                broken, error),
               "nonfinite implementation price rejected"))
    return 1;
  badPrice = selection;
  badPrice.regions[0].wallMs = -1;
  if (!require(!materializeProgramOperatorGraph(source, badPrice, target, "abi",
                                                broken, error),
               "negative implementation price rejected"))
    return 1;
  auto overlap = copyJSONWithoutFields(*parsed);
  auto *values = overlap.getAsObject()->getArray("values");
  *(*values)[3].getAsObject()->getObject("allocation_view")->get("offsets") =
      j::Array{2};
  std::shared_ptr<ProgramOperatorSource> invalid;
  if (!require(!parseProgramOperatorSource(overlap, invalid, error),
               "overwritten workspace slice reader rejected"))
    return 1;
  auto overflow = copyJSONWithoutFields(*parsed);
  *(*overflow.getAsObject()->getArray("values"))[2]
       .getAsObject()
       ->getObject("allocation_view")
       ->get("offsets") = j::Array{INT64_MAX};
  if (!require(!parseProgramOperatorSource(overflow, invalid, error),
               "view address overflow rejected before integer addition"))
    return 1;
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}

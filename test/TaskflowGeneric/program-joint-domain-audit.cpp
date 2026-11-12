#include "Taskflow/GenericAnalysis.h"
#include "Taskflow/ProgramJointSearch.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
constexpr int64_t maxExactNs = (int64_t{1} << 53) - 1;
const char *source = R"MLIR(
module {
  func.func @shared(
      %X: memref<1x1xf32> {orbit.role="input", orbit.name="X"},
      %Y: memref<1x1xf32> {orbit.role="output", orbit.name="Y"},
      %Z: memref<1x1xf32> {orbit.role="output", orbit.name="Z"})
      attributes {orbit.noalias, llvm.emit_c_interface} {
    %da = taskflow.task @a
        will_reads(%X : memref<1x1xf32>)
        will_writes(%Y : memref<1x1xf32>)
        [original_read_memrefs(%X : memref<1x1xf32>),
         original_write_memrefs(%Y : memref<1x1xf32>)]
        : (memref<1x1xf32>, memref<1x1xf32>) -> memref<1x1xf32> {
    ^bb0(%x: memref<1x1xf32>, %y: memref<1x1xf32>):
      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %v = affine.load %x[%i, %j] : memref<1x1xf32>
          affine.store %v, %y[%i, %j] : memref<1x1xf32>
        }
      }
      taskflow.yield done_writes(%y : memref<1x1xf32>)
    }
    %db = taskflow.task @b
        will_reads(%X : memref<1x1xf32>)
        will_writes(%Z : memref<1x1xf32>)
        [original_read_memrefs(%X : memref<1x1xf32>),
         original_write_memrefs(%Z : memref<1x1xf32>)]
        : (memref<1x1xf32>, memref<1x1xf32>) -> memref<1x1xf32> {
    ^bb0(%x: memref<1x1xf32>, %z: memref<1x1xf32>):
      affine.for %i = 0 to 1 {
        affine.for %j = 0 to 1 {
          %v = affine.load %x[%i, %j] : memref<1x1xf32>
          affine.store %v, %z[%i, %j] : memref<1x1xf32>
        }
      }
      taskflow.yield done_writes(%z : memref<1x1xf32>)
    }
    return
  }
}
)MLIR";
const char *configuration = R"JSON(
{
  "generic_air": {
    "cost_model": {
      "kind": "event-mechanism",
      "schema": "orbit.direct-air.cost.v1",
      "mode": "estimate",
      "model_id": "finite-test",
      "implementations": [
        "scalar"
      ],
      "boundary": "unit-test-device",
      "parameters": {
        "compute.startup": {
          "value": 4,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        },
        "compute.operation": {
          "value": 0,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        },
        "compute.memory_access": {
          "value": 0,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        },
        "configuration.startup": {
          "value": 0,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        },
        "host.submit": {
          "value": 0,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        },
        "host.complete": {
          "value": 0,
          "unit": "ns",
          "provenance": {
            "kind": "assumed",
            "source": "finite unit test"
          }
        }
      }
    }
  },
  "program_joint": {
    "architecture": {
      "domains": [
        {
          "id": "system",
          "kind": "system",
          "parent": ""
        },
        {
          "id": "tile0",
          "parent": "system",
          "kind": "tile",
          "coordinates": {
            "column": 0,
            "row": 2
          }
        },
        {
          "id": "tile1",
          "parent": "system",
          "kind": "tile",
          "coordinates": {
            "column": 1,
            "row": 2
          }
        }
      ],
      "resources": [
        {
          "id": "host.worker",
          "kind": "compute",
          "device": "host",
          "capacity": 1,
          "domain": "system"
        },
        {
          "id": "submit",
          "kind": "submission",
          "device": "host",
          "capacity": 1,
          "domain": "system"
        },
        {
          "id": "core0",
          "kind": "compute",
          "device": "npu",
          "capacity": 1,
          "domain": "tile0"
        },
        {
          "id": "dma0",
          "kind": "transfer",
          "device": "npu",
          "capacity": 1,
          "domain": "system"
        },
        {
          "id": "core1",
          "kind": "compute",
          "device": "npu",
          "capacity": 1,
          "domain": "tile1"
        },
        {
          "id": "dma1",
          "kind": "transfer",
          "device": "npu",
          "capacity": 1,
          "domain": "system"
        }
      ],
      "memories": [
        {
          "id": "host",
          "kind": "system-bo",
          "domain": "system",
          "capacity_bytes": 1000000,
          "accessible_resources": [
            "host.worker"
          ]
        },
        {
          "id": "mem0",
          "kind": "local-sram",
          "domain": "tile0",
          "capacity_bytes": 12,
          "accessible_resources": [
            "core0"
          ]
        },
        {
          "id": "mem1",
          "kind": "local-sram",
          "domain": "tile1",
          "capacity_bytes": 12,
          "accessible_resources": [
            "core1"
          ]
        }
      ],
      "links": [
        {
          "id": "in0",
          "source": "host",
          "target": "mem0",
          "resource": "dma0",
          "domain": "system",
          "bandwidth_bytes_per_ms": 500000,
          "latency_ms": 0,
          "provenance": {
            "kind": "architecture",
            "source": "finite test assumption; no hardware claim"
          }
        },
        {
          "id": "out0",
          "source": "mem0",
          "target": "host",
          "resource": "dma0",
          "domain": "system",
          "bandwidth_bytes_per_ms": 4000000,
          "latency_ms": 0,
          "provenance": {
            "kind": "architecture",
            "source": "finite test assumption; no hardware claim"
          }
        },
        {
          "id": "in1",
          "source": "host",
          "target": "mem1",
          "resource": "dma1",
          "domain": "system",
          "bandwidth_bytes_per_ms": 500000,
          "latency_ms": 0,
          "provenance": {
            "kind": "architecture",
            "source": "finite test assumption; no hardware claim"
          }
        },
        {
          "id": "out1",
          "source": "mem1",
          "target": "host",
          "resource": "dma1",
          "domain": "system",
          "bandwidth_bytes_per_ms": 4000000,
          "latency_ms": 0,
          "provenance": {
            "kind": "architecture",
            "source": "finite test assumption; no hardware claim"
          }
        }
      ]
    },
    "host_memory": "host",
    "chunks": [
      1
    ],
    "max_configurations": 2
  }
}
)JSON";
} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  GenericAnalysis analysis;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error)) {
    llvm::errs() << error;
    return 1;
  }
  unsigned checks = 0;
  bool ok = true;
  auto require = [&](bool passed, const char *label) {
    ++checks;
    if (!passed) {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
  };
  // Expected requires an explicit success check even for this static fixture.
  auto config = []() { return llvm::cantFail(j::parse(configuration)); };
  auto build = [&](j::Value c, ProgramJointDomain &d) {
    return buildProgramJointDomain(analysis, c, d, error);
  };
  auto architecture = [](j::Value &c) {
    return c.getAsObject()
        ->getObject("program_joint")
        ->getObject("architecture");
  };
  auto parameters = [](j::Value &c) {
    return c.getAsObject()
        ->getObject("generic_air")
        ->getObject("cost_model")
        ->getObject("parameters");
  };
  ProgramJointDomain known;
  require(build(config(), known), "build fully priced synthetic source domain");
  require(known.complete && known.costsComplete && known.unknownCosts.empty(),
          "structural and price coverage independently complete");
  auto baseline = baselineProgramJointDomain(known);
  require(baseline.hasPlan, "known serial baseline is independently checked");
  // A second known memory must not hide a selectable memory whose capacity
  // cannot be checked. This is an input-information error, not infeasibility.
  auto missing_capacity = config();
  auto *missing_memories = architecture(missing_capacity)->getArray("memories");
  (*(*missing_memories)[1].getAsObject())["capacity_known"] = false;
  (*(*missing_memories)[1].getAsObject())["capacity_bytes"] = nullptr;
  ProgramJointDomain unknown_capacity;
  require(!build(std::move(missing_capacity), unknown_capacity) &&
              error.find("insufficient target information") !=
                  std::string::npos,
          "selectable unknown local capacity is rejected explicitly");
  // An unreachable local memory cannot participate in any placement. Its
  // unknown capacity does not remove a legal alternative from the domain.
  auto unused_capacity = config();
  auto *unused_architecture = architecture(unused_capacity);
  auto *unused_memories = unused_architecture->getArray("memories");
  (*(*unused_memories)[1].getAsObject())["capacity_known"] = false;
  (*(*unused_memories)[1].getAsObject())["capacity_bytes"] = nullptr;
  auto *unused_links = unused_architecture->getArray("links");
  j::Array retained_links;
  for (size_t i = 2; i < unused_links->size(); ++i)
    retained_links.push_back(j::Object(*(*unused_links)[i].getAsObject()));
  (*unused_architecture)["links"] = std::move(retained_links);
  ProgramJointDomain unused_unknown_capacity;
  require(build(std::move(unused_capacity), unused_unknown_capacity) &&
              unused_unknown_capacity.places.size() == 1 &&
              unused_unknown_capacity.places.front().binding.memory == "mem1",
          "unknown capacity on unused memory does not reject the input");
  // Host BOs are caller-owned source storage, not an unlimited placement.
  // Preserve unknown metadata and make the local-only capacity scope explicit.
  auto unknown_host = config();
  auto *host_memory =
      architecture(unknown_host)->getArray("memories")->front().getAsObject();
  (*host_memory)["capacity_known"] = false;
  (*host_memory)["capacity_bytes"] = nullptr;
  ProgramJointDomain host_unknown_capacity;
  require(build(std::move(unknown_host), host_unknown_capacity),
          "caller-owned host BO capacity may remain unknown");
  auto host_receipt = serializeProgramJointDomain(host_unknown_capacity);
  auto *host_metadata = host_receipt.getAsObject()
                            ->getObject("architecture")
                            ->getArray("memories")
                            ->front()
                            .getAsObject();
  require(
      host_metadata->getBoolean("capacity_known") == false &&
          host_metadata->get("capacity_bytes")->getAsNull().has_value() &&
          host_receipt.getAsObject()->getString("host_capacity_scope") ==
              "caller-owned source BOs; local placement capacity only",
      "unknown host capacity is preserved without an unlimited-capacity claim");
  // The known route still gives a usable incumbent, but an unpriced parallel
  // alternative could be faster. Its existence prevents a full-domain proof.
  auto c = config();
  auto *links = architecture(c)->getArray("links");
  auto extra = j::Object(*links->front().getAsObject());
  extra["id"] = "unpriced";
  extra["bandwidth_bytes_per_ms"] = 0;
  links->push_back(std::move(extra));
  ProgramJointDomain unknown;
  require(build(std::move(c), unknown),
          "unpriced alternative preserves source domain");
  require(unknown.complete && !unknown.costsComplete &&
              !unknown.unknownCosts.empty(),
          "unknown price does not pretend generation is incomplete or route "
          "infeasible");
  int64_t payload = 0, ns = 0;
  size_t group = 0, place = 0;
  require(programJointGroupFits(unknown, group, place, payload, error),
          "group remains structurally feasible with an unpriced alternative");
  auto value = unknown.groups[group].inputs.front();
  auto link = unknown.target.links.size() - 1;
  require(programJointTransferSupported(unknown, unknown.groups[group].variant,
                                        value, link, error),
          "missing link price is structurally supported");
  require(!programJointTransferNs(unknown, unknown.groups[group].variant, value,
                                  link, ns, error),
          "missing link price is not zero duration");
  auto fallback = baselineProgramJointDomain(unknown);
  require(fallback.hasPlan, "known route remains available as a baseline");
  fallback.status = ConstraintSearchStatus::Optimal;
  auto serialized = serializeProgramJointSolution(unknown, fallback);
  require(serialized.getAsObject()->getBoolean("full_domain_optimal") ==
                  false &&
              serialized.getAsObject()->getString("optimization_scope") ==
                  "known-cost-generated-subset",
          "unknown alternative prevents full-domain optimum scope");
  auto structural = known;
  structural.target.links[0].sourceLayout = "unsupported-layout";
  require(!programJointTransferSupported(structural, 0, value, 0, error),
          "unsupported layout is distinct from an unknown supported route");
  auto partial_override = known;
  partial_override.model.parameters["link.in0.bandwidth"] = {
      1, "bytes/ns", "architecture", "explicit partial override"};
  require(!programJointTransferNs(partial_override, 0, value, 0, ns, error),
          "partial link override cannot silently use the architecture price");
  // Removing every price must preserve structural support without inventing
  // either a zero-cost baseline or a mathematical infeasibility result.
  c = config();
  for (auto &l : *architecture(c)->getArray("links"))
    (*l.getAsObject())["bandwidth_bytes_per_ms"] = 0;
  ProgramJointDomain noPrices;
  require(build(std::move(c), noPrices) && !noPrices.costsComplete,
          "all missing route costs preserve unknown coverage");
  fallback = baselineProgramJointDomain(noPrices);
  require(!fallback.hasPlan &&
              fallback.status == ConstraintSearchStatus::Unknown,
          "no fully priced plan is unknown, never infeasible or zero-cost");
  c = config();
  auto *model =
      c.getAsObject()->getObject("generic_air")->getObject("cost_model");
  (*model)["mode"] = "strict";
  for (auto &p : *parameters(c))
    (*p.second.getAsObject()->getObject("provenance"))["kind"] = "architecture";
  for (auto &l : *architecture(c)->getArray("links"))
    (*l.getAsObject()->getObject("provenance"))["kind"] = "assumed";
  ProgramJointDomain strict;
  require(build(std::move(c), strict) && !strict.costsComplete,
          "strict assumed links are unknown costs, not removed feasible "
          "placements");
  // Even a solver claiming OPTIMAL cannot cover groups discarded by a cap.
  c = config();
  auto *joint = c.getAsObject()->getObject("program_joint");
  (*joint)["max_groups"] = 1;
  ProgramJointDomain capped;
  require(build(std::move(c), capped) && !capped.complete,
          "generation cap is explicitly incomplete");
  auto pretend = baseline;
  pretend.status = ConstraintSearchStatus::Optimal;
  auto scope = serializeProgramJointSolution(capped, pretend);
  require(scope.getAsObject()->getBoolean("full_domain_optimal") == false,
          "generated subset cannot claim intended-domain optimality");
  // Adjacent integer nanoseconds stop being distinguishable above this bound.
  // Test conversion and checked sums without constructing a huge solver model.
  auto boundary = known;
  boundary.model.parameters["link.in0.startup"] = {
      double(maxExactNs), "ns", "architecture", "explicit integer boundary"};
  boundary.model.parameters["link.in0.bandwidth"] = {
      std::numeric_limits<double>::max(), "bytes/ns", "architecture",
      "synthetic precision test"};
  require(programJointTransferNs(boundary, 0, value, 0, ns, error) &&
              ns == maxExactNs,
          "largest permitted integer nanosecond remains exact");
  boundary.model.parameters["link.in0.startup"].value = double(maxExactNs) + 1;
  require(!programJointTransferNs(boundary, 0, value, 0, ns, error),
          "2^53 boundary rejected before integer nanoseconds collapse");
  boundary = known;
  boundary.variants[0].valueBytes[value] = maxExactNs + 2;
  boundary.target.links[0].maxBytes = std::numeric_limits<int64_t>::max();
  require(programJointTransferSupported(boundary, 0, value, 0, error) &&
              !programJointTransferNs(boundary, 0, value, 0, ns, error),
          "unsafe byte conversion is unknown pricing rather than unsupported "
          "transfer");
  boundary = known;
  boundary.submitNs = maxExactNs;
  boundary.configurationNs = 1;
  require(!baselineProgramJointDomain(boundary).hasPlan,
          "configuration addition cannot overflow exact integer horizon");
  boundary = known;
  boundary.completeNs = maxExactNs;
  require(!baselineProgramJointDomain(boundary).hasPlan,
          "final completion addition cannot overflow exact integer horizon");
  auto unsafe = baseline;
  unsafe.makespanNs = maxExactNs + 1;
  ProgramGraph graph;
  require(!materializeProgramJointSolution(known, unsafe, graph, error),
          "independent checker rejects unsafe integer objective");
  c = config();
  (*parameters(c)->getObject("configuration.startup"))["value"] =
      double(maxExactNs) + 1;
  ProgramJointDomain unsafeParameter;
  require(!build(std::move(c), unsafeParameter),
          "unsafe input cost rejected instead of rounded to another integer");
  llvm::outs() << "program joint domain audit checks: " << checks << "\n";
  return ok ? 0 : 1;
}

#include "Taskflow/ProgramAllHeterogeneousCost.h"
#include "Taskflow/ProgramMechanismCost.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
using Features = std::map<std::string, double>;
bool save(const std::string &path, const j::Value &value, std::string &error) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec);
  if (ec) {
    error = path + ": " + ec.message();
    return false;
  }
  out << llvm::formatv("{0:2}\n", value);
  out.close();
  if (!out.has_error())
    return true;
  error = path + ": " + out.error().message();
  out.clear_error();
  return false;
}
j::Object serialize(const Features &features) {
  j::Object object;
  for (const auto &entry : features)
    object[entry.first] = entry.second;
  return object;
}
std::optional<double> price(const j::Object *model, const Features &features,
                            std::string &unknown) {
  const auto *coefficients =
      model ? model->getObject("coefficients_ns") : nullptr;
  const auto *ranges = model ? model->getObject("ranges") : nullptr;
  if (!coefficients || coefficients->empty() || !ranges) {
    unknown =
        "phase needs explicit coefficients, provenance and feature ranges";
    return {};
  }
  double total = 0;
  for (const auto &entry : *coefficients) {
    const auto *coefficient = entry.second.getAsObject();
    const auto value =
        coefficient ? coefficient->getNumber("value") : std::nullopt;
    const auto kind =
        coefficient ? coefficient->getString("kind") : std::nullopt;
    const auto source =
        coefficient ? coefficient->getString("source") : std::nullopt;
    const auto *range = ranges->getArray(entry.first);
    auto feature = features.find(entry.first.str());
    if (!value || !std::isfinite(*value) || *value < 0 ||
        (kind != "calibrated" && kind != "architecture") || !source ||
        source->empty() || feature == features.end() || !range ||
        range->size() != 2) {
      unknown = "invalid or uncalibrated phase term: " + entry.first.str();
      return {};
    }
    const auto low = (*range)[0].getAsNumber(),
               high = (*range)[1].getAsNumber();
    if (!low || !high || !std::isfinite(*low) || !std::isfinite(*high) ||
        *low < 0 || *high < *low || feature->second < *low ||
        feature->second > *high) {
      unknown = "phase feature outside calibration: " + entry.first.str();
      return {};
    }
    total += feature->second * *value;
  }
  if (!std::isfinite(total) || total < 0) {
    unknown = "phase prediction overflow";
    return {};
  }
  return total / 1e6;
}
bool quantities(const ProgramAllHeterogeneousRegion &region,
                const j::Object *runtimeRegion, Features &features,
                std::string &error) {
  features = {{"call", 1},
              {"configuration_count", 1},
              {"configuration_argument_bindings", 0},
              {"source_operations", 0},
              {"source_memory_accesses", 0},
              {"dma_bytes", 0},
              {"dma_transfers", 0},
              {"phases", 0},
              {"hardware_stages", 0},
              {"matrix_instances", 0},
              {"layout_bytes", 0},
              {"physical_scratch_bytes", 0},
              {"abi_bytes", 0},
              {"to_device_bytes", 0},
              {"from_device_bytes", 0},
              {"abi_bo_count", double(region.projection.arguments.size())}};
  // This existing source-domain counter is used for quantities only. Its
  // unknown prices are never substituted for measured implementation costs.
  ProgramMechanismModel counter;
  counter.id = "orbit.source-quantities-only.v32";
  counter.mode = "estimate";
  counter.boundary = "source-quantities-only";
  ProgramMechanismEvaluation counted;
  ProgramTimingRequest ignored;
  if (!buildProgramMechanismCosts(region.selectedGraph, counter, counted,
                                  ignored, error, false, false))
    return false;
  // Work is counted from actual checked instance domains, including tile
  // tails. A mode set alone cannot tell which source work ran as matrix code.
  std::map<size_t, const MappingTaskInstance *> byEvent;
  for (const auto &instance : region.selectedGraph.instances) {
    byEvent[instance.event] = &instance;
    const auto prefix = "implementation." + instance.implementation +
                        ".lanes." + std::to_string(instance.lanes) + ".";
    features[prefix + "instances"] += 1;
    features[prefix + "source_operations"] += 0;
    features[prefix + "source_memory_accesses"] += 0;
  }
  for (const auto &value : counted.quantities) {
    const auto *row = value.getAsObject();
    if (!row)
      continue;
    auto event = row->getInteger("event");
    auto instance =
        event && *event >= 0 ? byEvent.find(size_t(*event)) : byEvent.end();
    for (const auto *key : {"source_operations", "source_memory_accesses"}) {
      auto quantity = row->getNumber(key);
      if (!quantity) {
        error = "actual instance source work is unknown";
        return false;
      }
      features[key] += *quantity;
      if (instance != byEvent.end()) {
        const auto &selected = *instance->second;
        const auto prefix = "implementation." + selected.implementation +
                            ".lanes." + std::to_string(selected.lanes) + ".";
        features[prefix + key] += *quantity;
      }
    }
  }
  for (const auto &event : region.selectedGraph.events)
    if (event.kind == "transfer") {
      features["dma_bytes"] += event.transferBytes;
      features["dma_transfers"] += 1;
    }
  for (const auto &instance : region.selectedGraph.instances)
    if (instance.implementation == "aie2p-bf16-mmul-f32acc")
      features["matrix_instances"] += 1;
  if (region.hasNativePlan) {
    features["phases"] = region.native.phases.size();
    features["hardware_stages"] = region.native.hardwareStages.empty()
                                      ? 1
                                      : region.native.hardwareStages.size();
  }
  for (const auto &binding : region.projection.arguments) {
    const auto &storage =
        region.selectedGraph.source.storages[binding.childStorage];
    auto type = mlir::cast<mlir::MemRefType>(storage.root.getType());
    double bytes = (type.getElementTypeBitWidth() + 7) / 8;
    for (auto extent : storage.shape)
      bytes *= extent;
    features["abi_bytes"] += bytes;
    // The prepared executor explicitly syncs every ABI BO to device, including
    // output BOs; this conservative policy is part of the calibration identity.
    if (!binding.orderOnly)
      features["to_device_bytes"] += bytes;
    if (binding.role == "output")
      features["from_device_bytes"] += bytes;
  }
  if (runtimeRegion) {
    if (auto count = runtimeRegion->getInteger("physical_invocation_count")) {
      if (*count < 1) {
        error = "physical configuration count must be positive";
        return false;
      }
      features["configuration_count"] = *count;
    }
    auto abi = runtimeRegion->getArray("npu_abi");
    if (!abi)
      abi = runtimeRegion->getArray("abi");
    if (abi) {
      features["abi_bo_count"] = abi->size();
      features["abi_bytes"] = features["to_device_bytes"] =
          features["from_device_bytes"] = 0;
      for (const auto &value : *abi) {
        const auto *binding = value.getAsObject();
        auto dtype = binding ? binding->getString("dtype") : std::nullopt;
        auto shape = binding ? binding->getArray("shape") : nullptr;
        if (!shape || (dtype != "f32" && dtype != "bf16")) {
          error = "physical runtime ABI has unsupported element geometry";
          return false;
        }
        double bytes = dtype == "bf16" ? 2 : 4;
        for (const auto &extent : *shape) {
          auto n = extent.getAsInteger();
          if (!n || *n <= 0) {
            error = "invalid physical ABI extent";
            return false;
          }
          bytes *= *n;
        }
        features["abi_bytes"] += bytes;
        if (!binding->getBoolean("order_only").value_or(false))
          features["to_device_bytes"] += bytes;
        if (binding->getString("role") == "output")
          features["from_device_bytes"] += bytes;
      }
    }
    if (auto matrix = runtimeRegion->getObject("matrix_realization")) {
      features["layout_bytes"] =
          matrix->getNumber("local_layout_payload_bytes").value_or(0);
      features["physical_scratch_bytes"] =
          matrix->getNumber("physical_scratch_bytes").value_or(0);
      // This physical receipt's layout/scratch belongs to the selected matrix
      // adapter; it is not scalar work or an idealized vector speedup.
      features["implementation.aie2p-bf16-mmul-f32acc.lanes.1.layout_bytes"] =
          features["layout_bytes"];
      features["implementation.aie2p-bf16-mmul-f32acc.lanes.1.physical_scratch_"
               "bytes"] = features["physical_scratch_bytes"];
    }
  }
  // Every configuration binds the full physical ABI, including order-only
  // and scratch arguments. This is separate from payload synchronization.
  features["configuration_argument_bindings"] =
      features["configuration_count"] * features["abi_bo_count"];
  return true;
}
std::set<std::string>
implementations(const ProgramAllHeterogeneousRegion &region) {
  std::set<std::string> result;
  for (const auto &instance : region.selectedGraph.instances)
    result.insert(instance.implementation);
  return result;
}
} // namespace

std::string
orbit::taskflow::heterogeneousCostSourceFamily(mlir::ModuleOp source) {
  mlir::OwningOpRef<mlir::ModuleOp> normalized = source.clone();
  for (const char *attribute :
       {"orbit.air_tile_sizes", "orbit.mapping", "orbit.implementation",
        "orbit.vector_lanes", "orbit.parent_implementation",
        "orbit.task_implementations", "orbit.generated_initializer_tasks",
        "orbit.program_partition_extent", "orbit.plan"})
    normalized->getOperation()->removeAttr(attribute);
  normalized->walk([&](mlir::Operation *operation) {
    operation->removeAttr("orbit.search_origin");
    operation->removeAttr("orbit.distributed_initialization");
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation))
      function.setName("calibration_region");
  });
  return staticFingerprint(j::Object{
      {"normalization", "local-source-without-mapping-attributes.v32"},
      {"source", heterogeneousProgramFamily(*normalized)}});
}

ProgramSearchEvaluation orbit::taskflow::evaluateProgramAllHeterogeneousCost(
    const ProgramAllHeterogeneousCandidate &candidate, const j::Value &config,
    HeterogeneousAIREmitter emitter) {
  ProgramSearchEvaluation result;
  result.feasibilityKnown = false;
  std::string error;
  const auto *root = config.getAsObject();
  const auto *model = root ? root->getObject("heterogeneous_cost") : nullptr;
  j::Value featuresJSON = j::Object{}, manifest = nullptr;
  std::vector<HeterogeneousSelectedRegion> selected;
  for (const auto &region : candidate.regions)
    selected.push_back(
        {*region.source, region.hasNativePlan ? &region.native : nullptr});
  const auto output = candidate.evidenceDirectory + "/implementation";
  if (!exportHeterogeneousProgram(candidate.source, config, output, manifest,
                                  error, emitter, &candidate.parent,
                                  &selected)) {
    result.message = "selected heterogeneous backend rejected: " + error;
    return result;
  }
  ProgramTimingRequest request;
  request.graphId = programGraphIdentity(candidate.parent.graph);
  request.modelIdentity =
      model ? staticFingerprint(*root->get("heterogeneous_cost")) : "unpriced";
  request.boundary =
      "prepared-host-to-host: reset, exact serial regions, output copy";
  j::Array regions;
  const auto *rows = model ? model->getArray("regions") : nullptr;
  bool calibrated =
      model &&
      model->getString("schema") == "orbit.heterogeneous.edge-cost.v32" &&
      model->getString("runtime_identity") ==
          "orbit.serial-prepared-typed-native-wait2.v32";
  const auto *executionContract =
      manifest.getAsObject()
          ? manifest.getAsObject()->getObject("execution_contract")
          : nullptr;
  calibrated = calibrated && executionContract &&
               executionContract->getString("runtime_identity") ==
                   "orbit.serial-prepared-typed-native-wait2.v32";
  calibrated =
      calibrated && manifest.getAsObject() &&
      manifest.getAsObject()->getString("execution_contract_identity") !=
          "unbound" &&
      model->getString("execution_contract_identity") ==
          manifest.getAsObject()->getString("execution_contract_identity");
  std::string previous = "application.reset";
  Features application{{"call", 1},
                       {"regions", double(candidate.regions.size())},
                       {"configuration_count", 0},
                       {"input_bytes", 0},
                       {"output_bytes", 0}};
  const auto &parent = candidate.parent.graph;
  for (const auto &storage : parent.source.storages) {
    auto type = mlir::cast<mlir::MemRefType>(storage.root.getType());
    double bytes = (type.getElementTypeBitWidth() + 7) / 8;
    for (auto extent : storage.shape)
      bytes *= extent;
    if (storage.role == "input")
      application["input_bytes"] += bytes;
    if (storage.role == "output")
      application["output_bytes"] += bytes;
  }
  if (const auto *runtimeRegions = manifest.getAsObject()->getArray("regions"))
    for (const auto &value : *runtimeRegions)
      if (const auto *region = value.getAsObject())
        if (region->getString("device") == "npu")
          application["configuration_count"] +=
              region->getInteger("physical_invocation_count").value_or(1);
  auto activity = [&](const std::string &id, const j::Object *phase,
                      const Features &features, bool runtimeOnly) {
    ProgramActivityCost cost;
    cost.id = id;
    cost.boundary = request.boundary;
    cost.provenance =
        "disjoint measured runtime phase; exact selected child features";
    cost.runtimeOnly = runtimeOnly;
    cost.resourceDemands["NPU.hostcall"] = 1;
    cost.durationMs =
        calibrated ? price(phase, features, cost.unknownReason) : std::nullopt;
    if (!calibrated)
      cost.unknownReason = "missing calibrated v32 runtime model";
    return cost;
  };
  const auto *app = model ? model->getObject("application") : nullptr;
  request.costs.push_back(activity("application.reset",
                                   app ? app->getObject("reset") : nullptr,
                                   application, true));
  for (const auto &region : candidate.regions) {
    Features features;
    auto runtimeRegions = manifest.getAsObject()->getArray("regions");
    const auto *runtimeRegion =
        runtimeRegions && region.region < runtimeRegions->size()
            ? (*runtimeRegions)[region.region].getAsObject()
            : nullptr;
    if (region.hasNativePlan &&
        (!runtimeRegion ||
         runtimeRegion->getString("physical_abi_graph_id") !=
             programGraphIdentity(region.selectedGraph) ||
         !runtimeRegion->getArray("npu_abi"))) {
      result.message =
          "selected NPU cost requires matching physical AIR ABI evidence";
      return result;
    }
    if (!quantities(region, runtimeRegion, features, error)) {
      result.message = "selected implementation features unknown: " + error;
      return result;
    }
    const auto device = candidate.parent.slices[region.region].device;
    const auto family = heterogeneousCostSourceFamily(*region.source);
    const auto kinds = implementations(region);
    const j::Object *row = nullptr;
    if (rows)
      for (const auto &value : *rows) {
        const auto *possible = value.getAsObject();
        if (!possible || possible->getString("device") != device ||
            possible->getString("source_family") != family)
          continue;
        std::set<std::string> declared;
        if (const auto *list = possible->getArray("implementations"))
          for (const auto &kind : *list)
            if (auto text = kind.getAsString())
              declared.insert(text->str());
        if (declared == kinds) {
          if (row) {
            result.message = "duplicate local implementation calibration";
            return result;
          }
          row = possible;
        }
      }
    const auto *phases = row ? row->getObject("phases") : nullptr;
    j::Array names;
    for (const auto &kind : kinds)
      names.push_back(kind);
    regions.push_back(j::Object{
        {"region", int64_t(region.region)},
        {"device", device},
        {"source_family", family},
        {"implementations", std::move(names)},
        {"features", serialize(features)},
        {"child_graph_id", programGraphIdentity(region.selectedGraph)}});
    const std::vector<std::string> phaseNames =
        device == "cpu" ? std::vector<std::string>{"body"}
                        : std::vector<std::string>{"sync_to", "submit", "wait",
                                                   "sync_from"};
    for (const auto &phase : phaseNames) {
      auto cost = activity(
          "region." + std::to_string(region.region) + "." + phase,
          phases ? phases->getObject(phase) : nullptr, features, false);
      // A single-launch calibration cannot price a split implementation.
      // Both submission and completion include per-configuration overhead.
      if (device == "npu" && features.at("configuration_count") > 1 &&
          (phase == "submit" || phase == "wait")) {
        const auto *terms =
            phases && phases->getObject(phase)
                ? phases->getObject(phase)->getObject("coefficients_ns")
                : nullptr;
        if (!terms || !terms->get("configuration_count")) {
          cost.durationMs.reset();
          cost.unknownReason =
              "multiple configurations require calibrated launch-count terms";
        }
      }
      // New explicit mixed-task selection requires implementation-resolved
      // work calibration. Aggregate source_operations and a set of kind names
      // cannot price swapping unlike tasks between scalar and matrix code.
      if (device == "npu" && phase == "wait" &&
          (*region.source)->hasAttr("orbit.task_implementations")) {
        const auto *terms =
            phases && phases->getObject(phase)
                ? phases->getObject(phase)->getObject("coefficients_ns")
                : nullptr;
        size_t activeKinds = 0;
        bool covered = true;
        for (const auto &feature : features)
          if (llvm::StringRef(feature.first).starts_with("implementation.") &&
              llvm::StringRef(feature.first).ends_with(".source_operations") &&
              feature.second > 0) {
            ++activeKinds;
            covered &= terms && terms->get(feature.first);
          }
        if (activeKinds > 1 && !covered) {
          cost.durationMs.reset();
          cost.unknownReason = "explicit mixed task implementations require "
                               "per-implementation source-work calibration";
        }
      }
      cost.predecessors = {previous};
      if (device == "cpu" || phase == "wait")
        cost.events = parent.executionRegions[region.region].memberEvents;
      for (size_t index = 0; index < parent.regionEvents.size(); ++index) {
        const auto &event = parent.regionEvents[index];
        if (event.region != region.region)
          continue;
        if (device == "cpu" ||
            (phase == "sync_to" && event.kind == "visible-to-device") ||
            (phase == "submit" && event.kind == "submit") ||
            (phase == "wait" && event.kind == "complete") ||
            (phase == "sync_from" && event.kind == "visible-to-host"))
          cost.regionEvents.push_back(index);
      }
      previous = cost.id;
      for (size_t event : cost.events)
        cost.resourceDemands[parent.events[event].resource] = 1;
      for (size_t event : cost.regionEvents)
        cost.resourceDemands[parent.regionEvents[event].resource] = 1;
      request.costs.push_back(std::move(cost));
    }
  }
  for (const auto *phase : {"control", "copy"}) {
    auto cost =
        activity("application." + std::string(phase),
                 app ? app->getObject(phase) : nullptr, application, true);
    cost.predecessors = {previous};
    previous = cost.id;
    request.costs.push_back(std::move(cost));
  }
  ProgramTimingEvaluation timed;
  if (!evaluateProgramTiming(parent, request, timed, error)) {
    result.message = "phase ownership/timing rejected: " + error;
    return result;
  }
  featuresJSON = j::Object{{"schema", "orbit.heterogeneous.edge-features.v32"},
                           {"candidate_id", candidate.assignmentId},
                           {"parent_graph_id", request.graphId},
                           {"application", serialize(application)},
                           {"regions", std::move(regions)}};
  if (!save(candidate.evidenceDirectory + "/edge-features.json", featuresJSON,
            error)) {
    result.message = error;
    return result;
  }
  result.feasible = true;
  result.feasibilityKnown = true;
  result.costMs = timed.makespanMs;
  result.message =
      result.costMs
          ? "calibrated disjoint CPU/NPU runtime phases"
          : "selected implementation exported; runtime phase costs unknown";
  j::Array regionIdentities;
  for (const auto &region : candidate.regions)
    regionIdentities.push_back(programGraphIdentity(region.selectedGraph));
  result.result =
      j::Object{{"schema", "orbit.heterogeneous.edge-evaluation.v32"},
                {"candidate_id", candidate.assignmentId},
                {"parent_graph_id", request.graphId},
                {"region_graph_ids", std::move(regionIdentities)},
                {"features", std::move(featuresJSON)},
                {"timing", serializeProgramTiming(timed)},
                {"implementation_manifest", output + "/manifest.json"},
                {"physical_execution", false}};
  return result;
}

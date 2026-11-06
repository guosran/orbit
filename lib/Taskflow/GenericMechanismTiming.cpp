#include "Taskflow/GenericMechanismTiming.h"
#include "Taskflow/ProgramMechanismCost.h"
#include <cmath>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &error, const std::string &message) {
  error = "native graph mechanism cost: " + message;
  return false;
}
}
bool orbit::taskflow::evaluateGenericMechanismCost(const GenericAnalysis &source,
    const GenericMappingPlan &plan, const llvm::json::Value &model,
    const llvm::json::Value &target, GenericDirectAIRCost &cost,
    std::string &error) {
  cost = {}; cost.enabled = true; cost.model = llvm::json::Value(model);
  cost.target = llvm::json::Value(target); cost.modelDigest = staticFingerprint(model);
  if (!collectGenericCostFeatures(source, plan, cost.features, error)) return false;
  cost.featureDigest = staticFingerprint(serializeGenericCostFeatures(cost.features));
  auto object = model.getAsObject();
  if (!object || object->getString("schema") != "orbit.direct-air.cost.v1" ||
      object->getString("kind") != "event-mechanism")
    return fail(error, "existing direct-AIR schema with event-mechanism kind required");
  ProgramMechanismModel parameters;
  parameters.id = cost.modelDigest;
  auto mode = object->getString("mode"), id = object->getString("model_id"),
       boundary = object->getString("boundary");
  if (!mode || !id || id->empty() || !boundary || boundary->empty())
    return fail(error, "explicit mode, model_id and boundary required");
  parameters.mode = mode->str(); parameters.boundary = boundary->str();
  auto implementations = object->getArray("implementations");
  auto values = object->getObject("parameters");
  if (!implementations || implementations->empty() || !values)
    return fail(error, "implementation coverage and parameter table required");
  for (const auto &entry : *implementations) {
    auto text = entry.getAsString();
    if (!text || text->empty()) return fail(error, "invalid implementation coverage");
    parameters.implementations.push_back(text->str());
  }
  for (const auto &entry : *values) {
    auto row = entry.second.getAsObject();
    auto provenance = row ? row->getObject("provenance") : nullptr;
    auto unit = row ? row->getString("unit") : std::nullopt;
    auto kind = provenance ? provenance->getString("kind") : std::nullopt;
    auto sourceName = provenance ? provenance->getString("source") : std::nullopt;
    if (!row || !provenance || !unit || !kind || !sourceName)
      return fail(error, "each parameter needs unit and explicit kind/source provenance: " + entry.first.str());
    ProgramMechanismParameter parameter;
    parameter.unit = unit->str(); parameter.kind = kind->str();
    parameter.source = sourceName->str();
    if (auto value = row->get("value")) {
      if (value->getAsNumber()) parameter.value = *value->getAsNumber();
      else if (!value->getAsNull()) return fail(error, "parameter value must be a number or null");
    }
    parameters.parameters[entry.first.str()] = parameter;
  }
  // An explicitly priced architecture link can supply backend-known values.
  // A zero/uncalibrated default is never turned into infinite bandwidth or a
  // free physical link. Explicit parameter rows take precedence.
  for (const auto &link : plan.communication.links) {
    if (link.bandwidthBytesPerMs <= 0 || link.provenance.source.empty()) continue;
    std::string prefix = "link." + link.id;
    for (const auto &row : std::vector<std::pair<std::string, double>>{
        {prefix + ".startup", link.latencyMs * 1e6},
        {prefix + ".bandwidth", link.bandwidthBytesPerMs / 1e6}}) {
      if (parameters.parameters.count(row.first)) continue;
      ProgramMechanismParameter p; p.value = row.second;
      p.unit = row.first == prefix + ".startup" ? "ns" : "bytes/ns";
      p.kind = link.provenance.kind;
      if (p.kind != "calibrated" && p.kind != "architecture" && p.kind != "assumed")
        p.kind = "assumed"; // Unknown provenance never becomes an architecture fact.
      p.source = link.provenance.source;
      parameters.parameters[row.first] = p;
    }
  }
  if (auto required = object->getObject("target")) {
    auto actual = target.getAsObject();
    for (const auto &entry : *required) {
      if (!actual || !actual->get(entry.first) || *actual->get(entry.first) != entry.second) {
        cost.status = "unknown-target-mismatch";
        cost.reason = "declared graph-mechanism target condition mismatch: " + entry.first.str();
        return true;
      }
    }
  }
  ProgramMechanismEvaluation evaluated;
  if (!evaluateProgramMechanismCost(plan.communication, parameters, evaluated, error)) return false;
  cost.eventTiming = serializeProgramMechanismCost(evaluated);
  cost.valueMs = evaluated.timing.makespanMs;
  cost.status = evaluated.status;
  cost.reason = cost.valueMs ?
      "Public per-activity costs and resource-constrained schedule; provenance and assumptions explicit. This is algorithm-validation estimation, not a validated total-time or ranking claim." :
      "Required graph activities remain unpriced; inspect public timeline, known subtotals and unknown reasons. Strict mode does not substitute assumed parameters.";
  return true;
}

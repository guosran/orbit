#include "Taskflow/ProgramOperatorSource.h"
#include "Taskflow/JSONUtils.h"
#include "Taskflow/ProgramOperatorMapping.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
bool fail(std::string &error, llvm::StringRef message) {
  error = "operator source: " + message.str();
  return false;
}
bool text(const j::Object &object, llvm::StringRef key, std::string &result) {
  auto value = object.getString(key);
  if (!value || value->empty())
    return false;
  result = value->str();
  return true;
}
bool digest(llvm::StringRef value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool integers(const j::Object &object, llvm::StringRef key,
              std::vector<int64_t> &result) {
  auto *values = object.getArray(key);
  if (!values)
    return false;
  for (const auto &value : *values) {
    auto n = value.getAsInteger();
    if (!n)
      return false;
    result.push_back(*n);
  }
  return true;
}
int64_t elementBytes(llvm::StringRef dtype) {
  if (dtype == "f32" || dtype == "i32")
    return 4;
  if (dtype == "bf16" || dtype == "f16" || dtype == "i16")
    return 2;
  if (dtype == "i8" || dtype == "ui8")
    return 1;
  if (dtype == "i64" || dtype == "f64")
    return 8;
  return 0;
}
int64_t bytes(const ProgramOperatorValue &value) {
  int64_t result = elementBytes(value.dtype);
  if (!result)
    return -1;
  for (int64_t size : value.shape) {
    if (size <= 0 || result > INT64_MAX / size)
      return -1;
    result *= size;
  }
  return result;
}
bool references(const j::Object &object, llvm::StringRef key,
                const std::map<std::string, size_t> &ids,
                std::vector<size_t> &result) {
  auto *values = object.getArray(key);
  if (!values)
    return false;
  std::set<size_t> unique;
  for (const auto &value : *values) {
    auto id = value.getAsString();
    if (!id)
      return false;
    auto found = ids.find(id->str());
    if (found == ids.end() || !unique.insert(found->second).second)
      return false;
    result.push_back(found->second);
  }
  return true;
}
} // namespace

bool orbit::taskflow::parseProgramOperatorSource(
    const j::Value &document, std::shared_ptr<ProgramOperatorSource> &result,
    std::string &error) {
  auto *root = document.getAsObject();
  if (!root || root->getString("schema") != "orbit.program-operator-source.v1")
    return fail(error, "wrong source schema");
  auto source = std::make_shared<ProgramOperatorSource>();
  if (!text(*root, "source_id", source->sourceId) ||
      !text(*root, "numerical_contract", source->numericalContract))
    return fail(error, "missing source identity or numerical contract");
  auto *bindings = root->getArray("bindings"),
       *values = root->getArray("values"),
       *operators = root->getArray("operators");
  if (!bindings || !values || !operators ||
      operators->size() > programOperatorMaxOperators ||
      values->size() > programOperatorMaxValues)
    return fail(error, "missing or oversized finite source tables");
  std::map<std::string, size_t> valueIds, operatorIds;
  for (const auto &v : *bindings) {
    auto *o = v.getAsObject();
    ProgramOperatorBinding binding;
    if (!o || !text(*o, "id", binding.id) || !text(*o, "file", binding.file) ||
        !text(*o, "sha256", binding.sha256) ||
        !text(*o, "symbol", binding.symbol))
      return fail(error, "incomplete source binding");
    source->bindings.push_back(std::move(binding));
  }
  for (const auto &v : *values) {
    auto *o = v.getAsObject();
    ProgramOperatorValue value;
    if (!o || !text(*o, "id", value.id) || !text(*o, "dtype", value.dtype) ||
        !text(*o, "role", value.role) || !text(*o, "layout", value.layout) ||
        !integers(*o, "shape", value.shape) ||
        !valueIds.emplace(value.id, source->values.size()).second)
      return fail(error, "invalid or duplicate source value");
    if (auto memory = o->getString("initial_location"))
      value.initialMemory = memory->str();
    if (auto init = o->getString("initialization"))
      value.initialization = init->str();
    source->values.push_back(std::move(value));
  }
  for (size_t index = 0; index < values->size(); ++index)
    if (auto *view = (*values)[index].getAsObject()->getObject("view")) {
      auto base = view->getString("base");
      ProgramOperatorView typed;
      if (!base || !valueIds.count(base->str()) ||
          !integers(*view, "offsets", typed.offsets) ||
          !integers(*view, "strides", typed.strides))
        return fail(error, "invalid physical view description");
      typed.base = valueIds.at(base->str());
      source->values[index].view = std::move(typed);
    }
  for (size_t index = 0; index < values->size(); ++index)
    if (auto *view =
            (*values)[index].getAsObject()->getObject("allocation_view")) {
      auto base = view->getString("base");
      ProgramOperatorView typed;
      if (!base || !valueIds.count(base->str()) ||
          !integers(*view, "offsets", typed.offsets) ||
          !integers(*view, "strides", typed.strides))
        return fail(error, "invalid physical view description");
      typed.base = valueIds.at(base->str());
      source->values[index].allocationView = std::move(typed);
    }
  for (size_t index = 0; index < values->size(); ++index)
    if (auto base =
            (*values)[index].getAsObject()->getString("state_alias_of")) {
      if (!valueIds.count(base->str()))
        return fail(error, "invalid state alias base");
      source->values[index].stateAliasOf = valueIds.at(base->str());
    }
  for (const auto &v : *operators) {
    auto *o = v.getAsObject();
    std::string id;
    if (!o || !text(*o, "id", id) ||
        !operatorIds.emplace(id, operatorIds.size()).second)
      return fail(error, "invalid or duplicate source operator");
  }
  for (const auto &v : *operators) {
    auto &o = *v.getAsObject();
    ProgramSourceOperator op;
    auto ordinal = o.getInteger("source_ordinal");
    if (!text(o, "id", op.id) || !text(o, "kind", op.kind) ||
        !text(o, "binding", op.binding) || !ordinal || *ordinal < 0 ||
        !references(o, "inputs", valueIds, op.inputs) ||
        !references(o, "outputs", valueIds, op.outputs) ||
        !references(o, "dependencies", operatorIds, op.dependencies))
      return fail(error, "incomplete operator semantics/interfaces");
    op.sourceOrdinal = *ordinal;
    if (auto *attributes = o.get("attributes"))
      op.attributes = copyJSONWithoutFields(*attributes);
    source->operators.push_back(std::move(op));
  }
  source->document = copyJSONWithoutFields(document);
  if (!checkProgramOperatorSource(*source, error))
    return false;
  result = std::move(source);
  return true;
}

bool orbit::taskflow::checkProgramOperatorSource(const ProgramOperatorSource &s,
                                                 std::string &error) {
  if (!digest(s.sourceId) || s.numericalContract.empty() ||
      s.operators.empty() || s.values.empty())
    return fail(error, "incomplete external source contract");
  std::set<std::string> bindingIds, valueIds, opIds;
  for (const auto &b : s.bindings)
    if (b.id.empty() || !bindingIds.insert(b.id).second || b.file.empty() ||
        b.symbol.empty() || !digest(b.sha256))
      return fail(error, "invalid source binding");
  std::vector<int64_t> producer(s.values.size(), -1);
  for (size_t i = 0; i < s.values.size(); ++i) {
    const auto &v = s.values[i];
    if (v.id.empty() || !valueIds.insert(v.id).second || bytes(v) <= 0 ||
        v.layout != "row_major" ||
        (v.role != "input" && v.role != "temporary" && v.role != "output" &&
         v.role != "workspace"))
      return fail(error, "invalid tensor shape/type/role/layout");
    if (v.role == "workspace" && (v.initialization != "uninitialized" ||
                                  v.view || v.allocationView || v.stateAliasOf))
      return fail(error,
                  "workspace requires explicit uninitialized raw storage");
    if (v.stateAliasOf && (v.view || *v.stateAliasOf >= i ||
                           s.values[*v.stateAliasOf].dtype != v.dtype ||
                           s.values[*v.stateAliasOf].shape != v.shape))
      return fail(error, "invalid/cyclic or differently shaped state alias");
    if (v.allocationView) {
      const auto &view = *v.allocationView;
      if (v.view || v.stateAliasOf || view.base >= i ||
          s.values[view.base].role != "workspace" ||
          s.values[view.base].dtype != v.dtype || v.shape.size() != 1 ||
          view.offsets.size() != 1 || view.strides != std::vector<int64_t>{1} ||
          view.offsets[0] < 0 ||
          __int128(view.offsets[0]) + v.shape[0] >
              bytes(s.values[view.base]) / elementBytes(v.dtype))
        return fail(error, "write-only allocation view must be a bounded "
                           "contiguous workspace slice");
    }
    if (v.view) {
      const auto &view = *v.view;
      if (view.base >= i || view.offsets.size() != v.shape.size() ||
          view.strides.size() != v.shape.size() ||
          s.values[view.base].dtype != v.dtype)
        return fail(error, "invalid/cyclic tensor view");
      // Strides and offsets address the base's flat element allocation. A view
      // may be noncontiguous; source shape alone is not an allocation promise.
      int64_t last = 0;
      for (size_t axis = 0; axis < v.shape.size(); ++axis) {
        if (view.offsets[axis] < 0 || view.strides[axis] <= 0)
          return fail(error, "negative view address");
        __int128 term = (__int128(view.offsets[axis]) + v.shape[axis] - 1) *
                        view.strides[axis];
        if (term > INT64_MAX - last)
          return fail(error, "view address overflow");
        last += int64_t(term);
      }
      if (last >= bytes(s.values[view.base]) / elementBytes(v.dtype))
        return fail(error, "view escapes base tensor allocation");
    }
  }
  for (size_t i = 0; i < s.operators.size(); ++i) {
    const auto &op = s.operators[i];
    if (op.id.empty() || op.kind.empty() || !opIds.insert(op.id).second ||
        !bindingIds.count(op.binding) || op.sourceOrdinal < 0 ||
        !op.attributes.getAsObject())
      return fail(error, "unbound/duplicate operator semantics");
    for (size_t dependency : op.dependencies)
      if (dependency >= i)
        return fail(error, "non-topological source control edge");
    for (size_t input : op.inputs)
      if (input >= s.values.size() || s.values[input].role == "workspace" ||
          (producer[input] < 0 && s.values[input].role != "input"))
        return fail(error, "source reads undefined tensor version");
    for (size_t output : op.outputs) {
      if (output >= s.values.size() || producer[output] >= 0 ||
          s.values[output].role == "input")
        return fail(error,
                    "source tensor version has invalid/multiple writers");
      if (s.values[output].view &&
          std::find(op.inputs.begin(), op.inputs.end(),
                    s.values[output].view->base) == op.inputs.end())
        return fail(error, "view result has no source base operand");
      if (s.values[output].stateAliasOf &&
          std::find(op.inputs.begin(), op.inputs.end(),
                    *s.values[output].stateAliasOf) == op.inputs.end())
        return fail(error,
                    "state mutation does not consume previous state version");
      producer[output] = i;
    }
  }
  for (size_t i = 0; i < s.values.size(); ++i)
    if (producer[i] < 0 && s.values[i].role != "input" &&
        s.values[i].role != "workspace")
      return fail(error, "source value has no defining operator");
  std::vector<size_t> roots(s.values.size()), stateVersions(s.values.size()),
      latest(s.values.size());
  for (size_t v = 0; v < s.values.size(); ++v) {
    roots[v] = v;
    stateVersions[v] = v;
    if (s.values[v].view) {
      roots[v] = roots[s.values[v].view->base];
      stateVersions[v] = stateVersions[s.values[v].view->base];
    } else if (s.values[v].stateAliasOf)
      roots[v] = roots[*s.values[v].stateAliasOf];
    else if (s.values[v].allocationView)
      roots[v] = s.values[v].allocationView->base;
    latest[v] = v;
  }
  std::vector<bool> overwritten(s.values.size());
  std::map<size_t, std::vector<size_t>> allocationWriters;
  auto writeValue = [&](size_t value) {
    while (s.values[value].view)
      value = s.values[value].view->base;
    return s.values[value].allocationView ? value : SIZE_MAX;
  };
  auto partialWrite = [&](size_t value) {
    size_t writer = writeValue(value);
    if (writer == SIZE_MAX)
      return false;
    const auto &view = *s.values[writer].allocationView;
    return view.offsets[0] != 0 || s.values[writer].shape[0] !=
                                       bytes(s.values[view.base]) /
                                           elementBytes(s.values[writer].dtype);
  };
  for (const auto &op : s.operators) {
    for (size_t input : op.inputs) {
      size_t writer = writeValue(input);
      if (writer != SIZE_MAX && overwritten[writer])
        return fail(error, "reader consumes an overwritten workspace slice");
    }
    for (size_t input : op.inputs)
      if (!partialWrite(input) && stateVersions[input] != latest[roots[input]])
        return fail(error, "reader consumes overwritten state version or view");
    for (size_t output : op.outputs) {
      if (s.values[output].stateAliasOf)
        latest[roots[output]] = output;
      if (s.values[output].allocationView) {
        const auto &view = *s.values[output].allocationView;
        if (!partialWrite(output))
          latest[roots[output]] = output;
        auto &writers = allocationWriters[view.base];
        writers.erase(
            std::remove_if(writers.begin(), writers.end(),
                           [&](size_t previous) {
                             const auto &old =
                                 *s.values[previous].allocationView;
                             bool overlap =
                                 view.offsets[0] <
                                     old.offsets[0] +
                                         s.values[previous].shape[0] &&
                                 old.offsets[0] < view.offsets[0] +
                                                      s.values[output].shape[0];
                             if (overlap)
                               overwritten[previous] = true;
                             return overlap;
                           }),
            writers.end());
        writers.push_back(output);
      }
    }
  }
  if (staticFingerprint(serializeProgramOperatorSource(s)) !=
      staticFingerprint(s.document))
    return fail(error,
                "typed source semantics differ from registered document");
  return true;
}

j::Value orbit::taskflow::serializeProgramOperatorSource(
    const ProgramOperatorSource &s) {
  j::Array bindings, values, operators;
  for (const auto &b : s.bindings)
    bindings.push_back(j::Object{{"id", b.id},
                                 {"file", b.file},
                                 {"sha256", b.sha256},
                                 {"symbol", b.symbol}});
  for (const auto &v : s.values) {
    j::Array shape;
    for (auto n : v.shape)
      shape.push_back(n);
    j::Object value{{"id", v.id},
                    {"dtype", v.dtype},
                    {"shape", std::move(shape)},
                    {"role", v.role},
                    {"layout", v.layout}};
    if (v.view) {
      j::Array offsets, strides;
      for (auto n : v.view->offsets)
        offsets.push_back(n);
      for (auto n : v.view->strides)
        strides.push_back(n);
      value["view"] = j::Object{{"base", s.values[v.view->base].id},
                                {"offsets", std::move(offsets)},
                                {"strides", std::move(strides)}};
    }
    if (v.allocationView) {
      j::Array offsets, strides;
      for (auto n : v.allocationView->offsets)
        offsets.push_back(n);
      for (auto n : v.allocationView->strides)
        strides.push_back(n);
      value["allocation_view"] =
          j::Object{{"base", s.values[v.allocationView->base].id},
                    {"offsets", std::move(offsets)},
                    {"strides", std::move(strides)}};
    }
    if (!v.initialization.empty())
      value["initialization"] = v.initialization;
    if (v.stateAliasOf)
      value["state_alias_of"] = s.values[*v.stateAliasOf].id;
    if (!v.initialMemory.empty())
      value["initial_location"] = v.initialMemory;
    values.push_back(std::move(value));
  }
  for (const auto &op : s.operators) {
    j::Array inputs, outputs, dependencies;
    for (size_t v : op.inputs)
      inputs.push_back(s.values[v].id);
    for (size_t v : op.outputs)
      outputs.push_back(s.values[v].id);
    for (size_t o : op.dependencies)
      dependencies.push_back(s.operators[o].id);
    operators.push_back(
        j::Object{{"id", op.id},
                  {"kind", op.kind},
                  {"binding", op.binding},
                  {"source_ordinal", op.sourceOrdinal},
                  {"inputs", std::move(inputs)},
                  {"outputs", std::move(outputs)},
                  {"dependencies", std::move(dependencies)},
                  {"attributes", copyJSONWithoutFields(op.attributes)}});
  }
  return j::Object{{"schema", "orbit.program-operator-source.v1"},
                   {"source_id", s.sourceId},
                   {"numerical_contract", s.numericalContract},
                   {"bindings", std::move(bindings)},
                   {"values", std::move(values)},
                   {"operators", std::move(operators)}};
}

namespace {
std::vector<size_t> sorted(std::vector<size_t> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}
bool checkSelection(const ProgramOperatorSource &source,
                    const ProgramOperatorSelection &selection,
                    std::vector<size_t> &owners, std::string &error) {
  owners.assign(source.operators.size(), SIZE_MAX);
  std::set<std::string> ids;
  for (size_t r = 0; r < selection.regions.size(); ++r) {
    const auto &impl = selection.regions[r];
    if (!impl.executable || impl.sourceId != source.sourceId ||
        impl.id.empty() || !ids.insert(impl.id).second ||
        impl.entrypoint.empty() || impl.artifactIdentity.empty() ||
        impl.resource.empty() || impl.memory.empty() ||
        (impl.device != "cpu" && impl.device != "npu") ||
        impl.operators.empty() ||
        impl.costBoundary != "inclusive-region-wall" ||
        impl.transportOwnership != "implementation-inclusive" ||
        (impl.visibilityPolicy != "host-materialized" &&
         impl.visibilityPolicy != "device-resident") ||
        (impl.device == "cpu" ? impl.launchCount != 0 : impl.launchCount < 1) ||
        (impl.wallMs && (!std::isfinite(*impl.wallMs) || *impl.wallMs < 0)))
      return fail(error,
                  "invalid executable implementation/cost ownership contract");
    auto members = sorted(impl.operators);
    if (members.size() != impl.operators.size() ||
        members.back() >= source.operators.size())
      return fail(error, "invalid/duplicate implementation source coverage");
    // The first executable realization accepts contiguous source regions only.
    // A named fused implementation, not adjacency, grants this coverage.
    if (members.back() - members.front() + 1 != members.size())
      return fail(error,
                  "noncontiguous implementation coverage is unsupported");
    for (size_t op : members) {
      if (owners[op] != SIZE_MAX)
        return fail(error, "overlapping implementation coverage");
      owners[op] = r;
    }
    if (r && selection.regions[r - 1].operators.back() >= members.front())
      return fail(error,
                  "implementation regions violate source execution order");
  }
  if (std::find(owners.begin(), owners.end(), SIZE_MAX) != owners.end())
    return fail(error, "implementation selection omits source operators");
  std::vector<int64_t> producers(source.values.size(), -1);
  std::vector<std::vector<size_t>> readers(source.values.size());
  for (size_t o = 0; o < source.operators.size(); ++o) {
    for (size_t v : source.operators[o].outputs)
      producers[v] = o;
    for (size_t v : source.operators[o].inputs)
      readers[v].push_back(o);
  }
  for (size_t r = 0; r < selection.regions.size(); ++r) {
    std::vector<size_t> inputs, outputs;
    for (size_t o : selection.regions[r].operators) {
      for (size_t v : source.operators[o].inputs)
        if (producers[v] < 0 || owners[producers[v]] != r)
          inputs.push_back(v);
      for (size_t v : source.operators[o].outputs) {
        bool external = source.values[v].role == "output";
        for (size_t user : readers[v])
          external |= owners[user] != r;
        if (external)
          outputs.push_back(v);
      }
    }
    if (sorted(inputs) != sorted(selection.regions[r].inputs) ||
        sorted(outputs) != sorted(selection.regions[r].outputs))
      return fail(
          error,
          "implementation ABI does not exactly cover source boundary values");
  }
  return true;
}
} // namespace

namespace {
size_t physicalRoot(const ProgramOperatorSource &source, size_t value) {
  while (source.values[value].view || source.values[value].allocationView ||
         source.values[value].stateAliasOf)
    value = source.values[value].view
                ? source.values[value].view->base
                : (source.values[value].allocationView
                       ? source.values[value].allocationView->base
                       : *source.values[value].stateAliasOf);
  return value;
}
std::string portMemory(const ProgramOperatorImplementation &impl, size_t value,
                       bool input) {
  const auto &locations = input ? impl.inputLocations : impl.outputLocations;
  auto found = locations.find(value);
  return found == locations.end() ? impl.memory : found->second;
}
std::vector<std::string>
lifecycleKinds(const ProgramOperatorImplementation &impl) {
  if (impl.device == "npu" && impl.visibilityPolicy == "host-materialized")
    return {"visible-to-device", "submit", "complete", "visible-to-host"};
  return {"submit", "complete"};
}
} // namespace
bool orbit::taskflow::materializeProgramOperatorGraph(
    std::shared_ptr<const ProgramOperatorSource> source,
    const ProgramOperatorSelection &selection, const StaticConfig &target,
    llvm::StringRef hostMemory, ProgramGraph &graph, std::string &error) {
  if (!source || !checkProgramOperatorSource(*source, error))
    return false;
  std::vector<size_t> owners;
  if (!checkSelection(*source, selection, owners, error) ||
      !validateMemoryArchitecture(target, error))
    return false;
  const auto &s = *source;
  size_t count = selection.regions.size();
  std::vector<int64_t> producer(s.values.size(), -1);
  for (size_t o = 0; o < s.operators.size(); ++o)
    for (size_t v : s.operators[o].outputs)
      producer[v] = owners[o];
  std::set<size_t> physicalValues;
  for (const auto &impl : selection.regions) {
    physicalValues.insert(impl.inputs.begin(), impl.inputs.end());
    physicalValues.insert(impl.outputs.begin(), impl.outputs.end());
  }
  for (auto v :
       std::vector<size_t>(physicalValues.begin(), physicalValues.end())) {
    size_t base = v;
    while (s.values[base].view || s.values[base].allocationView ||
           s.values[base].stateAliasOf) {
      base = s.values[base].view ? s.values[base].view->base
                                 : (s.values[base].allocationView
                                        ? s.values[base].allocationView->base
                                        : *s.values[base].stateAliasOf);
      physicalValues.insert(base);
    }
  }
  graph = {};
  graph.operatorSource = std::move(source);
  graph.operatorSelection =
      std::make_shared<ProgramOperatorSelection>(selection);
  graph.domains = target.domains;
  graph.memories = target.memories;
  graph.resources = target.resources;
  graph.links = target.links;
  graph.targetIdentity = "operator-runtime-contract-v1";
  graph.bodyFingerprint = s.sourceId;
  std::map<size_t, size_t> copyForValue, allocationForRoot;
  for (size_t v : physicalValues)
    if (s.values[v].role == "workspace") {
      MappingAllocation allocation;
      allocation.storage = v;
      allocation.memory = s.values[v].initialMemory.empty()
                              ? hostMemory.str()
                              : s.values[v].initialMemory;
      allocation.key = s.values[v].id;
      allocation.bytes = bytes(s.values[v]);
      allocation.beginPhase = 0;
      allocation.endPhase = 1;
      allocation.addressing = "external-abi";
      allocationForRoot[v] = graph.allocations.size();
      graph.allocations.push_back(std::move(allocation));
    }
  for (size_t v : physicalValues) {
    if (s.values[v].role == "workspace")
      continue;
    size_t index = graph.regions.size();
    copyForValue[v] = index;
    const auto &value = s.values[v];
    size_t root = physicalRoot(s, v);
    graph.regions.push_back(
        {v, std::vector<int64_t>(value.shape.size(), 0), value.shape});
    graph.versions.push_back({index, producer[v]});
    std::string memory =
        value.initialMemory.empty() ? hostMemory.str() : value.initialMemory;
    if (producer[v] >= 0)
      memory = portMemory(selection.regions[producer[v]], v, false);
    size_t allocation = graph.allocations.size();
    if (root != v)
      allocation = allocationForRoot.at(root);
    else {
      MappingAllocation a;
      a.storage = v;
      a.memory = memory;
      a.key = value.id;
      a.bytes = bytes(value);
      a.beginPhase = producer[v] < 0 ? 0 : size_t(producer[v]);
      a.endPhase = a.beginPhase + 1;
      a.addressing = "external-abi";
      graph.allocations.push_back(std::move(a));
      allocationForRoot[v] = allocation;
    }
    if (graph.allocations[allocation].memory != memory)
      return fail(error, "physical state/view alias changes memory without a "
                         "transfer implementation");
    MappingCopy copy;
    copy.version = index;
    copy.allocation = allocation;
    copy.createdBy = producer[v];
    copy.availableFrom = producer[v] < 0 ? 0 : size_t(producer[v]);
    copy.releasedAfter = copy.availableFrom + 1;
    if (value.role == "output")
      copy.releasedAfter = count;
    graph.copies.push_back(std::move(copy));
    if (value.role == "input")
      graph.inputLocations[v] = memory;
    if (value.role == "output")
      graph.outputLocations[v] = memory;
  }
  for (size_t o = 0; o < s.operators.size(); ++o) {
    MappingTaskInstance instance;
    instance.task = o;
    instance.domainTask = o;
    instance.event = owners[o];
    instance.phase = owners[o];
    instance.kind = "external-source-operator";
    instance.implementation = selection.regions[owners[o]].id;
    graph.instances.push_back(std::move(instance));
    const auto &impl = selection.regions[owners[o]];
    graph.placements.push_back({o, impl.resource, impl.memory, impl.id});
  }
  for (size_t r = 0; r < count; ++r) {
    const auto &impl = selection.regions[r];
    MappingEvent event;
    event.kind = "implementation";
    event.task = impl.operators.front();
    event.phase = r;
    event.resource = impl.resource;
    event.memory = impl.memory;
    if (r)
      event.dependencies = {r - 1}; // conservative real serial runtime order
    for (size_t v : impl.inputs) {
      size_t copy = copyForValue.at(v);
      auto &location = graph.copies[copy];
      if (graph.allocations[location.allocation].memory !=
          portMemory(impl, v, true))
        return fail(
            error,
            "implementation input requires an undeclared memory transition");
      event.reads.push_back(copy);
      location.users.push_back(r);
      location.releasedAfter = std::max(location.releasedAfter, r + 1);
      size_t edge = graph.dataEdges.size();
      event.dataEdges.push_back(edge);
      location.consumerEdges.push_back(edge);
      graph.dataEdges.push_back(
          {producer[v],
           r,
           copy,
           copy,
           copy,
           {portMemory(impl, v, true), "row_major", s.values[v].dtype}});
    }
    for (size_t v : impl.outputs)
      event.writes.push_back(copyForValue.at(v));
    graph.events.push_back(std::move(event));
    MappingExecutionRegion region;
    region.id = impl.id;
    region.implementation = impl.id;
    region.placement = impl.device;
    region.submissionResource = impl.resource;
    region.beginEvent = r;
    region.endEvent = r + 1;
    region.memberEvents = {r};
    region.sourceTasks = impl.operators;
    region.submissionCount = impl.launchCount;
    region.endpointMemory = impl.memory;
    region.invocationLifecycle =
        impl.visibilityPolicy == "device-resident"
            ? "device-command"
            : (impl.device == "npu" ? "host-device" : "host-only");
    region.inputCopies = graph.events.back().reads;
    region.outputCopies = graph.events.back().writes;
    if (impl.device == "npu")
      region.deviceStages = {r};
    if (r)
      region.predecessorRegions = {r - 1};
    graph.executionRegions.push_back(std::move(region));
    for (const auto &kind : lifecycleKinds(impl)) {
      MappingRegionEvent activity;
      activity.kind = kind;
      activity.region = r;
      activity.resource = impl.resource;
      if (!graph.regionEvents.empty())
        activity.dependencies = {graph.regionEvents.size() - 1};
      if (kind == "visible-to-device")
        activity.copies = graph.executionRegions.back().inputCopies;
      if (kind == "visible-to-host")
        activity.copies = graph.executionRegions.back().outputCopies;
      graph.regionEvents.push_back(std::move(activity));
    }
  }
  for (auto &copy : graph.copies) {
    if (!copy.users.empty())
      copy.lastUsers = {copy.users.back()};
    auto &allocation = graph.allocations[copy.allocation];
    allocation.endPhase = std::max(allocation.endPhase, copy.releasedAfter);
  }
  return checkProgramOperatorGraph(graph, error);
}

bool orbit::taskflow::checkProgramOperatorGraph(const ProgramGraph &g,
                                                std::string &error) {
  if (g.operatorMapping && hasProgramOperatorPackedInputs(*g.operatorMapping))
    return checkProgramOperatorPackedGraph(g, error);
  if (!g.operatorSource || !g.operatorSelection || g.source.function ||
      !g.source.tasks.empty() || !g.source.storages.empty())
    return fail(error, "missing/mixed first-class operator source");
  const auto &s = *g.operatorSource;
  const auto &selection = *g.operatorSelection;
  std::vector<size_t> owners;
  if (!checkProgramOperatorSource(s, error) ||
      !checkSelection(s, selection, owners, error))
    return false;
  StaticConfig target;
  target.domains = g.domains;
  target.memories = g.memories;
  target.resources = g.resources;
  target.links = g.links;
  if (!validateMemoryArchitecture(target, error))
    return false;
  if (g.bodyFingerprint != s.sourceId ||
      g.instances.size() != s.operators.size() ||
      g.placements.size() != s.operators.size() ||
      g.events.size() != selection.regions.size() ||
      g.executionRegions.size() != selection.regions.size() ||
      g.regions.size() != g.versions.size() ||
      g.regions.size() != g.copies.size())
    return fail(error, "source/implementation coverage differs");
  std::vector<int64_t> producer(s.values.size(), -1);
  std::set<size_t> expectedValues;
  for (size_t o = 0; o < s.operators.size(); ++o)
    for (size_t v : s.operators[o].outputs)
      producer[v] = owners[o];
  for (const auto &impl : selection.regions) {
    expectedValues.insert(impl.inputs.begin(), impl.inputs.end());
    expectedValues.insert(impl.outputs.begin(), impl.outputs.end());
  }
  for (size_t v :
       std::vector<size_t>(expectedValues.begin(), expectedValues.end())) {
    while (s.values[v].view || s.values[v].allocationView ||
           s.values[v].stateAliasOf) {
      v = s.values[v].view
              ? s.values[v].view->base
              : (s.values[v].allocationView ? s.values[v].allocationView->base
                                            : *s.values[v].stateAliasOf);
      expectedValues.insert(v);
    }
  }
  for (auto it = expectedValues.begin(); it != expectedValues.end();)
    if (s.values[*it].role == "workspace")
      it = expectedValues.erase(it);
    else
      ++it;
  if (g.regions.size() != expectedValues.size())
    return fail(error, "invented or missing implementation ABI value");
  std::map<size_t, size_t> copyForValue;
  std::set<size_t> allocations;
  size_t initialCount = 0, finalCount = 0;
  for (size_t c = 0; c < g.regions.size(); ++c) {
    size_t v = g.regions[c].storage;
    if (!expectedValues.count(v) || !copyForValue.emplace(v, c).second)
      return fail(error, "duplicate/fabricated ABI tensor");
    const auto &value = s.values[v];
    const auto &region = g.regions[c];
    const auto &version = g.versions[c];
    const auto &copy = g.copies[c];
    if (region.sizes != value.shape ||
        region.offsets != std::vector<int64_t>(value.shape.size(), 0) ||
        version.region != c || version.producer != producer[v] ||
        copy.version != c || copy.createdBy != producer[v] ||
        copy.allocation >= g.allocations.size() ||
        copy.availableFrom != (producer[v] < 0 ? 0 : size_t(producer[v])) ||
        copy.releasedAfter <= copy.availableFrom ||
        copy.releasedAfter > g.events.size() || copy.layout != "row_major")
      return fail(error, "wrong ABI region/version/copy/lifetime");
    const auto &allocation = g.allocations[copy.allocation];
    size_t root = physicalRoot(s, v);
    if (allocation.storage != root ||
        allocation.bytes != bytes(s.values[root]) ||
        allocation.addressing != "external-abi" ||
        allocation.beginPhase > copy.availableFrom ||
        allocation.endPhase < copy.releasedAfter ||
        allocation.endPhase > g.events.size())
      return fail(error,
                  "wrong physical ABI allocation or state/view lifetime");
    if (root != v && s.values[root].role != "workspace" &&
        (!copyForValue.count(root) ||
         copy.allocation != g.copies[copyForValue.at(root)].allocation))
      return fail(error, "state/view allocates independent physical storage");
    auto memory =
        std::find_if(g.memories.begin(), g.memories.end(),
                     [&](const auto &m) { return m.id == allocation.memory; });
    if (memory == g.memories.end())
      return fail(error, "ABI memory absent from architecture");
    allocations.insert(copy.allocation);
    if (value.role == "input") {
      ++initialCount;
      auto found = g.inputLocations.find(v);
      if (found == g.inputLocations.end() ||
          found->second != allocation.memory ||
          (!value.initialMemory.empty() &&
           value.initialMemory != allocation.memory))
        return fail(error, "wrong initial copy boundary");
    }
    if (value.role == "output") {
      ++finalCount;
      auto found = g.outputLocations.find(v);
      if (found == g.outputLocations.end() ||
          found->second != allocation.memory ||
          copy.releasedAfter != g.events.size())
        return fail(error, "wrong final visibility boundary");
    }
  }
  if (allocations.size() != g.allocations.size() ||
      initialCount != g.inputLocations.size() ||
      finalCount != g.outputLocations.size())
    return fail(error, "invented allocations or external boundaries");
  for (size_t o = 0; o < s.operators.size(); ++o) {
    const auto &instance = g.instances[o];
    const auto &impl = selection.regions[owners[o]];
    const auto &placement = g.placements[o];
    if (instance.task != o || instance.domainTask != o ||
        instance.event != owners[o] || instance.phase != owners[o] ||
        instance.kind != "external-source-operator" ||
        !instance.domain.empty() || instance.lanes != 1 ||
        instance.parallelLoop != -1 || instance.resourceCount != 1 ||
        !instance.resourceShape.empty() || placement.resourceCount != 1 ||
        !placement.resourceShape.empty() ||
        instance.implementation != impl.id || placement.task != o ||
        placement.resource != impl.resource ||
        placement.memory != impl.memory || placement.implementation != impl.id)
      return fail(error,
                  "source operator missing/wrong implementation instance");
  }
  size_t edgeCursor = 0, lifecycleCursor = 0;
  std::vector<std::vector<size_t>> users(g.copies.size()),
      consumerEdges(g.copies.size());
  for (size_t r = 0; r < selection.regions.size(); ++r) {
    const auto &impl = selection.regions[r];
    const auto &event = g.events[r];
    const auto &region = g.executionRegions[r];
    auto resource =
        std::find_if(g.resources.begin(), g.resources.end(),
                     [&](const auto &a) { return a.id == impl.resource; });
    if (resource == g.resources.end() || resource->kind != "compute" ||
        resource->device != impl.device)
      return fail(error, "wrong implementation compute resource");
    std::vector<size_t> inputs, outputs, edges;
    for (size_t v : impl.inputs) {
      size_t c = copyForValue.at(v);
      inputs.push_back(c);
      users[c].push_back(r);
      consumerEdges[c].push_back(edgeCursor);
      if (g.copies[c].availableFrom > r || g.copies[c].releasedAfter <= r ||
          edgeCursor >= g.dataEdges.size())
        return fail(error, "missing/expired data supply");
      const auto &edge = g.dataEdges[edgeCursor];
      if (edge.producer != producer[v] || edge.consumer != r ||
          edge.region != c || edge.version != c || edge.copy != c ||
          edge.requirement.memory != portMemory(impl, v, true) ||
          edge.requirement.dtype != s.values[v].dtype ||
          edge.requirement.layout != "row_major" ||
          g.allocations[g.copies[c].allocation].memory !=
              portMemory(impl, v, true))
        return fail(error, "ABI data edge requires wrong value/copy/memory");
      edges.push_back(edgeCursor++);
    }
    for (size_t v : impl.outputs) {
      size_t c = copyForValue.at(v);
      outputs.push_back(c);
      if (g.allocations[g.copies[c].allocation].memory !=
          portMemory(impl, v, false))
        return fail(error, "wrong output physical memory");
    }
    std::vector<size_t> predecessor =
        r ? std::vector<size_t>{r - 1} : std::vector<size_t>{};
    if (event.kind != "implementation" || event.phase != r ||
        event.task != impl.operators.front() ||
        event.resource != impl.resource || event.memory != impl.memory ||
        event.reads != inputs || event.writes != outputs ||
        event.dependencies != predecessor || event.dataEdges != edges ||
        !event.orderEdges.empty())
      return fail(error, "wrong actual invocation event");
    std::string lifecycle =
        impl.visibilityPolicy == "device-resident"
            ? "device-command"
            : (impl.device == "npu" ? "host-device" : "host-only");
    if (region.id != impl.id || region.implementation != impl.id ||
        region.placement != impl.device ||
        region.submissionResource != impl.resource || region.beginEvent != r ||
        region.endEvent != r + 1 ||
        region.memberEvents != std::vector<size_t>{r} ||
        region.sourceTasks != impl.operators || region.inputCopies != inputs ||
        region.outputCopies != outputs ||
        region.endpointMemory != impl.memory ||
        region.submissionCount != impl.launchCount ||
        region.invocationLifecycle != lifecycle ||
        region.predecessorRegions != predecessor ||
        region.deviceStages != (impl.device == "npu" ? std::vector<size_t>{r}
                                                     : std::vector<size_t>{}))
      return fail(error, "region coverage/lifecycle fabricated");
    for (const auto &kind : lifecycleKinds(impl)) {
      if (lifecycleCursor >= g.regionEvents.size())
        return fail(error, "missing invocation lifecycle event");
      const auto &activity = g.regionEvents[lifecycleCursor];
      auto dependencies = lifecycleCursor
                              ? std::vector<size_t>{lifecycleCursor - 1}
                              : std::vector<size_t>{};
      auto copies =
          kind == "visible-to-device"
              ? inputs
              : (kind == "visible-to-host" ? outputs : std::vector<size_t>{});
      if (activity.region != r || activity.kind != kind ||
          activity.resource != impl.resource ||
          activity.dependencies != dependencies || activity.copies != copies)
        return fail(error,
                    "fake host sync/readback or missing invocation ordering");
      ++lifecycleCursor;
    }
  }
  if (edgeCursor != g.dataEdges.size() ||
      lifecycleCursor != g.regionEvents.size() || !g.orderEdges.empty())
    return fail(error, "fabricated physical activities");
  for (size_t c = 0; c < g.copies.size(); ++c) {
    std::vector<size_t> last;
    if (!users[c].empty())
      last = {users[c].back()};
    if (g.copies[c].users != users[c] ||
        g.copies[c].consumerEdges != consumerEdges[c] ||
        g.copies[c].lastUsers != last)
      return fail(error, "incorrect consumer/last-user copy witness");
  }
  if (g.operatorMapping)
    return checkProgramOperatorMapping(g, error);
  if (!g.schedule.events.empty() || !g.schedule.eventDurationsMs.empty() ||
      g.estimatedMakespanMs)
    return fail(error, "unproved source-operator timing allocation");
  return true;
}

j::Value orbit::taskflow::serializeProgramOperatorSelection(
    const ProgramOperatorSelection &selection) {
  j::Array regions;
  for (const auto &impl : selection.regions) {
    auto indices = [](const std::vector<size_t> &values) {
      j::Array array;
      for (size_t v : values)
        array.push_back(int64_t(v));
      return array;
    };
    j::Object inputs, outputs;
    for (const auto &p : impl.inputLocations)
      inputs[std::to_string(p.first)] = p.second;
    for (const auto &p : impl.outputLocations)
      outputs[std::to_string(p.first)] = p.second;
    regions.push_back(j::Object{
        {"id", impl.id},
        {"source_id", impl.sourceId},
        {"artifact_identity", impl.artifactIdentity},
        {"entrypoint", impl.entrypoint},
        {"device", impl.device},
        {"resource", impl.resource},
        {"memory", impl.memory},
        {"source_operators", indices(impl.operators)},
        {"inputs", indices(impl.inputs)},
        {"outputs", indices(impl.outputs)},
        {"wall_ms", impl.wallMs ? j::Value(*impl.wallMs) : j::Value(nullptr)},
        {"executable", impl.executable},
        {"launch_count", impl.launchCount},
        {"cost_boundary", impl.costBoundary},
        {"transport_ownership", impl.transportOwnership},
        {"visibility_policy", impl.visibilityPolicy},
        {"input_locations", std::move(inputs)},
        {"output_locations", std::move(outputs)}});
  }
  return j::Object{{"schema", "orbit.program-operator-selection.v1"},
                   {"regions", std::move(regions)}};
}

j::Value
orbit::taskflow::summarizeProgramOperatorGraph(const ProgramGraph &graph) {
  double wall = 0;
  bool complete = true;
  int64_t launches = 0;
  j::Array unknown;
  if (!graph.operatorSelection)
    return j::Object{{"status", "UNKNOWN"}};
  for (const auto &impl : graph.operatorSelection->regions) {
    if (impl.wallMs) {
      wall += *impl.wallMs;
      if (!std::isfinite(wall))
        complete = false;
    } else {
      complete = false;
      unknown.push_back(impl.id);
    }
    if (impl.launchCount > INT64_MAX - launches)
      complete = false;
    else
      launches += impl.launchCount;
  }
  return j::Object{
      {"kind", "external-implementation-cover"},
      {"source_operators", int64_t(graph.operatorSource->operators.size())},
      {"physical_invocations", int64_t(graph.events.size())},
      {"actual_npu_launches", launches},
      {"estimated_ms", complete ? j::Value(wall) : j::Value(nullptr)},
      {"unknown_implementations", std::move(unknown)},
      {"boundary",
       "inclusive-region-wall; internal operator durations not inferred"},
      {"proof_scope", "external source/artifact/interface contracts; not "
                      "native MLIR arithmetic proof"}};
}

bool orbit::taskflow::parseProgramOperatorImplementations(
    const ProgramOperatorSource &source, const j::Value &document,
    std::vector<ProgramOperatorImplementation> &implementations,
    std::string &error) {
  auto *root = document.getAsObject();
  if (!root ||
      root->getString("schema") !=
          "orbit.program-operator-implementations.v1" ||
      root->getString("source_id") != source.sourceId)
    return fail(error, "implementation catalogue/source identity mismatch");
  auto *entries = root->getArray("implementations");
  if (!entries)
    return fail(error, "missing implementation catalogue");
  std::map<std::string, size_t> opIds, valueIds;
  for (size_t i = 0; i < source.operators.size(); ++i)
    opIds[source.operators[i].id] = i;
  for (size_t i = 0; i < source.values.size(); ++i)
    valueIds[source.values[i].id] = i;
  std::set<std::string> implementationIds;
  for (const auto &entry : *entries) {
    auto *o = entry.getAsObject();
    ProgramOperatorImplementation impl;
    if (!o || !text(*o, "id", impl.id) ||
        !implementationIds.insert(impl.id).second ||
        !text(*o, "source_id", impl.sourceId) ||
        !text(*o, "artifact_identity", impl.artifactIdentity) ||
        !text(*o, "entrypoint", impl.entrypoint) ||
        !text(*o, "device", impl.device) ||
        !text(*o, "resource", impl.resource) ||
        !text(*o, "memory", impl.memory) ||
        !text(*o, "cost_boundary", impl.costBoundary) ||
        !text(*o, "transport_ownership", impl.transportOwnership) ||
        !text(*o, "visibility_policy", impl.visibilityPolicy))
      return fail(error, "incomplete local implementation binding");
    auto ids = [&](llvm::StringRef numeric, llvm::StringRef named,
                   const auto &table, std::vector<size_t> &result) {
      if (o->getArray(named))
        return references(*o, named, table, result);
      std::vector<int64_t> values;
      if (!integers(*o, numeric, values))
        return false;
      std::set<size_t> seen;
      for (int64_t v : values) {
        if (v < 0 || size_t(v) >= table.size() || !seen.insert(v).second)
          return false;
        result.push_back(v);
      }
      return true;
    };
    if (!ids("source_operators", "operators", opIds, impl.operators) ||
        !ids("inputs", "input_values", valueIds, impl.inputs) ||
        !ids("outputs", "output_values", valueIds, impl.outputs))
      return fail(error, "invalid implementation source/ABI references");
    auto executable = o->getBoolean("executable");
    auto launches = o->getInteger("launch_count");
    if (!executable || !launches)
      return fail(error, "missing actual executable/launch contract");
    impl.executable = *executable;
    impl.launchCount = *launches;
    auto *wall = o->get("wall_ms");
    if (!wall)
      return fail(error, "cost must be explicit number or null");
    if (auto price = wall->getAsNumber()) {
      if (!std::isfinite(*price) || *price < 0)
        return fail(error, "invalid implementation wall cost");
      impl.wallMs = *price;
    } else if (wall->kind() != j::Value::Null)
      return fail(error, "cost must be finite number or null");
    auto locations = [&](llvm::StringRef name,
                         std::map<size_t, std::string> &result) {
      auto *entries = o->getObject(name);
      if (!entries)
        return true;
      for (const auto &p : *entries) {
        size_t v = 0;
        std::string key = p.first.str();
        auto found = valueIds.find(key);
        if (found != valueIds.end())
          v = found->second;
        else {
          uint64_t number = 0;
          if (llvm::StringRef(key).getAsInteger(10, number) ||
              number >= source.values.size())
            return false;
          v = number;
        }
        auto memory = p.second.getAsString();
        if (!memory || memory->empty() ||
            !result.emplace(v, memory->str()).second)
          return false;
      }
      return true;
    };
    if (!locations("input_locations", impl.inputLocations) ||
        !locations("output_locations", impl.outputLocations))
      return fail(error, "invalid implementation memory port");
    implementations.push_back(std::move(impl));
  }
  return true;
}

bool orbit::taskflow::certifyProgramOperatorChoices(
    const ProgramOperatorSource &source,
    const std::vector<ProgramOperatorImplementation> &implementations,
    const std::vector<ProgramOperatorChoiceGroup> &groups, j::Value &receipt,
    std::string &error) {
  std::vector<int64_t> producer(source.values.size(), -1);
  std::vector<std::vector<size_t>> readers(source.values.size());
  for (size_t o = 0; o < source.operators.size(); ++o) {
    for (size_t v : source.operators[o].outputs)
      producer[v] = o;
    for (size_t v : source.operators[o].inputs)
      readers[v].push_back(o);
  }
  for (const auto &impl : implementations) {
    if (impl.sourceId != source.sourceId || !impl.executable ||
        impl.id.empty() || impl.artifactIdentity.empty() ||
        impl.entrypoint.empty() || impl.resource.empty() ||
        impl.memory.empty() || impl.operators.empty() ||
        sorted(impl.operators) != impl.operators ||
        impl.operators.back() >= source.operators.size() ||
        impl.operators.back() - impl.operators.front() + 1 !=
            impl.operators.size() ||
        (impl.device != "cpu" && impl.device != "npu") ||
        impl.costBoundary != "inclusive-region-wall" ||
        impl.transportOwnership != "implementation-inclusive" ||
        (impl.visibilityPolicy != "host-materialized" &&
         impl.visibilityPolicy != "device-resident") ||
        (impl.device == "cpu" ? impl.launchCount != 0 : impl.launchCount < 1) ||
        (impl.wallMs && (!std::isfinite(*impl.wallMs) || *impl.wallMs < 0)))
      return fail(error, "invalid local executable implementation certificate");
    auto contains = [&](int64_t o) {
      return o >= 0 && std::binary_search(impl.operators.begin(),
                                          impl.operators.end(), size_t(o));
    };
    std::vector<size_t> inputs, outputs;
    for (size_t o : impl.operators) {
      for (size_t v : source.operators[o].inputs)
        if (!contains(producer[v]))
          inputs.push_back(v);
      for (size_t v : source.operators[o].outputs) {
        bool external = source.values[v].role == "output";
        for (size_t reader : readers[v])
          external |= !contains(reader);
        if (external)
          outputs.push_back(v);
      }
    }
    if (sorted(inputs) != sorted(impl.inputs) ||
        sorted(outputs) != sorted(impl.outputs) ||
        sorted(impl.inputs).size() != impl.inputs.size() ||
        sorted(impl.outputs).size() != impl.outputs.size())
      return fail(error,
                  "local implementation has incorrect source boundary ABI");
  }
  std::vector<size_t> sourceOwners(source.operators.size(), SIZE_MAX);
  j::Array groupEvidence;
  std::set<std::string> groupIds;
  std::map<size_t, std::string> globalProduced, globalRequired;
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto &group = groups[g];
    if (group.id.empty() || !groupIds.insert(group.id).second ||
        group.choices.empty() || group.choiceIds.size() != group.choices.size())
      return fail(error, "invalid local choice group");
    std::vector<size_t> referenceMembers;
    std::map<size_t, std::string> referenceInputs, referenceOutputs;
    std::set<std::string> choiceIds;
    for (size_t c = 0; c < group.choices.size(); ++c) {
      if (!choiceIds.insert(group.choiceIds[c]).second ||
          group.choices[c].empty())
        return fail(error, "duplicate/empty local choice");
      std::vector<size_t> members;
      std::map<size_t, size_t> choiceOwners;
      for (size_t index : group.choices[c]) {
        if (index >= implementations.size())
          return fail(error,
                      "local choice references nonexistent implementation");
        for (size_t op : implementations[index].operators) {
          if (!choiceOwners.emplace(op, index).second)
            return fail(error, "local choice overlaps source coverage");
          members.push_back(op);
        }
      }
      members = sorted(std::move(members));
      std::map<size_t, std::string> inputs, outputs;
      for (size_t index : group.choices[c]) {
        const auto &impl = implementations[index];
        for (size_t v : impl.inputs) {
          if (producer[v] < 0 || !choiceOwners.count(size_t(producer[v])))
            inputs[v] = portMemory(impl, v, true);
          else if (portMemory(
                       implementations[choiceOwners.at(size_t(producer[v]))], v,
                       false) != portMemory(impl, v, true))
            return fail(
                error,
                "local choice requires undeclared internal memory transition");
        }
        for (size_t v : impl.outputs) {
          bool external = source.values[v].role == "output";
          for (size_t reader : readers[v])
            external |= !choiceOwners.count(reader);
          if (external)
            outputs[v] = portMemory(impl, v, false);
        }
      }
      if (c == 0) {
        referenceMembers = members;
        referenceInputs = inputs;
        referenceOutputs = outputs;
      } else if (members != referenceMembers || inputs != referenceInputs ||
                 outputs != referenceOutputs)
        return fail(
            error,
            "local choices change source coverage or cross-group memory ABI");
    }
    for (size_t op : referenceMembers) {
      if (sourceOwners[op] != SIZE_MAX)
        return fail(error, "choice groups overlap source operators");
      sourceOwners[op] = g;
    }
    for (auto input : referenceInputs) {
      auto prior = globalRequired.emplace(input);
      if (!prior.second && prior.first->second != input.second)
        return fail(error,
                    "cross-group consumers disagree on physical input memory");
    }
    globalProduced.insert(referenceOutputs.begin(), referenceOutputs.end());
    groupEvidence.push_back(
        j::Object{{"id", group.id},
                  {"source_operators", int64_t(referenceMembers.size())},
                  {"local_choices", int64_t(group.choices.size())},
                  {"invariant_external_abi", true},
                  {"exact_source_cover", true}});
  }
  if (std::find(sourceOwners.begin(), sourceOwners.end(), SIZE_MAX) !=
      sourceOwners.end())
    return fail(error, "choice product omits source operators");
  for (auto required : globalRequired) {
    if (producer[required.first] >= 0) {
      auto found = globalProduced.find(required.first);
      if (found == globalProduced.end() || found->second != required.second)
        return fail(error, "cross-group producer/consumer memory mismatch");
    } else if (!source.values[required.first].initialMemory.empty() &&
               source.values[required.first].initialMemory != required.second)
      return fail(error,
                  "initial copy location differs from certified choice ABI");
  }
  receipt = j::Object{
      {"schema", "orbit.program-operator-choice-certificate.v1"},
      {"source_id", source.sourceId},
      {"source_operators", int64_t(source.operators.size())},
      {"groups", std::move(groupEvidence)},
      {"candidate_legality",
       "product of exact local source covers with invariant cross-group ABI; "
       "source order/state/version semantics checked before certification"},
      {"cost_ownership", "actual implementation invocation/transport fees; no "
                         "adjacency-based fusion"},
      {"physical_capacity_scope",
       "ABI allocations checked on selected graph; external implementation "
       "private/preparation admission remains backend-owned"}};
  return true;
}

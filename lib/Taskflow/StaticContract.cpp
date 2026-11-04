#include "Taskflow/StaticMapping.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
using namespace orbit::taskflow;
namespace {
using O = llvm::json::Object;
using V = llvm::json::Value;
using A = llvm::json::Array;
const O &obj(const V &v) {
  auto p = v.getAsObject();
  if (!p)
    throw std::runtime_error("expected JSON object");
  return *p;
}
const V &field(const O &o, llvm::StringRef k) {
  auto p = o.get(k);
  if (!p)
    throw std::runtime_error("missing field " + k.str());
  return *p;
}
std::string str(const O &o, llvm::StringRef k, std::string d = "") {
  auto p = o.get(k);
  if (!p)
    return d;
  auto s = p->getAsString();
  if (!s)
    throw std::runtime_error("expected string " + k.str());
  return s->str();
}
const A &arr(const O &o, llvm::StringRef k) {
  auto p = o.getArray(k);
  if (!p)
    throw std::runtime_error("expected array " + k.str());
  return *p;
}
double num(const O &o, llvm::StringRef k, bool required = true, double d = 0) {
  auto p = o.get(k);
  if (!p && !required)
    return d;
  if (!p)
    throw std::runtime_error("missing cost/number " + k.str());
  auto x = p->getAsNumber();
  if (!x || !std::isfinite(*x) || *x < 0)
    throw std::runtime_error("invalid nonnegative number " + k.str());
  return *x;
}
int64_t integer(const O &o, llvm::StringRef k, bool required = true,
                int64_t d = 0) {
  auto p = o.get(k);
  if (!p && !required)
    return d;
  if (!p)
    throw std::runtime_error("missing integer " + k.str());
  auto x = p->getAsInteger();
  if (!x || *x < 0)
    throw std::runtime_error("invalid nonnegative integer " + k.str());
  return *x;
}
bool flag(const O &o, llvm::StringRef k, bool d = false) {
  auto p = o.get(k);
  if (!p)
    return d;
  auto x = p->getAsBoolean();
  if (!x)
    throw std::runtime_error("expected bool " + k.str());
  return *x;
}
std::vector<std::string> strings(const O &o, llvm::StringRef k,
                                 bool required = true) {
  std::vector<std::string> r;
  if (!o.get(k) && !required)
    return r;
  for (auto &v : arr(o, k)) {
    auto x = v.getAsString();
    if (!x)
      throw std::runtime_error("non-string array element");
    r.push_back(x->str());
  }
  return r;
}
std::map<std::string, std::string> stringMap(const O &o, llvm::StringRef k,
                                             bool required = true) {
  std::map<std::string, std::string> r;
  if (!o.get(k) && !required)
    return r;
  for (auto &v : obj(field(o, k))) {
    auto x = v.second.getAsString();
    if (!x)
      throw std::runtime_error("non-string map value");
    r[v.first.str()] = x->str();
  }
  return r;
}
std::map<std::string, int64_t> demands(const O &o, llvm::StringRef k,
                                       bool required = true) {
  std::map<std::string, int64_t> r;
  if (!o.get(k) && !required)
    return r;
  for (auto &v : obj(field(o, k))) {
    auto x = v.second.getAsInteger();
    if (!x || *x <= 0)
      throw std::runtime_error(
          "resource demands/multiples must be positive integers");
    r[v.first.str()] = *x;
  }
  return r;
}
Range range(const V &v) {
  auto a = v.getAsArray();
  if (!a || a->size() != 2)
    throw std::runtime_error("range requires two bounds");
  auto lo = (*a)[0].getAsInteger(), hi = (*a)[1].getAsInteger();
  if (!lo || !hi || *lo < 0 || *hi < *lo)
    throw std::runtime_error("invalid range");
  return {*lo, *hi};
}
StaticProvenance provenance(const O &o) {
  auto &p = obj(field(o, "provenance"));
  StaticProvenance r{str(p, "kind"), str(p, "source")};
  if ((r.kind != "synthetic" && r.kind != "theoretical" &&
       r.kind != "calibrated" && r.kind != "measured") ||
      r.source.empty())
    throw std::runtime_error("explicit cost provenance required");
  return r;
}
A ss(const std::vector<std::string> &v) {
  A a;
  for (auto &s : v)
    a.push_back(s);
  return a;
}
O sm(const std::map<std::string, std::string> &v) {
  O o;
  for (auto &p : v)
    o[p.first] = p.second;
  return o;
}
O dm(const std::map<std::string, int64_t> &v) {
  O o;
  for (auto &p : v)
    o[p.first] = p.second;
  return o;
}
V prov(const StaticProvenance &p) {
  return O{{"kind", p.kind}, {"source", p.source}};
}
A rr(Range r) { return A{r.lo, r.hi}; }
V access(const StaticAccess &a) {
  return O{{"buffer", a.buffer},
           {"region", A{rr(a.m), rr(a.n)}},
           {"allocation", a.allocation},
           {"version", a.version}};
}
A accesses(const std::vector<StaticAccess> &v) {
  A a;
  for (auto &r : v)
    a.push_back(access(r));
  return a;
}
StaticAccess parseAccess(const V &v) {
  auto &o = obj(v);
  auto &r = arr(o, "region");
  if (r.size() != 2)
    throw std::runtime_error("rank-two access expected");
  return {str(o, "buffer"), str(o, "allocation"), str(o, "version"),
          range(r[0]), range(r[1])};
}
std::vector<StaticAccess> parseAccesses(const O &o, llvm::StringRef k) {
  std::vector<StaticAccess> a;
  for (auto &v : arr(o, k))
    a.push_back(parseAccess(v));
  return a;
}
void unique(std::set<std::string> &ids, const std::string &id) {
  if (id.empty() || !ids.insert(id).second)
    throw std::runtime_error("missing/duplicate id " + id);
}
V compute(const Compute &c) {
  O o{{"kind", c.kind},
      {"inputs", ss(c.inputs)},
      {"output", c.output},
      {"m", rr(c.m)},
      {"n", rr(c.n)}};
  if (c.kind == "matmul") {
    o["k"] = rr(c.k);
    o["accumulate"] = c.accumulate;
  }
  return o;
}
Compute parseCompute(const V &v) {
  auto &o = obj(v);
  Compute c;
  c.kind = str(o, "kind");
  c.inputs = strings(o, "inputs");
  c.output = str(o, "output");
  c.m = range(field(o, "m"));
  c.n = range(field(o, "n"));
  if (c.kind == "matmul") {
    c.k = range(field(o, "k"));
    c.accumulate = flag(o, "accumulate");
  }
  return c;
}
V step(const StaticStep &s) {
  return O{{"id", s.id},
           {"task_id", s.taskId},
           {"operation_index", s.operationIndex},
           {"operation", compute(s.operation)},
           {"depends_on", ss(s.dependsOn)}};
}
} // namespace
// LLVM JSON canonical output recursively sorts object keys and prints doubles
// with %.17g. Unlike Python shortest-float dumps, this convention round-trips
// identically through LLVM JSON; independent tests implement it separately.
std::string orbit::taskflow::staticFingerprint(const llvm::json::Value &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
  auto d = llvm::SHA256::hash(llvm::arrayRefFromStringRef(s));
  return llvm::toHex(d, true);
}
bool orbit::taskflow::parseStaticConfig(const V &v, StaticConfig &c,
                                        std::string &e) {
  try {
    c = StaticConfig{};
    auto &o = obj(v);
    if (str(o, "schema") != "orbit.static.config.v1")
      throw std::runtime_error("unsupported static config schema");
    c.original = v;
    auto &hw = obj(field(o, "hardware"));
    std::set<std::string> ids;
    std::map<std::string, StaticResource> resources;
    std::map<std::string, StaticMemory> memories;
    for (auto &v : arr(hw, "resources")) {
      auto &o = obj(v);
      StaticResource r{str(o, "id"), str(o, "kind"), str(o, "device"),
                       integer(o, "capacity")};
      unique(ids, r.id);
      if (r.capacity <= 0 ||
          (r.kind != "compute" && r.kind != "communication") ||
          (r.kind == "compute" && r.device.empty()))
        throw std::runtime_error("invalid resource kind/device/capacity");
      resources[r.id] = r;
      c.resources.push_back(r);
    }
    ids.clear();
    for (auto &v : arr(hw, "memories")) {
      auto &o = obj(v);
      StaticMemory m{str(o, "id"), integer(o, "capacity_bytes"),
                     strings(o, "accessible_devices")};
      unique(ids, m.id);
      if (m.capacityBytes <= 0 || m.accessibleDevices.empty())
        throw std::runtime_error("invalid memory capacity/access");
      memories[m.id] = m;
      c.memories.push_back(m);
    }
    if (resources.empty() || memories.empty())
      throw std::runtime_error("hardware resources and memories required");
    auto transfers = [&](llvm::StringRef key,
                         std::vector<StaticTransfer> &out) {
      ids.clear();
      auto a = hw.getArray(key);
      if (!a)
        return;
      for (auto &v : *a) {
        auto &o = obj(v);
        StaticTransfer t;
        t.id = str(o, "id");
        unique(ids, t.id);
        t.resource = str(o, "resource");
        if (!resources.count(t.resource))
          throw std::runtime_error("unknown transfer resource");
        t.latencyMs = num(o, "latency_ms");
        t.provenance = provenance(o);
        t.temporaryBytes = integer(o, "temporary_bytes", false);
        t.source = str(o, "source");
        t.target = str(o, "target");
        t.memory = str(o, "memory");
        t.sourceLayout = str(o, "source_layout");
        t.targetLayout = str(o, "target_layout");
        t.sourceDtype = str(o, "source_dtype");
        t.targetDtype = str(o, "target_dtype");
        t.sourceDevice = str(o, "source_device");
        t.targetDevice = str(o, "target_device");
        if (key != "synchronizations") {
          t.bandwidthBytesPerMs = num(o, "bandwidth_bytes_per_ms");
          if (t.bandwidthBytesPerMs <= 0)
            throw std::runtime_error("nonpositive bandwidth");
          t.minBytes = integer(o, "min_bytes", false);
          t.maxBytes = integer(o, "max_bytes", false, INT64_MAX);
          if (t.minBytes > t.maxBytes)
            throw std::runtime_error("invalid communication validity range");
        }
        if (key == "links" &&
            (!memories.count(t.source) || !memories.count(t.target)))
          throw std::runtime_error("link unknown memory");
        if (key == "conversions" &&
            (!memories.count(t.memory) || t.sourceLayout.empty() ||
             t.targetLayout.empty() || t.sourceDtype != "f32" ||
             t.targetDtype != "f32"))
          throw std::runtime_error(
              "unsupported conversion memory/layout/dtype");
        if (key == "synchronizations") {
          if (!memories.count(t.memory))
            throw std::runtime_error("sync unknown memory");
          auto &devices = memories[t.memory].accessibleDevices;
          if (std::find(devices.begin(), devices.end(), t.sourceDevice) ==
                  devices.end() ||
              std::find(devices.begin(), devices.end(), t.targetDevice) ==
                  devices.end())
            throw std::runtime_error("sync devices cannot access memory");
        }
        out.push_back(t);
      }
    };
    transfers("links", c.links);
    transfers("conversions", c.conversions);
    transfers("synchronizations", c.synchronizations);
    ids.clear();
    for (auto &v : arr(o, "implementations")) {
      auto &o = obj(v);
      StaticImplementation i;
      i.id = str(o, "id");
      unique(ids, i.id);
      i.version = str(o, "version");
      i.device = str(o, "device");
      i.memory = str(o, "memory");
      i.layout = str(o, "layout");
      i.dtype = str(o, "dtype");
      i.numericalContract = str(o, "numerical_contract");
      i.operations = strings(o, "operations");
      i.resources = demands(o, "resources");
      i.temporaryBytes = integer(o, "temporary_bytes");
      if (i.version.empty() || i.operations.empty() || i.layout.empty() ||
          i.dtype != "f32" || i.numericalContract.empty() ||
          !memories.count(i.memory))
        throw std::runtime_error("invalid implementation contract");
      auto &accessible = memories[i.memory].accessibleDevices;
      if (std::find(accessible.begin(), accessible.end(), i.device) ==
          accessible.end())
        throw std::runtime_error("implementation cannot access memory");
      bool own = false;
      for (auto &r : i.resources) {
        if (!resources.count(r.first) || r.second > resources[r.first].capacity)
          throw std::runtime_error("invalid implementation resource demand");
        own |= resources[r.first].kind == "compute" &&
               resources[r.first].device == i.device;
      }
      if (!own)
        throw std::runtime_error(
            "implementation requires compute resource on its device");
      auto &cost = obj(field(o, "cost"));
      i.cost.kind = str(cost, "kind");
      i.cost.scope = str(cost, "scope");
      if (i.cost.kind != "affine" || i.cost.scope != "compute_only")
        throw std::runtime_error(
            "opaque graph costs cannot become compute_only");
      i.cost.fixedMs = num(cost, "fixed_ms");
      i.cost.perElementMs = num(cost, "per_element_ms");
      i.cost.perFlopMs = num(cost, "per_flop_ms");
      i.cost.provenance = provenance(cost);
      auto &bounds = obj(field(cost, "shape_range"));
      for (auto axis : {"m", "n", "k"}) {
        auto r = range(field(bounds, axis));
        if (r.lo < 1)
          throw std::runtime_error("shape bounds positive inclusive");
        i.cost.shapeRange[axis] = r;
      }
      if (auto *l = o.getObject("legality")) {
        i.fullKOnly = flag(*l, "full_k_only");
        i.overwriteZeroAccumulator = flag(*l, "overwrite_zero_accumulator");
        i.allowAccumulate = flag(*l, "accumulate", true);
        i.multiples = demands(*l, "multiples", false);
      }
      c.implementations.push_back(i);
    }
    if (auto regions = o.getArray("region_implementations")) {
      for (auto &value : *regions) {
        auto &r = obj(value);
        StaticRegionImplementation i;
        i.id = str(r, "id");
        unique(ids, i.id);
        i.version = str(r, "version");
        i.pattern = str(r, "pattern");
        i.mode = str(r, "mode");
        i.featureAxis = str(r, "feature_axis");
        i.device = str(r, "device");
        i.boundaryMemory = str(r, "boundary_memory");
        i.layout = str(r, "layout");
        i.dtype = str(r, "dtype");
        i.numericalContract = str(r, "numerical_contract");
        i.preparedState = str(r, "prepared_state");
        i.resources = demands(r, "resources");
        if (i.version.empty() || i.device != "npu" || i.layout != "row_major" ||
            i.dtype != "f32" || i.numericalContract.empty() ||
            i.preparedState != "warm_contexts_weights" ||
            (i.pattern != "matmul_pair" && i.pattern != "matmul_single") ||
            (i.mode != "chain" && i.mode != "independent") ||
            (i.pattern == "matmul_single" && i.mode != "independent") ||
            (i.pattern == "matmul_pair"
                 ? i.featureAxis != "h"
                 : (i.featureAxis != "k" && i.featureAxis != "n")) ||
            !memories.count(i.boundaryMemory))
          throw std::runtime_error(
              "unsupported region implementation contract");
        auto &accessible = memories[i.boundaryMemory].accessibleDevices;
        if (std::find(accessible.begin(), accessible.end(), "cpu") ==
            accessible.end())
          throw std::runtime_error(
              "region boundary requires CPU-accessible host memory");
        bool cpu = false, npu = false, comm = false;
        for (auto &d : i.resources) {
          if (!resources.count(d.first) ||
              d.second != resources[d.first].capacity)
            throw std::runtime_error("opaque region must conservatively "
                                     "reserve full declared resource capacity");
          auto &resource = resources[d.first];
          cpu |= resource.kind == "compute" && resource.device == "cpu";
          npu |= resource.kind == "compute" && resource.device == i.device;
          comm |= resource.kind == "communication";
        }
        if (!cpu || !npu || !comm)
          throw std::runtime_error("opaque host-to-host region needs CPU, NPU "
                                   "and communication resources");
        for (auto &w : obj(field(r, "workspace_bytes"))) {
          auto bytes = w.second.getAsInteger();
          if (!bytes || *bytes < 0 || !memories.count(w.first.str()))
            throw std::runtime_error("invalid region workspace reservation");
          i.workspaceBytes[w.first.str()] = *bytes;
        }
        if (i.workspaceBytes.empty())
          throw std::runtime_error(
              "explicit region workspace reservation required");
        if (auto legality = r.getObject("legality")) {
          i.multiples = demands(*legality, "multiples", false);
          i.powerOfTwo = strings(*legality, "power_of_two", false);
          for (auto &axis : i.powerOfTwo)
            if (axis != "m" && axis != "k" && axis != "h" && axis != "n")
              throw std::runtime_error("unknown power-of-two axis");
        }
        // The upstream epilogue requires power-of-two reduction length.
        if (std::find(i.powerOfTwo.begin(), i.powerOfTwo.end(), "k") ==
            i.powerOfTwo.end())
          i.powerOfTwo.push_back("k");
        if (i.pattern == "matmul_pair" &&
            std::find(i.powerOfTwo.begin(), i.powerOfTwo.end(), "h") ==
                i.powerOfTwo.end())
          i.powerOfTwo.push_back("h");
        auto &cost = obj(field(r, "cost"));
        if (str(cost, "kind") != "bilinear_m_h" ||
            str(cost, "scope") != "opaque_warm_host_to_host")
          throw std::runtime_error("region cost requires opaque warm boundary");
        auto coefficient = [&](const char *name) {
          auto v = field(cost, name).getAsNumber();
          if (!v || !std::isfinite(*v))
            throw std::runtime_error("unknown region model coefficient");
          return *v;
        };
        i.fixedMs = coefficient("fixed_ms");
        i.perMMs = coefficient("per_m_ms");
        i.perHMs = coefficient("per_h_ms");
        i.perMHMs = coefficient("per_mh_ms");
        i.modelId = str(cost, "model_id");
        i.trainingIds = strings(cost, "training_ids");
        i.excludedIds = strings(cost, "excluded_ids");
        i.provenance = provenance(cost);
        if (i.modelId.empty() || i.trainingIds.empty())
          throw std::runtime_error(
              "region model needs identity and training provenance");
        std::set<std::string> training;
        for (auto &id : i.trainingIds)
          unique(training, id);
        for (auto &id : i.excludedIds)
          if (training.count(id))
            throw std::runtime_error(
                "excluded validation input leaked into training ids");
        auto &shape = obj(field(cost, "shape_range"));
        for (auto axis : {"m", "h", "k", "n"}) {
          auto bounds = range(field(shape, axis));
          if (bounds.lo < 1)
            throw std::runtime_error("invalid region shape bounds");
          i.shapeRange[axis] = bounds;
        }
        for (auto m : {i.shapeRange["m"].lo, i.shapeRange["m"].hi})
          for (auto h : {i.shapeRange["h"].lo, i.shapeRange["h"].hi}) {
            double t = i.fixedMs + i.perMMs * m + i.perHMs * h +
                       i.perMHMs * double(m) * h;
            if (!std::isfinite(t) || t <= 0)
              throw std::runtime_error(
                  "region model nonpositive within validity box");
          }
        if (i.pattern == "matmul_pair" &&
            (i.shapeRange["k"].lo != i.shapeRange["k"].hi ||
             i.shapeRange["n"].lo != i.shapeRange["n"].hi))
          throw std::runtime_error("pair model requires fixed K and N");
        if (i.pattern == "matmul_single" &&
            (!(i.shapeRange[i.featureAxis] == i.shapeRange["h"]) ||
             (i.featureAxis == "n"
                  ? i.shapeRange["k"].lo != i.shapeRange["k"].hi
                  : i.shapeRange["n"].lo != i.shapeRange["n"].hi)))
          throw std::runtime_error(
              "single-op model feature/shape scope mismatch");
        c.regionImplementations.push_back(i);
      }
    }
    auto &a = obj(field(o, "assumptions"));
    c.assumptions.initialLocations = stringMap(a, "initial_locations");
    c.assumptions.outputLocations = stringMap(a, "output_locations");
    c.assumptions.outputDevices = stringMap(a, "output_devices", false);
    c.assumptions.allowedNumericalContracts =
        strings(a, "allowed_numerical_contracts");
    c.assumptions.overlap = str(a, "overlap");
    c.assumptions.executionState =
        str(a, "execution_state", "cold_inputs_only");
    if (c.assumptions.executionState != "cold_inputs_only" &&
        c.assumptions.executionState != "warm_contexts_weights")
      throw std::runtime_error("unsupported execution state");
    c.assumptions.programSourceId = str(a, "program_source_id");
    c.assumptions.executions = integer(a, "executions", false, 1);
    c.assumptions.maxAssignments = integer(a, "max_assignments", false, 256);
    c.assumptions.beamWidth = integer(a, "beam_width", false, 16);
    c.assumptions.maxRegionCells =
        integer(a, "max_region_cells", false, 100000);
    if (c.assumptions.executions != 1 || c.assumptions.maxAssignments <= 0 ||
        c.assumptions.beamWidth <= 0 || c.assumptions.maxRegionCells <= 0 ||
        c.assumptions.allowedNumericalContracts.empty() ||
        (c.assumptions.overlap != "forbid" &&
         c.assumptions.overlap != "explicit"))
      throw std::runtime_error("unsupported assumptions");
    for (auto *m :
         {&c.assumptions.initialLocations, &c.assumptions.outputLocations})
      for (auto &p : *m)
        if (!memories.count(p.second))
          throw std::runtime_error("unknown initial/output memory");
    for (auto &p : c.assumptions.outputDevices) {
      auto l = c.assumptions.outputLocations.find(p.first);
      if (l == c.assumptions.outputLocations.end())
        throw std::runtime_error("output device without location");
      auto &dev = memories[l->second].accessibleDevices;
      if (std::find(dev.begin(), dev.end(), p.second) == dev.end())
        throw std::runtime_error("output device cannot access memory");
    }
    return true;
  } catch (const std::exception &x) {
    e = x.what();
    return false;
  }
}
std::vector<StaticAccess> orbit::taskflow::staticReads(const Compute &c) {
  std::vector<StaticAccess> r;
  auto add = [&](std::string b, Range m, Range n) {
    r.push_back({b, "", "", m, n});
  };
  if (c.kind == "matmul") {
    if (c.inputs.size() != 2)
      return {};
    add(c.inputs[0], c.m, c.k);
    add(c.inputs[1], c.k, c.n);
    if (c.accumulate)
      add(c.output, c.m, c.n);
  } else
    for (auto &b : c.inputs)
      add(b, c.m, c.n);
  return r;
}
std::vector<StaticAccess> orbit::taskflow::staticWrites(const Compute &c) {
  return {{c.output, "", "", c.m, c.n}};
}
double
orbit::taskflow::staticImplementationCost(const Compute &c,
                                          const StaticImplementation &i) {
  double m = c.m.hi - c.m.lo, n = c.n.hi - c.n.lo, k = c.k.hi - c.k.lo;
  return i.cost.fixedMs + i.cost.perElementMs * m * n +
         i.cost.perFlopMs * (c.kind == "matmul" ? 2 * m * n * k : 0);
}
bool orbit::taskflow::staticImplementationSupports(
    const StaticAnalysis &a, const StaticStep &s, const StaticImplementation &i,
    const StaticConfig &cfg) {
  auto &c = s.operation;
  auto has = [](auto &v, auto &x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };
  if (!has(i.operations, c.kind) ||
      !has(cfg.assumptions.allowedNumericalContracts, i.numericalContract) ||
      (!i.allowAccumulate && c.accumulate))
    return false;
  std::map<std::string, int64_t> size{
      {"m", c.m.hi - c.m.lo},
      {"n", c.n.hi - c.n.lo},
      {"k", c.kind == "matmul" ? c.k.hi - c.k.lo : 1}};
  for (auto &v : size) {
    auto p = i.cost.shapeRange.find(v.first);
    if (p == i.cost.shapeRange.end() || v.second < p->second.lo ||
        v.second > p->second.hi)
      return false;
  }
  for (auto &v : i.multiples)
    if (!size.count(v.first) || size[v.first] % v.second)
      return false;
  if (i.fullKOnly) {
    if (c.kind != "matmul" || c.inputs.empty())
      return false;
    auto p = std::find_if(a.program.buffers.begin(), a.program.buffers.end(),
                          [&](auto &b) { return b.id == c.inputs[0]; });
    if (p == a.program.buffers.end() || c.k.lo != 0 || c.k.hi != p->cols)
      return false;
  }
  if (i.overwriteZeroAccumulator) {
    auto p = std::find_if(a.steps.begin(), a.steps.end(), [&](auto &t) {
      return t.taskId == s.taskId && t.operationIndex + 1 == s.operationIndex;
    });
    if (c.kind != "matmul" || p == a.steps.end() ||
        p->operation.kind != "fill" || p->operation.output != c.output ||
        !(p->operation.m == c.m) || !(p->operation.n == c.n))
      return false;
  }
  return true;
}
V orbit::taskflow::serializeStaticAnalysis(const StaticAnalysis &a) {
  A steps;
  for (auto &s : a.steps)
    steps.push_back(step(s));
  return O{{"schema", "orbit.static.analysis.v1"},
           {"identity_scheme", "llvm-json-sorted-17g-sha256"},
           {"program", printPlan(a.program)},
           {"steps", std::move(steps)},
           {"body_fingerprint", a.bodyFingerprint},
           {"semantic_source", "actual_taskflow_mlir_body"}};
}
V orbit::taskflow::serializeStaticPlan(const StaticPlan &p,
                                       bool includeIdentity) {
  O o{{"schema", p.schema},
      {"identity_scheme", p.identityScheme},
      {"program_id", p.programId},
      {"candidate_id", p.candidateId},
      {"configuration_id", p.configurationId},
      {"assignment", sm(p.assignment)},
      {"prediction_kind", p.predictionKind},
      {"execution_status", p.executionStatus},
      {"schedule_policy", p.schedulePolicy},
      {"cost_boundary", p.costBoundary},
      {"storage_policy", p.storagePolicy},
      {"initial_state", p.initialState},
      {"region_cell_count", p.regionCellCount}};
  if (includeIdentity)
    o["plan_id"] = p.planId;
  A steps, events, allocations;
  for (auto &s : p.steps)
    steps.push_back(step(s));
  o["steps"] = std::move(steps);
  for (auto &e : p.events) {
    O q{{"id", e.id},
        {"kind", e.kind},
        {"depends_on", ss(e.dependsOn)},
        {"resource_demands", dm(e.resourceDemands)},
        {"duration_ms", e.durationMs},
        {"start_ms", e.startMs},
        {"end_ms", e.endMs},
        {"reads", accesses(e.reads)},
        {"writes", accesses(e.writes)},
        {"bytes", e.bytes},
        {"provenance", prov(e.provenance)},
        {"cost_boundary", e.costBoundary}};
    if (e.kind == "compute") {
      q["step_id"] = e.stepId;
      q["task_id"] = e.taskId;
      q["operation_index"] = e.operationIndex;
      q["implementation"] = e.implementationId;
      q["implementation_id"] = e.implementationId;
      q["device"] = e.device;
      q["cost_provenance"] = prov(e.provenance);
    }
    if (e.kind == "region") {
      q["implementation_id"] = e.implementationId;
      q["device"] = e.device;
      q["covered_steps"] = ss(e.coveredSteps);
      q["region_pattern"] = e.regionPattern;
      q["region_mode"] = e.regionMode;
      q["model_id"] = e.modelId;
      q["prepared_state"] = e.preparedState;
      q["region_shape"] = dm(e.regionShape);
    }
    if (e.kind == "copy")
      q["link"] = e.link;
    if (e.kind == "convert")
      q["conversion"] = e.conversion;
    if (e.kind == "sync") {
      q["synchronization"] = e.synchronization;
      q["source_device"] = e.sourceDevice;
      q["target_device"] = e.targetDevice;
      q["memory"] = e.memory;
    }
    events.push_back(std::move(q));
  }
  o["events"] = std::move(events);
  for (auto &a : p.allocations) {
    O q{{"id", a.id},
        {"buffer", a.buffer.empty() ? V(nullptr) : V(a.buffer)},
        {"memory", a.memory},
        {"layout", a.layout},
        {"dtype", a.dtype},
        {"bytes", a.bytes},
        {"created_by", a.createdBy.empty() ? V(nullptr) : V(a.createdBy)},
        {"last_users", ss(a.lastUsers)},
        {"lifetime", a.lifetime},
        {"start_ms", a.startMs},
        {"end_ms", a.endMs}};
    q["region"] = a.buffer.empty() ? V(nullptr) : V(A{rr(a.m), rr(a.n)});
    if (a.initial)
      q["initial"] = true;
    allocations.push_back(std::move(q));
  }
  o["allocations"] = std::move(allocations);
  o["outputs"] = accesses(p.outputs);
  o["output_dependencies"] = ss(p.outputDependencies);
  o["output_consumers"] = sm(p.outputConsumers);
  auto &t = p.totals;
  o["makespan_ms"] = t.makespanMs;
  o["totals"] = O{{"makespan_ms", t.makespanMs},
                  {"compute_ms", t.computeMs},
                  {"copy_ms", t.copyMs},
                  {"convert_ms", t.convertMs},
                  {"sync_ms", t.syncMs},
                  {"peak_memory_bytes", dm(t.peakMemoryBytes)},
                  {"transfer_bytes", t.transferBytes}};
  if (p.initialState == "warm_contexts_weights")
    o["totals"].getAsObject()->operator[]("region_wall_ms") = t.regionWallMs;
  return o;
}
V orbit::taskflow::serializeStaticResult(const StaticMappingResult &r) {
  A alternatives;
  for (auto &a : r.alternatives) {
    O v{{"status", a.status}, {"assignment", sm(a.assignment)}};
    if (a.plan) {
      v["score"] = a.plan->totals.makespanMs;
      v["score_unit"] = "ms";
      v["plan"] = serializeStaticPlan(*a.plan);
    } else {
      v["score"] = nullptr;
      v["score_unit"] = nullptr;
      v["reason"] = a.reason;
    }
    alternatives.push_back(std::move(v));
  }
  return O{
      {"schema", "orbit.static.mapping-result.v1"},
      {"identity_scheme", "llvm-json-sorted-17g-sha256"},
      {"status", r.status},
      {"best_plan", r.bestPlan ? serializeStaticPlan(*r.bestPlan) : V(nullptr)},
      {"alternatives", std::move(alternatives)},
      {"reasons", ss(r.reasons)},
      {"search",
       O{{"algorithm", r.regionExactCover
                           ? "bounded_region_exact_cover_search"
                           : "bounded_complete_assignment_search"},
         {"exhaustive_assignments", r.exhaustiveAssignments},
         {"possible_assignments", r.regionExactCover && !r.exhaustiveAssignments
                                      ? V(nullptr)
                                      : V(r.possibleAssignments)},
         {"evaluated_cover_count_is_lower_bound",
          r.regionExactCover && !r.exhaustiveAssignments},
         {"evaluated_assignments", r.evaluatedAssignments},
         {"budget", r.budget},
         {"selection_reason", "lowest independently checked complete makespan; "
                              "no global schedule optimality claim"}}}};
}
bool orbit::taskflow::parseStaticPlan(const V &v, StaticPlan &p,
                                      std::string &e) {
  try {
    p = StaticPlan{};
    auto &o = obj(v);
    p.schema = str(o, "schema");
    p.identityScheme = str(o, "identity_scheme", "llvm-json-sorted-17g-sha256");
    if (p.schema != "orbit.static.plan.v1")
      throw std::runtime_error("unsupported static plan schema");
    p.programId = str(o, "program_id");
    p.candidateId = str(o, "candidate_id");
    p.configurationId = str(o, "configuration_id");
    p.planId = str(o, "plan_id");
    p.assignment = stringMap(o, "assignment");
    p.predictionKind = str(o, "prediction_kind");
    p.executionStatus = str(o, "execution_status");
    p.schedulePolicy = str(o, "schedule_policy");
    p.costBoundary = str(o, "cost_boundary");
    p.storagePolicy = str(o, "storage_policy");
    p.initialState = str(o, "initial_state");
    p.regionCellCount = integer(o, "region_cell_count", false);
    for (auto &v : arr(o, "steps")) {
      auto &o = obj(v);
      StaticStep s;
      s.id = str(o, "id");
      s.taskId = str(o, "task_id");
      if (integer(o, "operation_index") > UINT32_MAX)
        throw std::runtime_error("operation index overflow");
      s.operationIndex = integer(o, "operation_index");
      s.dependsOn = strings(o, "depends_on");
      s.operation = parseCompute(field(o, "operation"));
      s.reads = staticReads(s.operation);
      s.writes = staticWrites(s.operation);
      p.steps.push_back(s);
    }
    for (auto &v : arr(o, "events")) {
      auto &o = obj(v);
      StaticEvent q;
      q.id = str(o, "id");
      q.kind = str(o, "kind");
      q.dependsOn = strings(o, "depends_on");
      q.resourceDemands = demands(o, "resource_demands");
      q.durationMs = num(o, "duration_ms");
      q.startMs = num(o, "start_ms");
      q.endMs = num(o, "end_ms");
      q.reads = parseAccesses(o, "reads");
      q.writes = parseAccesses(o, "writes");
      q.bytes = integer(o, "bytes", false);
      q.provenance = provenance(o);
      q.costBoundary = str(o, "cost_boundary");
      q.stepId = str(o, "step_id");
      q.taskId = str(o, "task_id");
      if (integer(o, "operation_index", false) > UINT32_MAX)
        throw std::runtime_error("operation index overflow");
      q.operationIndex = integer(o, "operation_index", false);
      q.implementationId =
          str(o, "implementation_id", str(o, "implementation"));
      if (o.get("implementation") &&
          str(o, "implementation") != q.implementationId)
        throw std::runtime_error("implementation aliases disagree");
      if (auto cp = o.get("cost_provenance"))
        if (staticFingerprint(*cp) != staticFingerprint(field(o, "provenance")))
          throw std::runtime_error("cost provenance aliases disagree");
      q.device = str(o, "device");
      if (q.kind == "region") {
        q.coveredSteps = strings(o, "covered_steps");
        q.regionPattern = str(o, "region_pattern");
        q.regionMode = str(o, "region_mode");
        q.modelId = str(o, "model_id");
        q.preparedState = str(o, "prepared_state");
        q.regionShape = demands(o, "region_shape");
      }
      q.link = str(o, "link");
      q.conversion = str(o, "conversion");
      q.synchronization = str(o, "synchronization");
      q.sourceDevice = str(o, "source_device");
      q.targetDevice = str(o, "target_device");
      q.memory = str(o, "memory");
      p.events.push_back(q);
    }
    for (auto &v : arr(o, "allocations")) {
      auto &o = obj(v);
      StaticAllocation a;
      a.id = str(o, "id");
      a.buffer = o.getString("buffer").value_or("").str();
      a.memory = str(o, "memory");
      a.layout = str(o, "layout");
      a.dtype = str(o, "dtype");
      a.bytes = integer(o, "bytes");
      a.createdBy = o.getString("created_by").value_or("").str();
      a.lifetime = str(o, "lifetime");
      a.initial = flag(o, "initial");
      a.lastUsers = strings(o, "last_users");
      a.startMs = num(o, "start_ms");
      a.endMs = num(o, "end_ms");
      if (!a.buffer.empty()) {
        auto &r = arr(o, "region");
        if (r.size() != 2)
          throw std::runtime_error("allocation region rank mismatch");
        a.m = range(r[0]);
        a.n = range(r[1]);
      }
      p.allocations.push_back(a);
    }
    p.outputs = parseAccesses(o, "outputs");
    p.outputDependencies = strings(o, "output_dependencies", false);
    p.outputConsumers = stringMap(o, "output_consumers", false);
    auto &t = obj(field(o, "totals"));
    p.totals.makespanMs = num(t, "makespan_ms");
    if (num(o, "makespan_ms") != p.totals.makespanMs)
      throw std::runtime_error("top-level makespan mismatch");
    p.totals.computeMs = num(t, "compute_ms");
    p.totals.copyMs = num(t, "copy_ms");
    p.totals.convertMs = num(t, "convert_ms");
    p.totals.syncMs = num(t, "sync_ms");
    p.totals.regionWallMs = num(t, "region_wall_ms", false);
    p.totals.transferBytes = integer(t, "transfer_bytes");
    for (auto &x : obj(field(t, "peak_memory_bytes"))) {
      auto n = x.second.getAsInteger();
      if (!n || *n < 0)
        throw std::runtime_error("invalid peak memory");
      p.totals.peakMemoryBytes[x.first.str()] = *n;
    }
    return true;
  } catch (const std::exception &x) {
    e = x.what();
    return false;
  }
}

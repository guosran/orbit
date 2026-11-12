// External Qwen runtime adapter: finite public ORBIT search, no native
// lowering.
#include "Taskflow/ProgramBeamSearch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
using namespace orbit::taskflow;
namespace j = llvm::json;
static j::Value readJSON(const char *path) {
  auto b = llvm::MemoryBuffer::getFile(path);
  if (!b)
    throw std::runtime_error("cannot read input");
  auto v = j::parse((*b)->getBuffer());
  if (!v) {
    llvm::consumeError(v.takeError());
    throw std::runtime_error("invalid JSON");
  }
  return std::move(*v);
}
static std::string string(const j::Object &o, llvm::StringRef k) {
  auto v = o.getString(k);
  if (!v || v->empty())
    throw std::runtime_error("missing string " + k.str());
  return v->str();
}
static double number(const j::Object &o, llvm::StringRef k) {
  auto v = o.getNumber(k);
  if (!v || !std::isfinite(*v) || *v < 0)
    throw std::runtime_error("invalid nonnegative finite cost " + k.str());
  return *v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "usage: qwen-local-search graph.json local-costs.json beam|cp");
    auto graphValue = readJSON(argv[1]), costValue = readJSON(argv[2]);
    auto *graph = graphValue.getAsObject(), *cost = costValue.getAsObject();
    if (!graph || !cost)
      throw std::runtime_error("inputs must be objects");
    if (string(*graph, "schema") != "orbit.external-qwen-runtime.v1")
      throw std::runtime_error("wrong graph schema");
    if (string(*cost, "schema") != "orbit.external-qwen-local-cost.v1")
      throw std::runtime_error("wrong cost schema");
    auto sourceID = string(*graph, "source_id");
    if (sourceID != string(*cost, "source_id"))
      throw std::runtime_error("cost/source mismatch");
    if (string(*cost, "cost_boundary") !=
        "synchronous-op-wall-includes-launch-and-transfer")
      throw std::runtime_error("unsupported timing boundary");
    auto *nodes = graph->getArray("nodes"), *prices = cost->getArray("nodes");
    if (!nodes || nodes->empty() || !prices || prices->size() != nodes->size())
      throw std::runtime_error("incomplete node coverage");
    std::map<std::string, const j::Object *> priceMap;
    for (auto &p : *prices) {
      auto *o = p.getAsObject();
      if (!o || !priceMap.emplace(string(*o, "id"), o).second)
        throw std::runtime_error("duplicate/bad price node");
    }
    struct Node {
      std::string id, family;
      int dimension;
      double npu, cpu;
    };
    std::vector<Node> checked;
    std::set<std::string> ids;
    std::map<std::string, int64_t> families;
    int variableCounts[2] = {0, 0};
    for (auto &v : *nodes) {
      auto *n = v.getAsObject();
      if (!n)
        throw std::runtime_error("bad graph node");
      auto id = string(*n, "id"), family = string(*n, "family");
      if (!ids.insert(id).second)
        throw std::runtime_error("duplicate source node");
      auto *binding = n->getObject("source_binding");
      if (!binding || binding->empty())
        throw std::runtime_error("unbound source node");
      auto i = priceMap.find(id);
      if (i == priceMap.end())
        throw std::runtime_error("unpriced source node");
      const auto &p = *i->second;
      if (string(p, "family") != family)
        throw std::runtime_error("family identity mismatch");
      int d = family == "rms_norm" ? 0 : (family == "swiglu" ? 1 : -1);
      double np = 0, cp = 0;
      if (d >= 0) {
        auto *choices = p.getObject("cost_by_choice_ms");
        if (!choices || choices->size() != 2)
          throw std::runtime_error("both CPU/NPU choices required");
        np = number(*choices, "npu");
        cp = number(*choices, "cpu");
        ++variableCounts[d];
      } else
        np = cp = number(p, "cost_ms");
      checked.push_back({id, family, d, np, cp});
      ++families[family];
    }
    if (variableCounts[0] != 73 || variableCounts[1] != 36 ||
        families["matmul"] != 144 || families["opaque_decode"] != 127 ||
        families["rope"] != 72 || families["attention"] != 36 ||
        families["lm_head"] != 1 || families["kv_store"] != 36 ||
        families["kv_handoff"] != 1 || families.size() != 9)
      throw std::runtime_error("incomplete 36-layer / 128-output graph");
    // Residual is separately registered local host work, never an entire-plan
    // price.
    if (string(*cost, "host_residual_boundary") !=
        "total_request_ms - sum(prefill_op_records_ms) - "
        "sum(inclusive_decode_steps_ms); includes kv handoff, embedding, "
        "inline/control, capture copies outside dispatch")
      throw std::runtime_error("unrecognized residual accounting boundary");
    const double residual = number(*cost, "host_residual_ms");
    ProgramSearchSpace space;
    for (const char *family : {"rms_norm", "swiglu"}) {
      ProgramSearchDimension d;
      d.id = std::string("placement.") + family;
      d.scope = "global-runtime-family";
      d.status = "searched";
      for (const char *device : {"npu", "cpu"}) {
        ProgramSearchChoice c;
        c.id = device;
        c.binding = j::Object{{"device", device}, {"family", family}};
        d.choices.push_back(std::move(c));
      }
      space.dimensions.push_back(std::move(d));
    }
    int64_t calls = 0;
    auto evaluate = [&](const std::vector<int64_t> &a) {
      ++calls;
      ProgramSearchEvaluation e;
      e.feasible = true;
      double total = residual;
      j::Object terms;
      for (const auto &n : checked) {
        double t = n.dimension < 0 || a[n.dimension] == 0 ? n.npu : n.cpu;
        total += t;
        terms[n.family] = terms.getNumber(n.family).value_or(0) + t;
      }
      if (!std::isfinite(total))
        throw std::runtime_error("cost sum overflow");
      terms["host_residual"] = residual;
      e.costMs = total;
      e.result = j::Object{{"kind", "external-source-bound-runtime-cost"},
                           {"source_id", sourceID},
                           {"family_cost_ms", std::move(terms)},
                           {"complete_checked_cost_ms", total}};
      return e;
    };
    ProgramSearchOptions options;
    options.wallSeconds = 600;
    options.maxEvaluations = 16;
    std::string algo = argv[3];
    ProgramSearchResult r;
    if (algo == "beam")
      r = searchProgramSpaceBeam(space, evaluate, options, 4, 4);
    else if (algo == "cp")
      r = searchProgramSpace(space, evaluate, options);
    else
      throw std::runtime_error("unknown algorithm");
    j::Object output{
        {"schema", "orbit.external-qwen-search-result.v1"},
        {"adapter_scope", "external source-bound runtime graph; not native "
                          "ProgramGraph lowering"},
        {"source_id", sourceID},
        {"cost_boundary", "synchronous-op-wall-includes-launch-and-transfer"},
        {"extra_launch_or_transfer_edge_cost_ms", 0},
        {"evaluator_calls", calls},
        {"report", std::move(r.report)}};
    if (r.hasBest) {
      output["best_cost_ms"] = *r.bestEvaluation.costMs;
      output["rms_norm_device"] = r.bestAssignment[0] ? "cpu" : "npu";
      output["swiglu_device"] = r.bestAssignment[1] ? "cpu" : "npu";
      output["best_terms"] = std::move(r.bestEvaluation.result);
    }
    llvm::outs() << j::Value(std::move(output)) << "\n";
    return r.hasBest ? 0 : 2;
  } catch (const std::exception &e) {
    llvm::errs() << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

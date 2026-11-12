#include "Taskflow/ProgramJointMILP.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>

using namespace orbit::taskflow;
static const char *source = R"mlir(
module {
 func.func @shared(%X: memref<1x1xf32> {orbit.role="input",orbit.name="X"},
                   %Y: memref<1x1xf32> {orbit.role="output",orbit.name="Y"},
                   %Z: memref<1x1xf32> {orbit.role="output",orbit.name="Z"}) attributes {orbit.noalias} {
  %a = taskflow.task @first will_reads(%X : memref<1x1xf32>) will_writes(%Y : memref<1x1xf32>) [original_read_memrefs(%X : memref<1x1xf32>),original_write_memrefs(%Y : memref<1x1xf32>)] : (memref<1x1xf32>,memref<1x1xf32>)->memref<1x1xf32> {
  ^bb0(%x: memref<1x1xf32>,%o: memref<1x1xf32>):
   affine.for %i=0 to 1 {affine.for %j=0 to 1 {
    %v=affine.load %x[%i,%j] : memref<1x1xf32>
    affine.store %v,%o[%i,%j] : memref<1x1xf32>
   }}
   taskflow.yield done_writes(%o : memref<1x1xf32>)
  }
  %b = taskflow.task @second will_reads(%X : memref<1x1xf32>) will_writes(%Z : memref<1x1xf32>) [original_read_memrefs(%X : memref<1x1xf32>),original_write_memrefs(%Z : memref<1x1xf32>)] : (memref<1x1xf32>,memref<1x1xf32>)->memref<1x1xf32> {
  ^bb0(%x: memref<1x1xf32>,%o: memref<1x1xf32>):
   affine.for %i=0 to 1 {affine.for %j=0 to 1 {
    %v=affine.load %x[%i,%j] : memref<1x1xf32>
    affine.store %v,%o[%i,%j] : memref<1x1xf32>
   }}
   taskflow.yield done_writes(%o : memref<1x1xf32>)
  }
  return
 }
})mlir";

static llvm::json::Value config(int capacity, int memory, int compute,
                                bool alternate) {
  namespace j = llvm::json;
  j::Array domains{
      j::Object{{"id", "system"}, {"kind", "system"}, {"parent", ""}}};
  j::Array resources{j::Object{{"id", "host.worker"},
                               {"kind", "compute"},
                               {"device", "host"},
                               {"capacity", 1},
                               {"domain", "system"}},
                     j::Object{{"id", "bus"},
                               {"kind", "shared"},
                               {"device", "npu"},
                               {"capacity", capacity},
                               {"domain", "system"}}};
  resources.push_back(j::Object{{"id", "submit"},
                                {"kind", "submission"},
                                {"device", "host"},
                                {"capacity", 1},
                                {"domain", "system"}});
  j::Array memories{
      j::Object{{"id", "host"},
                {"kind", "system-bo"},
                {"domain", "system"},
                {"capacity_bytes", 1000000},
                {"accessible_resources", j::Array{"host.worker"}}}};
  j::Array links;
  auto link = [&](std::string id, std::string src, std::string dst,
                  std::string resource, double bandwidth) {
    links.push_back(
        j::Object{{"id", id},
                  {"source", src},
                  {"target", dst},
                  {"resource", resource},
                  {"domain", "system"},
                  {"bandwidth_bytes_per_ms", bandwidth},
                  {"latency_ms", 0},
                  {"provenance",
                   j::Object{{"kind", "architecture"},
                             {"source", "synthetic finite solver contract"}}}});
  };
  for (int i = 0; i < 2; ++i) {
    std::string n = std::to_string(i), core = "core" + n, mem = "mem" + n,
                dma = "dma" + n;
    domains.push_back(
        j::Object{{"id", "tile" + n}, {"parent", "system"}, {"kind", "tile"}});
    resources.push_back(j::Object{{"id", core},
                                  {"kind", "compute"},
                                  {"device", "npu"},
                                  {"capacity", 1},
                                  {"domain", "tile" + n}});
    resources.push_back(j::Object{{"id", dma},
                                  {"kind", "transfer"},
                                  {"device", "npu"},
                                  {"capacity", 1},
                                  {"domain", "system"},
                                  {"shared_resources", j::Object{{"bus", 1}}}});
    memories.push_back(j::Object{{"id", mem},
                                 {"kind", "local-sram"},
                                 {"domain", "tile" + n},
                                 {"capacity_bytes", memory},
                                 {"accessible_resources", j::Array{core}}});
    link("in" + n, "host", mem, dma, 500000);
    link("out" + n, mem, "host", dma, 4000000);
    if (alternate) {
      std::string fast = "fast" + n;
      resources.push_back(
          j::Object{{"id", fast},
                    {"kind", "transfer"},
                    {"device", "npu"},
                    {"capacity", 1},
                    {"domain", "system"},
                    {"shared_resources", j::Object{{"fast.bus", 1}}}});
      link("fast.in" + n, "host", mem, fast, 2000000);
    }
  }
  if (alternate)
    resources.push_back(j::Object{{"id", "fast.bus"},
                                  {"kind", "shared"},
                                  {"device", "npu"},
                                  {"capacity", 1},
                                  {"domain", "system"}});
  j::Object parameters;
  for (auto item :
       std::vector<std::pair<std::string, int>>{{"compute.startup", compute},
                                                {"compute.operation", 0},
                                                {"compute.memory_access", 0},
                                                {"configuration.startup", 0},
                                                {"host.submit", 0},
                                                {"host.complete", 0}})
    parameters[item.first] =
        j::Object{{"value", item.second},
                  {"unit", "ns"},
                  {"provenance",
                   j::Object{{"kind", "assumed"},
                             {"source", "finite test; no hardware claim"}}}};
  return j::Object{
      {"generic_air",
       j::Object{
           {"cost_model", j::Object{{"kind", "event-mechanism"},
                                    {"schema", "orbit.direct-air.cost.v1"},
                                    {"mode", "estimate"},
                                    {"model_id", "milp-contract"},
                                    {"implementations", j::Array{"scalar"}},
                                    {"boundary", "synthetic-device"},
                                    {"parameters", std::move(parameters)}}}}},
      {"program_joint",
       j::Object{{"architecture", j::Object{{"domains", std::move(domains)},
                                            {"resources", std::move(resources)},
                                            {"memories", std::move(memories)},
                                            {"links", std::move(links)}}},
                 {"host_memory", "host"},
                 {"chunks", j::Array{1}},
                 {"max_configurations", 2}}}};
}

// Independent tiny integer calendar oracle. Enumerate the legal groupings,
// input routes and every topological serial-generation order. Earliest integer
// placement for a fixed order gives an active schedule; some optimum is active.
// This uses neither solver's model, selected solution nor event checker.
static int64_t oracle(int capacity, int memory, int compute, bool alternate,
                      int secondCompute = -1, int inputNs = 8) {
  struct Op {
    int duration, resource;
    std::vector<int> predecessors;
  };
  int64_t best = INT64_MAX;
  if (secondCompute < 0)
    secondCompute = compute;
  for (bool merged : {false, true}) {
    if (merged && memory < 12)
      continue;
    int groups = merged ? 1 : 2;
    for (int routeMask = 0; routeMask < (alternate ? (1 << groups) : 1);
         ++routeMask) {
      std::vector<Op> ops;
      for (int g = 0; g < groups; ++g) {
        int base = ops.size();
        bool fast = routeMask & (1 << g);
        ops.push_back({fast ? 2 : inputNs, fast ? 1 : 0, {}});
        ops.push_back(
            {merged ? compute + secondCompute : (g ? secondCompute : compute),
             2 + g,
             {base}});
        ops.push_back({1, 0, {base + 1}});
        if (merged)
          ops.push_back({1, 0, {base + 1}});
      }
      std::vector<int> order(ops.size());
      std::iota(order.begin(), order.end(), 0);
      do {
        int end[6] = {};
        std::vector<std::vector<int>> calendar(
            4, std::vector<int>(
                   std::max(100, 4 * (inputNs + compute + secondCompute + 10)),
                   0));
        bool legal = true;
        int maximum = 0;
        for (int i : order) {
          auto &op = ops[i];
          int earliest = 0;
          for (int p : op.predecessors) {
            if (!end[p] && ops[p].duration) {
              legal = false;
              break;
            }
            if (std::find(order.begin(),
                          std::find(order.begin(), order.end(), i),
                          p) == std::find(order.begin(), order.end(), i)) {
              legal = false;
              break;
            }
            earliest = std::max(earliest, end[p]);
          }
          if (!legal)
            break;
          int resourceCapacity = op.resource == 0 ? capacity : 1;
          for (;; ++earliest) {
            bool fits = true;
            for (int t = earliest; t < earliest + op.duration; ++t)
              if (calendar[op.resource][t] >= resourceCapacity)
                fits = false;
            if (fits)
              break;
          }
          end[i] = earliest + op.duration;
          maximum = std::max(maximum, end[i]);
          for (int t = earliest; t < end[i]; ++t)
            ++calendar[op.resource][t];
        }
        if (legal)
          best = std::min<int64_t>(best, maximum);
      } while (std::next_permutation(order.begin(), order.end()));
    }
  }
  return best;
}

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.disableMultithreading();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  GenericAnalysis analysis;
  std::string error;
  int checks = 0, failures = 0;
  auto require = [&](bool okay, const std::string &label) {
    ++checks;
    if (!okay) {
      ++failures;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
    return okay;
  };
  if (!require(bool(module) && analyzeGenericProgram(*module, analysis, error),
               "parse source"))
    return 1;
  ConstraintSearchOptions options;
  options.wallSeconds = 10;
  options.workers = 1;
  ProgramJointDomain saved;
  for (auto tuple : std::vector<std::array<int, 4>>{{1, 8, 4, 0},
                                                    {1, 12, 4, 0},
                                                    {2, 12, 4, 0},
                                                    {2, 8, 0, 0},
                                                    {1, 8, 0, 0},
                                                    {1, 12, 4, 1}}) {
    int cap = tuple[0], mem = tuple[1], compute = tuple[2];
    bool alternate = tuple[3];
    ProgramJointDomain domain;
    if (!require(buildProgramJointDomain(analysis,
                                         config(cap, mem, compute, alternate),
                                         domain, error),
                 "build typed domain"))
      continue;
    auto baseline = baselineProgramJointDomain(domain);
    error = baseline.message;
    if (!require(baseline.hasPlan, "independent baseline"))
      continue;
    auto answer = solveProgramJointDomainMILP(domain, options, &baseline);
    int64_t exact = oracle(cap, mem, compute, alternate);
    if (programJointMILPAvailable()) {
      error = answer.message + " actual=" + std::to_string(answer.makespanNs) +
              " expected=" + std::to_string(exact);
      require(answer.hasPlan &&
                  answer.status == ConstraintSearchStatus::Optimal &&
                  answer.makespanNs == exact,
              "MILP equals enumeration cap=" + std::to_string(cap));
      require(answer.message.find("checked baseline primal start accepted") !=
                      std::string::npos &&
                  answer.selection == "joint-milp" &&
                  answer.telemetry.hasObjective,
              "checked baseline was submitted to SCIP, not only returned "
              "externally");
      auto cp = solveProgramJointDomain(domain, options, &baseline);
      if (cp.status != ConstraintSearchStatus::Unavailable)
        require(cp.status == ConstraintSearchStatus::Optimal &&
                    cp.makespanNs == exact,
                "CP equals independent enumeration");
    } else
      require(answer.status == ConstraintSearchStatus::Unavailable &&
                  answer.hasPlan && answer.selection == "baseline",
              "SCIP OFF baseline retained");
    ProgramGraph graph;
    require(materializeProgramJointSolution(domain, answer, graph, error) &&
                checkProgramJointSolution(domain, answer, graph, error),
            "independent winner check");
    if (cap == 2 && compute == 4 && !alternate)
      saved = domain;
  }
  if (!saved.variants.empty() && programJointMILPAvailable()) {
    auto baseline = baselineProgramJointDomain(saved);
    auto limited = options;
    limited.maxVariables = 3;
    auto r = solveProgramJointDomainMILP(saved, limited, &baseline);
    require(r.status == ConstraintSearchStatus::ModelLimit && r.hasPlan &&
                r.selection == "baseline",
            "model cap preserves checked baseline");
    limited = options;
    limited.wallSeconds = 1e-9;
    r = solveProgramJointDomainMILP(saved, limited, &baseline);
    require(r.status == ConstraintSearchStatus::Unknown && r.hasPlan &&
                r.telemetry.budgetExhausted,
            "shared total deadline");
    limited = options;
    limited.workers = 2;
    r = solveProgramJointDomainMILP(saved, limited, &baseline);
    require(r.status == ConstraintSearchStatus::InvalidInput && r.hasPlan,
            "unsupported parallel workers rejected explicitly");
    saved.complete = false;
    r = solveProgramJointDomainMILP(saved, options, &baseline);
    require(r.status != ConstraintSearchStatus::Optimal && r.hasPlan,
            "truncated structural domain never full optimal");
    saved.complete = true;
    saved.costsComplete = false;
    r = solveProgramJointDomainMILP(saved, options, &baseline);
    require(r.status != ConstraintSearchStatus::Optimal && r.hasPlan,
            "unknown cost domain never full optimal");
  }
  if (!saved.variants.empty()) {
    auto baseline = baselineProgramJointDomain(saved);
    for (int i = 0; i < 3; ++i) {
      auto invalid = options;
      invalid.workers = i == 0 ? 0 : (i == 1 ? 17 : 1);
      invalid.randomSeed = i == 2 ? -1 : 0;
      auto cp = solveProgramJointDomain(saved, invalid, &baseline);
      require(cp.status == (constraintSearchAvailable()
                                ? ConstraintSearchStatus::InvalidInput
                                : ConstraintSearchStatus::Unavailable),
              "joint CP typed worker/seed range");
      auto milp = solveProgramJointDomainMILP(saved, invalid, &baseline);
      require(milp.status == (programJointMILPAvailable()
                                  ? ConstraintSearchStatus::InvalidInput
                                  : ConstraintSearchStatus::Unavailable),
              "joint MILP typed worker/seed range");
    }
    auto legal = options;
    legal.randomSeed = std::numeric_limits<int>::max();
    auto cp = solveProgramJointDomain(saved, legal, &baseline);
    require(cp.hasPlan && cp.status != ConstraintSearchStatus::InvalidInput,
            "joint CP INT_MAX seed with one worker");
    auto milp = solveProgramJointDomainMILP(saved, legal, &baseline);
    require(milp.hasPlan && milp.status != ConstraintSearchStatus::InvalidInput,
            "joint MILP INT_MAX seed with one worker");
  }
  if (!saved.variants.empty()) {
    // Typed callers may append a structurally legal route without refreshing
    // costsComplete. Such an unpriced alternative must not silently disappear
    // from a claimed full-domain optimality proof.
    auto stale = saved;
    stale.complete = true;
    stale.costsComplete = true;
    stale.unknownCosts.clear();
    auto baseline = baselineProgramJointDomain(stale);
    auto alternative = stale.target.links.front();
    alternative.id = "typed-unpriced-alternative";
    alternative.bandwidthBytesPerMs = 0;
    size_t linkIndex = stale.target.links.size();
    stale.target.links.push_back(alternative);
    for (auto &place : stale.places)
      if (place.binding.memory == alternative.target)
        place.inputLinks.push_back(linkIndex);
    auto cp = solveProgramJointDomain(stale, options, &baseline);
    llvm::outs() << "typed-unpriced CP status="
                 << constraintSearchStatusName(cp.status) << " " << cp.message
                 << "\n";
    require(cp.hasPlan &&
                cp.status == (constraintSearchAvailable()
                                  ? ConstraintSearchStatus::Feasible
                                  : ConstraintSearchStatus::Unavailable),
            "unpriced typed alternative blocks CP full-domain optimum");
    auto milp = solveProgramJointDomainMILP(stale, options, &baseline);
    require(milp.hasPlan &&
                milp.status == (programJointMILPAvailable()
                                    ? ConstraintSearchStatus::Feasible
                                    : ConstraintSearchStatus::Unavailable),
            "unpriced typed alternative blocks MILP full-domain optimum");
    for (const auto *answer : {&cp, &milp}) {
      auto receipt = serializeProgramJointSolution(stale, *answer);
      auto *row = receipt.getAsObject();
      const bool available =
          answer->status != ConstraintSearchStatus::Unavailable;
      require(
          row && row->getBoolean("full_domain_optimal") == false &&
              (!available || (row->getBoolean("pricing_complete") == false &&
                              row->getBoolean("costs_complete") == false &&
                              row->getString("optimization_scope") ==
                                  "known-cost-generated-subset" &&
                              row->getString("bound_scope") ==
                                  "known-cost-generated-subset")),
          "both typed solver receipts scope missing prices and lower bounds");
      ProgramJointSolution restored;
      require(parseProgramJointSolution(receipt, restored, error) &&
                  restored.pricingComplete == answer->pricingComplete,
              "dynamic pricing coverage survives solution replay");
    }
  }
  // Twelve real source instances used to exceed the common variable budget
  // with start-anchor encoding even on unary resources. Keep the complete
  // local domain and verify that a short solve has a real SCIP incumbent.
  if (programJointMILPAvailable()) {
    std::string larger(source);
    auto replaceAll = [&](const std::string &from, const std::string &to) {
      size_t position = 0;
      while ((position = larger.find(from, position)) != std::string::npos) {
        larger.replace(position, from.size(), to);
        position += to.size();
      }
    };
    replaceAll("memref<1x1xf32>", "memref<6x1xf32>");
    replaceAll("%i=0 to 1", "%i=0 to 6");
    auto largerModule =
        mlir::parseSourceString<mlir::ModuleOp>(larger, &context);
    GenericAnalysis largerAnalysis;
    ProgramJointDomain largerDomain;
    auto largerConfig = config(1, 12, 4, false);
    (*largerConfig.getAsObject()->getObject(
        "program_joint"))["max_configurations"] = 12;
    bool built = largerModule &&
                 analyzeGenericProgram(*largerModule, largerAnalysis, error) &&
                 buildProgramJointDomain(largerAnalysis, largerConfig,
                                         largerDomain, error);
    if (require(built, "build twelve-instance unary-resource domain")) {
      auto baseline = baselineProgramJointDomain(largerDomain);
      auto shortBudget = options;
      shortBudget.wallSeconds = 5;
      auto answer =
          solveProgramJointDomainMILP(largerDomain, shortBudget, &baseline);
      error = answer.message;
      require(answer.hasPlan && answer.selection == "joint-milp" &&
                  answer.telemetry.hasObjective &&
                  (answer.status == ConstraintSearchStatus::Feasible ||
                   answer.status == ConstraintSearchStatus::Optimal),
              "short solve retains actual warm SCIP incumbent");
      require(answer.telemetry.variables < 100000 &&
                  answer.telemetry.constraints < 500000,
              "exact unary formulation fits the original common model cap");
      ProgramGraph graph;
      require(
          materializeProgramJointSolution(largerDomain, answer, graph, error) &&
              checkProgramJointSolution(largerDomain, answer, graph, error),
          "twelve-instance warm incumbent independently checked");
    }
  }
  // An asymmetric compute case with long inputs catches an invalid bound
  // based on the largest compute instead of the conservative global minimum.
  // The long first compute hides beneath the other group's 1000ns input.
  if (programJointMILPAvailable()) {
    std::string skewed(source);
    auto second = skewed.find("taskflow.task @second");
    auto store = skewed.find("affine.store %v", second);
    std::string operations;
    for (int i = 0; i < 96; ++i) {
      std::string previous = i ? "%sum" + std::to_string(i - 1) : "%v";
      operations += "%sum" + std::to_string(i) + "=arith.addf " + previous +
                    "," + previous + " : f32\n";
    }
    skewed.insert(store, operations);
    store = skewed.find("affine.store %v", store + operations.size());
    skewed.replace(store, std::string("affine.store %v").size(),
                   "affine.store %sum95");
    auto skewedModule =
        mlir::parseSourceString<mlir::ModuleOp>(skewed, &context);
    auto skewedConfig = config(1, 8, 4, false);
    auto *parameters = skewedConfig.getAsObject()
                           ->getObject("generic_air")
                           ->getObject("cost_model")
                           ->getObject("parameters");
    (*parameters->getObject("compute.operation"))["value"] = 1;
    for (auto &link : *skewedConfig.getAsObject()
                           ->getObject("program_joint")
                           ->getObject("architecture")
                           ->getArray("links"))
      if (link.getAsObject()->getString("source") == "host")
        (*link.getAsObject())["bandwidth_bytes_per_ms"] = 4000;
    GenericAnalysis analysis;
    ProgramJointDomain domain;
    bool built = skewedModule &&
                 analyzeGenericProgram(*skewedModule, analysis, error) &&
                 buildProgramJointDomain(analysis, skewedConfig, domain, error);
    if (require(built, "build asymmetric source-derived costs")) {
      require(
          domain.variants[0].computeNs == std::vector<int64_t>({4, 100}),
          "actual scalar source operations produce asymmetric duration table");
      auto exact = oracle(1, 8, 4, false, 100, 1000);
      require(exact == 2005, "independent calendar exposes hidden-compute "
                             "lower-bound counterexample");
      auto answer = solveProgramJointDomainMILP(domain, options);
      error = answer.message;
      require(answer.status == ConstraintSearchStatus::Optimal &&
                  answer.makespanNs == exact,
              "guarded bound preserves asymmetric exact optimum");
      auto cp = solveProgramJointDomain(domain, options);
      if (cp.status != ConstraintSearchStatus::Unavailable)
        require(cp.status == ConstraintSearchStatus::Optimal &&
                    cp.makespanNs == exact,
                "asymmetric oracle agrees with independent CP model");
    }
  }
  llvm::outs() << "program-joint-milp: " << checks << " checks, " << failures
               << " failures\n";
  return failures ? 1 : 0;
}

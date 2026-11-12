#include "Taskflow/GenericMechanismTiming.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramMechanismCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <set>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis analysis;
  GenericMappingPlan plan;
  std::string error;
  if (!module || !analyzeGenericProgram(*module, analysis, error) ||
      !planGenericMapping(analysis, llvm::json::Object{}, plan, error)) {
    llvm::errs() << error;
    return 1;
  }
  unsigned checks = 0;
  bool ok = true;
  auto expect = [&](bool condition, const char *message) {
    ++checks;
    if (!condition) {
      ok = false;
      llvm::errs() << message << ": " << error << "\n";
    }
    error.clear();
  };
  ProgramMechanismModel model;
  model.id = "synthetic-no-device";
  model.mode = "estimate";
  model.boundary = "synthetic-complete-graph";
  model.implementations = {"scalar"};
  for (const auto &entry :
       std::map<std::string, double>{{"compute.startup", 10},
                                     {"compute.operation", 3},
                                     {"compute.memory_access", 2},
                                     {"communication.startup", 50},
                                     {"communication.bandwidth", 16},
                                     {"configuration.startup", 100},
                                     {"host.submit", 1000},
                                     {"host.complete", 2000},
                                     {"host.visibility", 0}}) {
    ProgramMechanismParameter p;
    p.value = entry.second;
    p.unit = entry.first == "communication.bandwidth" ? "bytes/ns" : "ns";
    p.kind = "assumed";
    p.source = "synthetic unit-test parameter; no hardware claim";
    model.parameters[entry.first] = p;
  }
  ProgramMechanismEvaluation result;
  expect(
      evaluateProgramMechanismCost(plan.communication, model, result, error) &&
          result.timing.makespanMs && !result.assumptions.empty() &&
          result.status == "assumption-driven-estimate",
      "explicit assumptions produce a finite schedule");
  auto complete = result.timing.makespanMs;
  expect(complete && std::abs(*complete - .003486) < 1e-12,
         "actual source-copy work plus path, setup and host boundary prices");
  expect(result.quantities.size() == plan.communication.events.size(),
         "quantities bind every existing graph event");
  auto strict = model;
  strict.mode = "strict";
  expect(
      evaluateProgramMechanismCost(plan.communication, strict, result, error) &&
          !result.timing.makespanMs && !result.timing.unknowns.empty(),
      "strict mode cannot silently use assumptions");
  for (auto &entry : strict.parameters)
    entry.second.kind = "architecture";
  expect(
      evaluateProgramMechanismCost(plan.communication, strict, result, error) &&
          result.timing.makespanMs == complete && result.assumptions.empty(),
      "strict accepts explicitly supplied architecture parameters");
  auto missing = model;
  missing.parameters.erase("communication.bandwidth");
  expect(evaluateProgramMechanismCost(plan.communication, missing, result,
                                      error) &&
             !result.timing.makespanMs && result.timing.knownWorkMs > 0 &&
             result.timing.lowerBoundMs > 0,
         "missing bandwidth preserves known host/compute and lower bounds");
  auto bad = model;
  bad.parameters["compute.operation"].value = -1;
  expect(!evaluateProgramMechanismCost(plan.communication, bad, result, error),
         "negative coefficients rejected");
  bad = model;
  bad.parameters["communication.bandwidth"].value = 0;
  expect(!evaluateProgramMechanismCost(plan.communication, bad, result, error),
         "used zero bandwidth is not infinite throughput");
  bad = model;
  bad.parameters["compute.memory_access"].unit = "cycles";
  expect(!evaluateProgramMechanismCost(plan.communication, bad, result, error),
         "units cannot be guessed");
  auto unsupported = model;
  unsupported.implementations = {"other"};
  expect(evaluateProgramMechanismCost(plan.communication, unsupported, result,
                                      error) &&
             !result.timing.makespanMs,
         "unsupported implementation stays unknown");
  auto changed = model;
  changed.parameters["configuration.startup"].value = 1000;
  expect(evaluateProgramMechanismCost(plan.communication, changed, result,
                                      error) &&
             result.timing.makespanMs && complete &&
             std::abs(*result.timing.makespanMs - *complete - .0009) < 1e-12,
         "explicit group-count sensitivity changes the scheduled objective");
  expect(serializeProgramMechanismCost(result).getAsObject() != nullptr,
         "timing, parameters, quantities and assumptions serialize");
  llvm::json::Object parameters;
  for (const auto &entry : model.parameters)
    parameters[entry.first] = llvm::json::Object{
        {"value", *entry.second.value},
        {"unit", entry.second.unit},
        {"provenance", llvm::json::Object{{"kind", entry.second.kind},
                                          {"source", entry.second.source}}}};
  llvm::json::Value jsonModel =
      llvm::json::Object{{"schema", "orbit.direct-air.cost.v1"},
                         {"kind", "event-mechanism"},
                         {"mode", "estimate"},
                         {"model_id", "synthetic"},
                         {"boundary", model.boundary},
                         {"implementations", llvm::json::Array{"scalar"}},
                         {"parameters", std::move(parameters)}};
  llvm::json::Value config = llvm::json::Object{
      {"generic_air", llvm::json::Object{{"cost_model", jsonModel}}}};
  GenericMappingPlan priced;
  expect(planGenericMapping(analysis, config, priced, error) &&
             priced.hasCost && priced.directCost.valueMs == complete &&
             !priced.directCost.eventTiming.getAsNull(),
         "existing native evaluator consumes the public scheduled score");
  expect(checkGenericMapping(analysis, priced, error),
         "priced plan independently recomputes its score and coverage");
  auto corrupt = priced;
  corrupt.directCost.eventTiming = llvm::json::Object{{"forged", true}};
  expect(!checkGenericMapping(analysis, corrupt, error),
         "tampered timeline/price attachment rejected");
  // Backend link parameters may also be assumptions. Strict mode must not
  // launder their provenance into architecture facts during adapter fallback.
  llvm::json::Object strictParameters;
  for (const auto &entry : model.parameters) {
    if (entry.first == "communication.startup" ||
        entry.first == "communication.bandwidth")
      continue;
    strictParameters[entry.first] = llvm::json::Object{
        {"value", *entry.second.value},
        {"unit", entry.second.unit},
        {"provenance", llvm::json::Object{{"kind", "architecture"},
                                          {"source", entry.second.source}}}};
  }
  llvm::json::Value strictModel =
      llvm::json::Object{{"schema", "orbit.direct-air.cost.v1"},
                         {"kind", "event-mechanism"},
                         {"mode", "strict"},
                         {"model_id", "synthetic-link-fallback"},
                         {"boundary", model.boundary},
                         {"implementations", llvm::json::Array{"scalar"}},
                         {"parameters", std::move(strictParameters)}};
  auto architecture =
      serializeMemoryArchitecture(plan.memoryBinding.architecture);
  for (auto &entry : *architecture.getAsObject()->getArray("links")) {
    auto *link = entry.getAsObject();
    (*link)["latency_ms"] = .00005;
    (*link)["bandwidth_bytes_per_ms"] = 16000000.0;
    (*link)["provenance"] = llvm::json::Object{
        {"kind", "assumed"}, {"source", "synthetic assumed backend link"}};
  }
  auto linkedConfig = llvm::json::Value(llvm::json::Object{
      {"generic_air",
       llvm::json::Object{
           {"cost_model", llvm::json::Value(strictModel)},
           {"memory_architecture", llvm::json::Value(architecture)}}}});
  GenericMappingPlan fallback;
  expect(planGenericMapping(analysis, linkedConfig, fallback, error) &&
             !fallback.hasCost,
         "strict adapter preserves assumed target-link provenance");
  for (auto &entry : *architecture.getAsObject()->getArray("links"))
    (*entry.getAsObject()->getObject("provenance"))["kind"] = "architecture";
  (*linkedConfig.getAsObject()->getObject(
      "generic_air"))["memory_architecture"] = architecture;
  expect(planGenericMapping(analysis, linkedConfig, fallback, error) &&
             fallback.hasCost,
         "strict adapter accepts explicitly architecture-priced target links");
  std::string gemmPath = argv[1];
  gemmPath = gemmPath.substr(0, gemmPath.find_last_of('/') + 1) + "gemm.mlir";
  auto gemm = mlir::parseSourceFile<mlir::ModuleOp>(gemmPath, &context);
  GenericAnalysis gemmAnalysis;
  GenericMappingPlan gemmPlan;
  llvm::json::Value tileConfig = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{2, 2, 3}}}}};
  bool initialized =
      gemm && distributeNativeInitialization(*gemm, error) &&
      analyzeGenericProgram(*gemm, gemmAnalysis, error) &&
      planGenericMapping(gemmAnalysis, tileConfig, gemmPlan, error);
  expect(initialized, "actual distributed initializer fixture is valid");
  bool initializerPriced =
      initialized && evaluateProgramMechanismCost(gemmPlan.communication, model,
                                                  result, error);
  expect(initializerPriced && result.timing.makespanMs,
         "proved constant initializer and partial-K GEMM receive a finite "
         "complete cost");
  if (initializerPriced) {
    double initializerStores = 0, totalOperations = 0, totalAccesses = 0;
    size_t initializers = 0;
    for (const auto &quantity : result.quantities) {
      auto object = quantity.getAsObject();
      auto event = object->getInteger("event");
      if (!event)
        continue;
      auto ops = object->getNumber("source_operations"),
           accesses = object->getNumber("source_memory_accesses");
      expect(bool(ops) && bool(accesses),
             "all proved GEMM event quantities are known");
      if (ops)
        totalOperations += *ops;
      if (accesses)
        totalAccesses += *accesses;
      if (gemmPlan.communication.events[*event].kind != "initialize")
        continue;
      ++initializers;
      int64_t expected = 0;
      std::set<size_t> cells;
      for (auto id : gemmPlan.communication.events[*event].writes) {
        auto region = gemmPlan.communication
                          .versions[gemmPlan.communication.copies[id].version]
                          .region;
        expect(cells.insert(region).second,
               "initializer event has distinct actual write cells");
        int64_t count = 1;
        for (auto extent : gemmPlan.communication.regions[region].sizes)
          count *= extent;
        expected += count;
      }
      expect(ops && *ops == 0 && accesses && *accesses == expected,
             "initializer prices exactly its actual write cells, not "
             "controlling K iterations");
      if (accesses)
        initializerStores += *accesses;
    }
    expect(initializers > 0 && initializerStores == 16,
           "partial-K tiles initialize each of sixteen output elements once");
    expect(totalOperations == 256 && totalAccesses == 528,
           "real GEMM work is 256 arithmetic operations, 512 update accesses "
           "and 16 initializer stores");
  }
  std::string scalePath =
      gemmPath.substr(0, gemmPath.find_last_of('/') + 1) + "gemm-scale.mlir";
  auto gemmScale = mlir::parseSourceFile<mlir::ModuleOp>(scalePath, &context);
  GenericAnalysis scaleAnalysis;
  GenericMappingPlan scalePlan;
  bool scaleValid =
      gemmScale && distributeNativeInitialization(*gemmScale, error) &&
      analyzeGenericProgram(*gemmScale, scaleAnalysis, error) &&
      planGenericMapping(scaleAnalysis, tileConfig, scalePlan, error);
  expect(scaleValid &&
             evaluateProgramMechanismCost(scalePlan.communication, model,
                                          result, error) &&
             result.timing.makespanMs,
         "distributed initializer composes with the real GEMM-scale consumer "
         "graph");
  // Preserve a legal initializer/update mapping, but break the literal-store
  // proof with source arithmetic. Do not constant-fold or guess its work.
  auto nonconstant = mlir::parseSourceFile<mlir::ModuleOp>(gemmPath, &context);
  bool modified =
      nonconstant && distributeNativeInitialization(*nonconstant, error);
  if (modified) {
    auto function = *nonconstant->getOps<mlir::func::FuncOp>().begin();
    mlir::Operation *initializer = nullptr;
    for (auto &op : function.front())
      if (op.getName().getStringRef() == "taskflow.task") {
        initializer = &op;
        break;
      }
    mlir::affine::AffineStoreOp store;
    if (initializer)
      initializer->walk(
          [&](mlir::affine::AffineStoreOp candidate) { store = candidate; });
    modified = bool(store);
    if (store) {
      mlir::OpBuilder builder(store);
      auto value = store.getValueToStore();
      auto computed =
          builder.create<mlir::arith::AddFOp>(store.getLoc(), value, value);
      store->setOperand(0, computed);
    }
  }
  GenericAnalysis unprovedAnalysis;
  GenericMappingPlan unprovedPlan;
  bool validUnproved =
      modified &&
      analyzeGenericProgram(*nonconstant, unprovedAnalysis, error) &&
      planGenericMapping(unprovedAnalysis, tileConfig, unprovedPlan, error);
  expect(validUnproved, "arithmetic-valued initializer remains a legal "
                        "source-backed native plan");
  expect(validUnproved &&
             evaluateProgramMechanismCost(unprovedPlan.communication, model,
                                          result, error) &&
             !result.timing.makespanMs,
         "unproved nonliteral initializer retains unknown cost");
  if (validUnproved) {
    bool unknownInitializer = false;
    for (const auto &quantity : result.quantities) {
      auto object = quantity.getAsObject();
      auto event = object->getInteger("event");
      if (event &&
          unprovedPlan.communication.events[*event].kind == "initialize")
        unknownInitializer |=
            object->get("source_memory_accesses")->getAsNull().has_value();
    }
    expect(
        unknownInitializer,
        "nonliteral initializer exposes unknown work instead of a zero charge");
  }
  llvm::outs() << "{\"status\":\"" << (ok ? "passed" : "failed")
               << "\",\"checks\":" << checks
               << ",\"scope\":\"synthetic parameter provenance and graph "
                  "scheduling\"}\n";
  return ok ? 0 : 1;
}

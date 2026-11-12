#include "Taskflow/GenericNativeCompositeTiming.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace orbit::taskflow;
namespace {
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;
} // namespace
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis source;
  GenericMappingPlan native;
  GenericCostFeatures features;
  std::string error;
  unsigned checks = 0;
  bool ok = true;
  auto require = [&](bool value, const char *label) {
    if (value)
      ++checks;
    else {
      ok = false;
      llvm::errs() << label << ": " << error << "\n";
    }
    error.clear();
  };
  if (!module || !analyzeGenericProgram(*module, source, error) ||
      !planGenericMapping(source, O{}, native, error) ||
      !collectGenericCostFeatures(source, native, features, error)) {
    llvm::errs() << error << "\n";
    return 1;
  }
  A sources, implementations, operations;
  for (auto &s : features.sourceFingerprints)
    sources.push_back(s);
  for (auto &s : features.implementations)
    implementations.push_back(s);
  for (auto &s : features.operations)
    operations.push_back(s.first);
  O ranges;
  for (auto &p : features.values)
    ranges[p.first] = A{p.second, p.second};
  J target = O{{"id", "compiler-only-contract"},
               {"toolchain", "synthetic-test-toolchain"},
               {"runtime", "synthetic-test-runtime"},
               {"boundary", "prepared-command-host-to-host"}};
  // Synthetic numbers test wiring only; explicitly no hardware calibration
  // claim.
  J model = O{
      {"schema", "orbit.direct-air.cost.v1"},
      {"kind", "linear-composite"},
      {"calibration", "calibrated"},
      {"model_id", "synthetic-native-composite-test"},
      {"model_version", "1"},
      {"training_manifest_digest", std::string(64, 'a')},
      {"boundary", "prepared-command-host-to-host"},
      {"target", J(target)},
      {"domain", O{{"source_fingerprints", std::move(sources)},
                   {"implementations", std::move(implementations)},
                   {"operations", std::move(operations)},
                   {"feature_ranges", std::move(ranges)}}},
      {"terms", A{O{{"feature", "command_count"}, {"coefficient_ms", 0.123}}}}};
  ProgramTimingEvaluation timing;
  J receipt = nullptr;
  require(evaluateGenericNativeCompositeTiming(source, native,
                                               native.communication, model,
                                               target, timing, receipt, error),
          "price actual checked native graph");
  require(timing.makespanMs && std::abs(*timing.makespanMs - .123) < 1e-12,
          "complete call counted exactly once");
  require(timing.activities.size() == 1 &&
              timing.activities[0].events.size() ==
                  native.communication.events.size(),
          "one composite exclusively owns all body events");
  require(timing.activities[0].regionEvents.size() ==
                  native.communication.regionEvents.size() &&
              (native.communication.regionEvents.empty()
                   ? timing.activities[0].executionRegions.size() ==
                         native.communication.executionRegions.size()
                   : timing.activities[0].executionRegions.empty()),
          "all refined or unrefined invocation boundaries covered once");
  require(receipt.getAsObject()->getBoolean(
              "additive_body_or_lifecycle_costs") == false &&
              receipt.getAsObject()->getBoolean("complete_plan_prepricing") ==
                  false,
          "receipt forbids additive double charge and complete-plan price "
          "directory");
  auto foreignTarget = J(target);
  (*foreignTarget.getAsObject())["runtime"] = "different-runtime";
  require(evaluateGenericNativeCompositeTiming(
              source, native, native.communication, model, foreignTarget,
              timing, receipt, error) &&
              !timing.makespanMs,
          "runtime mismatch is unknown full cost, never zero");
  auto outside = J(model);
  auto *limits =
      outside.getAsObject()->getObject("domain")->getObject("feature_ranges");
  (*limits)["hardware_stage_count"] =
      A{features.values.at("hardware_stage_count") + 1,
        features.values.at("hardware_stage_count") + 1};
  require(evaluateGenericNativeCompositeTiming(
              source, native, native.communication, outside, target, timing,
              receipt, error) &&
              !timing.makespanMs,
          "outside calibrated control-count envelope remains unknown");
  auto changed = native.communication;
  changed.targetIdentity += ".different-checked-target";
  require(checkProgramGraph(changed, error) &&
              programGraphIdentity(changed) !=
                  programGraphIdentity(native.communication),
          "construct a valid distinct public target graph");
  require(evaluateGenericNativeCompositeTiming(
              source, native, changed, model, target, timing, receipt, error) &&
              !timing.makespanMs,
          "post-rebind selected graph cannot inherit old native stage counts");
  require(receipt.getAsObject()->getBoolean(
              "native_counts_applied_to_selected") == false &&
              receipt.getAsObject()->get("features")->getAsNull(),
          "mismatched graph gets no re-labelled native feature payload");
  auto invalid = native.communication;
  invalid.events.front().resource = "nonexistent-resource";
  require(!evaluateGenericNativeCompositeTiming(source, native, invalid, model,
                                                target, timing, receipt, error),
          "invalid graph rejected structurally");
  auto table = J(model);
  (*table.getAsObject())["kind"] = "observed-table";
  require(!evaluateGenericNativeCompositeTiming(source, native,
                                                native.communication, table,
                                                target, timing, receipt, error),
          "full-plan observed-price table is not this search provider");
  auto malformed = J(model);
  (*malformed.getAsObject())["terms"] =
      A{O{{"feature", "command_count"}, {"coefficient_ms", -1.}}};
  require(!evaluateGenericNativeCompositeTiming(source, native,
                                                native.communication, malformed,
                                                target, timing, receipt, error),
          "invalid negative full-call coefficient rejected");
  llvm::outs() << "generic native composite timing checks: " << checks << "\n";
  return ok ? 0 : 1;
}

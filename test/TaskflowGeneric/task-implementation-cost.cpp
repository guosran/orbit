#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdlib>

using namespace orbit::taskflow;
namespace j = llvm::json;
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto source = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  std::string error;
  int checks = 0;
  auto require = [&](bool success, const char *label) {
    ++checks;
    if (!success) {
      llvm::errs() << label << ": " << error << "\n";
      std::exit(1);
    }
    error.clear();
  };
  require(bool(source), "parse real three-task f32 source");
  require(setNativeTaskImplementations(*source, 3,
                                       {{0, "scalar", 1},
                                        {1, "scalar-product-vector-add-f32", 8},
                                        {2, "scalar", 1}},
                                       error),
          "select local scalar/mixed8/scalar implementations");
  GenericAnalysis analysis;
  GenericMappingPlan plan;
  GenericCostFeatures features;
  require(analyzeGenericProgram(*source, analysis, error) &&
              planGenericMapping(analysis, j::Object{}, plan, error) &&
              collectGenericCostFeatures(analysis, plan, features, error),
          "collect source-backed mixed local implementation features");
  require(features.values.at("explicit_task_implementation_choices") == 3 &&
              features.values.at("implemented_scalar_f32_addsub_operations") ==
                  32 &&
              features.values.at("implemented_mixed8_f32_product_operations") ==
                  16,
          "independent arithmetic oracle: two 16-add scalar tasks, one "
          "16-product mixed task");
  require(
      features.values.at("command_count") == 1 &&
          features.values.at("hardware_stage_count") == 3,
      "one command contains three independently checked native task stages");
  auto term = [](const char *feature, double coefficient) {
    return j::Value(
        j::Object{{"feature", feature}, {"coefficient_ms", coefficient}});
  };
  j::Array fingerprints, implementations, operations;
  j::Object ranges;
  for (const auto &entry : features.sourceFingerprints)
    fingerprints.push_back(entry);
  for (const auto &entry : features.implementations)
    implementations.push_back(entry);
  for (const auto &entry : features.operations)
    operations.push_back(entry.first);
  for (const auto &entry : features.values)
    ranges[entry.first] = j::Array{entry.second, entry.second};
  j::Value target = j::Object{{"id", "synthetic"},
                              {"toolchain", "test"},
                              {"runtime", "test"},
                              {"boundary", "prepared-command-host-to-host"}};
  // Synthetic API proof only, not a fitted hardware model or speedup result.
  j::Value model = j::Object{
      {"schema", "orbit.direct-air.cost.v1"},
      {"kind", "linear-composite"},
      {"calibration", "calibrated"},
      {"model_id", "synthetic-local-work-contract"},
      {"model_version", "1"},
      {"training_manifest_digest", std::string(64, '0')},
      {"boundary", "prepared-command-host-to-host"},
      {"target", j::Object{{"id", "synthetic"},
                           {"toolchain", "test"},
                           {"runtime", "test"}}},
      {"domain", j::Object{{"source_fingerprints", std::move(fingerprints)},
                           {"implementations", std::move(implementations)},
                           {"operations", std::move(operations)},
                           {"feature_ranges", std::move(ranges)}}},
      {"terms", j::Array{term("command_count", 0.1)}}};
  GenericDirectAIRCost cost;
  require(
      evaluateGenericDirectAIRCost(features, model, target, cost, error) &&
          !cost.valueMs && cost.status == "out_of_domain",
      "aggregate old linear model cannot price new explicit local mixed work");
  auto *terms = model.getAsObject()->getArray("terms");
  terms->push_back(term("implemented_scalar_f32_addsub_operations", 0.01));
  terms->push_back(term("implemented_mixed8_f32_product_operations", 0.001));
  require(evaluateGenericDirectAIRCost(features, model, target, cost, error) &&
              cost.valueMs && std::abs(*cost.valueMs - 0.436) < 1e-12,
          "implementation-resolved terms and ranges permit linear synthetic "
          "prediction");
  auto missingRange = j::Value(model);
  // Rebuild the object: pinned LLVM Object erase followed by copying is unsafe.
  auto *domain = missingRange.getAsObject()->getObject("domain");
  j::Object filtered;
  for (const auto &entry : *domain->getObject("feature_ranges"))
    if (entry.first != "implemented_mixed8_f32_product_operations")
      filtered[entry.first] = j::Value(entry.second);
  (*domain)["feature_ranges"] = std::move(filtered);
  require(evaluateGenericDirectAIRCost(features, missingRange, target, cost,
                                       error) &&
              !cost.valueMs && cost.status == "out_of_domain",
          "implementation work without a calibration range remains unknown");
  auto established = j::Value(model);
  auto *object = established.getAsObject();
  (*object)["kind"] = "implementation-composite";
  j::Object familyDomain;
  for (const auto &entry : *object->getObject("domain"))
    if (entry.first != "source_fingerprints")
      familyDomain[entry.first] = j::Value(entry.second);
  familyDomain["family_fingerprints"] =
      j::Array{features.regionFamilyFingerprint};
  familyDomain["architecture_fingerprints"] =
      j::Array{features.architectureFingerprint};
  familyDomain["allow_tails"] = false;
  (*object)["domain"] = std::move(familyDomain);
  (*object)["terms"] =
      j::Array{term("command_count", 0.1),
               term("implemented_scalar_f32_product_operations", 0.01),
               term("implemented_mixed8_f32_product_operations", 0.005),
               term("hardware_stage_count", 0.2)};
  const bool establishedValid =
      evaluateGenericDirectAIRCost(features, established, target, cost, error);
  if (establishedValid && !cost.valueMs)
    error = cost.reason;
  require(establishedValid && cost.valueMs &&
              std::abs(*cost.valueMs - 0.78) < 1e-12,
          "established strict implementation-composite remains accepted");
  llvm::outs() << "task-implementation-cost: " << checks
               << " checks passed (synthetic CPU only)\n";
}

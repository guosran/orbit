#include "Taskflow/GenericProgramContrast.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
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
  auto left = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  auto right = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!left || !right)
    return 1;
  std::string error;
  unsigned checks = 0;
  bool ok = true;
  auto expect = [&](bool v, const char *why) {
    ++checks;
    if (!v) {
      llvm::errs() << why << ": " << error << "\n";
      ok = false;
    }
    error.clear();
  };
  ProgramMappingChoice lchoice;
  lchoice.residency = "tiled-phase-local";
  lchoice.dataOrganization = "phase-local";
  lchoice.grouping = "task-shape-stages";
  ProgramMappingChoice rchoice;
  rchoice.residency = "readonly-panel";
  rchoice.dataOrganization = "retain-across-users";
  rchoice.grouping = "readonly-panel-scope";
  rchoice.storage = 0;
  rchoice.reuseLoop = 0;
  rchoice.panelLoop = 1;
  if (!setNativeMappingChoice(*left, lchoice, error) ||
      !setNativeMappingChoice(*right, rchoice, error) ||
      !setNativeAIRTileSizes(*left, 1, 16, 1, error) ||
      !setNativeAIRTileSizes(*right, 1, 16, 1, error)) {
    llvm::errs() << error;
    return 1;
  }
  GenericAnalysis la, ra;
  GenericMappingPlan lp, rp;
  llvm::json::Value config = llvm::json::Object{};
  if (!analyzeGenericProgram(*left, la, error) ||
      !analyzeGenericProgram(*right, ra, error) ||
      !planGenericMapping(la, config, lp, error) ||
      !planGenericMapping(ra, config, rp, error)) {
    llvm::errs() << error;
    return 1;
  }
  GenericProgramContrastFacts f;
  expect(collectGenericProgramContrastFacts(la, lp, ra, rp, f, error) &&
             f.supported,
         "native contrast supported");
  expect(f.changeClass == "readonly-copy-retention" &&
             f.delta["input_delivery_count"] == -3 &&
             f.delta["requested_dma_bytes"] == -192,
         "actual delivery and byte delta");
  expect(f.delta["hardware_stage_count"] == 0 &&
             f.delta["abi_to_sync_calls"] == 0 &&
             f.delta["abi_to_requested_bytes"] == 0,
         "same-stage/ABI invariants");
  auto target =
      llvm::json::Value(llvm::json::Object{{"id", "synthetic"},
                                           {"toolchain", "test"},
                                           {"runtime", "test"},
                                           {"condition", "two-context-test"}});
  auto makeModel = [&]() {
    return llvm::json::Value(llvm::json::Object{
        {"schema", "orbit.direct-air.cost.v1"},
        {"kind", "contrast-composite"},
        {"units", "ns"},
        {"model_id", "synthetic-contrast-contract"},
        {"training_manifest_digest", std::string(64, '0')},
        {"boundary", "prepared-command-host-to-host"},
        {"target", llvm::json::Value(target)},
        {"rows",
         llvm::json::Array{llvm::json::Object{
             {"change_class", f.changeClass},
             {"prediction_error_bound_ns", 25.0},
             {"domain",
              llvm::json::Object{
                  {"family_fingerprints",
                   llvm::json::Array{f.baseline.familyFingerprint}},
                  {"architecture_fingerprints",
                   llvm::json::Array{f.baseline.architectureFingerprint}},
                  {"implementations", llvm::json::Array{"scalar:1"}},
                  {"baseline",
                   llvm::json::Object{
                       {"source_task_count", llvm::json::Array{1, 1}},
                       {"input_delivery_count", llvm::json::Array{0, 4}},
                       {"requested_dma_bytes", llvm::json::Array{0, 1024}}}},
                  {"candidate",
                   llvm::json::Object{
                       {"source_task_count", llvm::json::Array{1, 1}},
                       {"input_delivery_count", llvm::json::Array{0, 4}},
                       {"requested_dma_bytes", llvm::json::Array{0, 1024}}}}}},
             {"terms",
              llvm::json::Array{
                  llvm::json::Object{{"feature", "input_delivery_count"},
                                     {"coefficient_ns", 100.0}},
                  llvm::json::Object{{"feature", "requested_dma_bytes"},
                                     {"coefficient_ns", 2.0}}}}}}}});
  };
  auto m = makeModel();
  GenericProgramContrastCost c;
  expect(evaluateGenericProgramContrast(f, m, target, c, error) && c.deltaNs &&
             std::abs(*c.deltaNs + 684) < 1e-10,
         "signed conditional delta");
  expect(c.lowerDeltaNs && *c.lowerDeltaNs == -709 && c.upperDeltaNs &&
             *c.upperDeltaNs == -659,
         "training uncertainty retained");
  auto &firstContribution = *c.contributions[0].getAsObject();
  auto originalKey =
      (*(*m.getAsObject()->getArray("rows"))[0].getAsObject()->getArray(
          "terms"))[0]
          .getAsObject()
          ->getString("feature");
  expect(firstContribution.getString("feature")->data() != originalKey->data(),
         "returned contribution owns model feature string across model "
         "destruction");
  auto serialized = serializeGenericProgramContrastCost(c);
  expect(serialized.getAsObject()->get("baseline_duration_ns")->kind() ==
                 llvm::json::Value::Null &&
             serialized.getAsObject()->get("candidate_duration_ns")->kind() ==
                 llvm::json::Value::Null,
         "absolute durations unknown");
  auto foreign = llvm::json::Value(target);
  (*foreign.getAsObject())["condition"] = "other";
  expect(evaluateGenericProgramContrast(f, m, foreign, c, error) && !c.deltaNs,
         "condition mismatch unknown");
  auto unknown = f;
  unknown.supported = false;
  unknown.reason = "unsupported changed local implementation";
  expect(evaluateGenericProgramContrast(unknown, m, target, c, error) &&
             !c.deltaNs && !c.reason.empty(),
         "unsupported pair does not cancel cost");
  auto corrupt = f;
  ++corrupt.delta["requested_dma_bytes"];
  expect(!evaluateGenericProgramContrast(corrupt, m, target, c, error),
         "mutated derived delta rejected");
  auto bad = makeModel();
  auto *row = (*bad.getAsObject()->getArray("rows"))[0].getAsObject();
  (*(*row->getArray("terms"))[0].getAsObject())["coefficient_ns"] = -1.0;
  expect(!evaluateGenericProgramContrast(f, bad, target, c, error),
         "negative component rate rejected");
  bad = makeModel();
  row = (*bad.getAsObject()->getArray("rows"))[0].getAsObject();
  row->getArray("terms")->push_back(
      llvm::json::Value((*row->getArray("terms"))[0]));
  expect(!evaluateGenericProgramContrast(f, bad, target, c, error),
         "duplicate covered term rejected");
  bad = makeModel();
  row = (*bad.getAsObject()->getArray("rows"))[0].getAsObject();
  (*row)["prediction_error_bound_ns"] = -1.0;
  expect(!evaluateGenericProgramContrast(f, bad, target, c, error),
         "invalid uncertainty rejected");
  bad = makeModel();
  (*bad.getAsObject())["training_manifest_digest"] = "invented";
  expect(!evaluateGenericProgramContrast(f, bad, target, c, error),
         "invalid calibration provenance rejected");
  bad = makeModel();
  row = (*bad.getAsObject()->getArray("rows"))[0].getAsObject();
  (*row->getObject("domain"))["family_fingerprints"] =
      llvm::json::Array{"foreign"};
  expect(evaluateGenericProgramContrast(f, bad, target, c, error) && !c.deltaNs,
         "source family OOD remains unknown");
  bad = makeModel();
  row = (*bad.getAsObject()->getArray("rows"))[0].getAsObject();
  row->getObject("domain")->getObject("baseline")->erase("requested_dma_bytes");
  expect(!evaluateGenericProgramContrast(f, bad, target, c, error),
         "priced feature requires a frozen domain on both plans");
  GenericProgramContrastFacts unchanged;
  expect(collectGenericProgramContrastFacts(la, lp, la, lp, unchanged, error) &&
             !unchanged.supported,
         "no invented contrast on unchanged graph");
  auto invalid = rp;
  invalid.choices.residency = "phase-local";
  expect(!collectGenericProgramContrastFacts(la, lp, ra, invalid, unchanged,
                                             error),
         "unbound mapping choice rejected");
  GenericDirectAIRCost full;
  GenericCostFeatures features;
  expect(collectGenericCostFeatures(la, lp, features, error) &&
             evaluateGenericDirectAIRCost(features, m, target, full, error) &&
             !full.valueMs,
         "contrast calibration cannot become absolute search score");
  llvm::outs() << "{\"status\":\"" << (ok ? "passed" : "failed")
               << "\",\"checks\":" << checks
               << ",\"parameters\":\"synthetic contract only\"}\n";
  return ok ? 0 : 1;
}

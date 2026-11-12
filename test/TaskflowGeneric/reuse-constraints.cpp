#include "Taskflow/GenericReuse.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  GenericAnalysis analysis;
  std::string error;
  unsigned checks = 0;
  auto require = [&](bool value, const char *what) {
    if (value)
      ++checks;
    else
      llvm::errs() << what << ": " << error << "\n";
    return value;
  };
  if (!require(module && analyzeGenericProgram(*module, analysis, error),
               "source analysis"))
    return 1;
  GenericReuseOptions options;
  options.catalogue = true;
  options.includeBaseline = true;
  options.currentTile = {12, 136, 1};
  options.maxTileProposals = 16;
  options.maxScopeSets = 256;
  options.maxCatalogueConfigurations = 4096;
  options.maxAttempts = 8192;
  options.maxPlannerCalls = 4096;
  llvm::json::Value config = llvm::json::Object{};
  if (argc == 3) {
    auto bytes = llvm::MemoryBuffer::getFile(argv[2]);
    if (!bytes)
      return 2;
    auto parsed = llvm::json::parse((*bytes)->getBuffer());
    if (!parsed) {
      llvm::errs() << llvm::toString(parsed.takeError());
      return 2;
    }
    config = std::move(*parsed);
  }
  clearGenericReuseCache();
  auto before = genericMappingStatistics();
  auto first = discoverGenericReuseConfigurations(analysis, config, options);
  auto after = genericMappingStatistics();
  if (!require(first.catalogueComplete && !first.proposals.empty(),
               "complete bounded catalogue"))
    return 1;
  if (!require(first.stats.plannerCalls ==
                       after.plannerCalls - before.plannerCalls &&
                   first.stats.checkerCalls ==
                       after.checkerCalls - before.checkerCalls &&
                   after.costProviderCalls == before.costProviderCalls,
               "raw unpriced planner/check counters"))
    return 1;
  bool nonhalf = false, multiple = false, tailDivisor = false, halving = false;
  std::set<std::string> ids;
  for (const auto &p : first.proposals) {
    auto *d = p.diagnostics.getAsObject();
    ids.insert(d->getString("configuration_id")->str());
    multiple |= p.mapping.residencyGroups.size() > 1;
    nonhalf |= p.tileSizes == std::vector<int64_t>({11, 136, 1});
    tailDivisor |= p.tileSizes == std::vector<int64_t>({17, 68, 1});
    halving |= p.tileSizes == std::vector<int64_t>({6, 136, 1});
    auto lower = d->getInteger("working_set_lower_bound_bytes");
    auto *metrics = d->getObject("metrics");
    if (lower && *lower >= 0 &&
        !require(*lower <= *metrics->getInteger("conservative_peak_bytes"),
                 "necessary pool lower bound never exceeds checked payload"))
      return 1;
  }
  if (!require(nonhalf, "non-half capacity boundary 11x136") ||
      !require(multiple, "complete multigroup configurations") ||
      !require(tailDivisor, "full-axis divisor tile17x68") ||
      !require(halving, "historical half tile remains fallback"))
    return 1;
  auto second = discoverGenericReuseConfigurations(analysis, config, options);
  if (!require(second.catalogueId == first.catalogueId &&
                   second.stats.cacheHits > 0 &&
                   second.stats.plannerCalls == 0 &&
                   second.stats.checkerCalls == 0,
               "checked-trial cache eliminates repeated planning"))
    return 1;
  auto model = config;
  (*model.getAsObject())["model_version"] = "different-model-version";
  auto invalidated =
      discoverGenericReuseConfigurations(analysis, model, options);
  if (!require(invalidated.stats.cacheHits == 0 &&
                   invalidated.stats.plannerCalls > 0 &&
                   invalidated.catalogueId != first.catalogueId,
               "model identity invalidates cache"))
    return 1;
  auto target = llvm::json::Value(llvm::json::Object{
      {"generic_air", llvm::json::Object{{"capacity_bytes", 24560}}}});
  auto targetChanged =
      discoverGenericReuseConfigurations(analysis, target, options);
  if (!require(targetChanged.stats.cacheHits == 0 &&
                   targetChanged.catalogueId != first.catalogueId,
               "target capacity invalidates cache"))
    return 1;
  // Parent source choices do not change the frozen root catalogue domain.
  auto group = first.proposals.back().mapping;
  if (!setNativeAIRTileSizes(*module, 3, 68, 1, error) ||
      !setNativeMappingChoice(*module, group, error) ||
      !analyzeGenericProgram(*module, analysis, error))
    return 1;
  auto fromParent =
      discoverGenericReuseConfigurations(analysis, config, options);
  if (!require(fromParent.catalogueId == first.catalogueId &&
                   fromParent.stats.plannerCalls == 0,
               "root catalogue independent of parent mapping/tile"))
    return 1;
  auto anotherTile = options;
  anotherTile.maxTileProposals = 1;
  anotherTile.currentTile = {9, 104, 1};
  auto tileChanged =
      discoverGenericReuseConfigurations(analysis, config, anotherTile);
  if (!require(tileChanged.stats.cacheHits == 0 &&
                   tileChanged.stats.plannerCalls > 0 &&
                   tileChanged.catalogueId != first.catalogueId,
               "tile tuple invalidates checked-trial cache"))
    return 1;
  mlir::OwningOpRef<mlir::ModuleOp> changedSource(
      mlir::cast<mlir::ModuleOp>(module->clone()));
  bool changed = false;
  changedSource->walk([&](mlir::arith::ConstantFloatOp op) {
    if (!changed) {
      op.setValueAttr(mlir::FloatAttr::get(op.getType(), 1.5));
      changed = true;
    }
  });
  GenericAnalysis changedFacts;
  if (!require(changed &&
                   analyzeGenericProgram(*changedSource, changedFacts, error),
               "modified source analysis"))
    return 1;
  auto sourceChanged =
      discoverGenericReuseConfigurations(changedFacts, config, options);
  if (!require(sourceChanged.stats.cacheHits == 0 &&
                   sourceChanged.stats.plannerCalls > 0 &&
                   sourceChanged.catalogueId != first.catalogueId,
               "actual arithmetic source invalidates cache"))
    return 1;
  auto hard = options;
  hard.requiredAlignment = {1, 8, 1};
  hard.requireDivisible = true;
  auto restricted = discoverGenericReuseConfigurations(analysis, config, hard);
  bool badAlignment = false, badTail = false;
  for (const auto &r : restricted.rejected) {
    auto *d = r.getAsObject();
    auto why = d->getString("reason").value_or("");
    badAlignment |= why.contains("alignment");
    badTail |= why.contains("tail");
  }
  for (const auto &p : restricted.proposals)
    if (!require(p.tileSizes[1] % 8 == 0 && 17 % p.tileSizes[0] == 0 &&
                     136 % p.tileSizes[1] == 0,
                 "hard restrictions constrain all admitted tiles"))
      return 1;
  if (!require(badAlignment && badTail,
               "invalid explicit alignment/tail rejected"))
    return 1;
  auto invalidOptions = options;
  invalidOptions.requiredAlignment = {0, 8, 1};
  auto malformed =
      discoverGenericReuseConfigurations(analysis, config, invalidOptions);
  if (!require(!malformed.catalogueComplete && malformed.proposals.empty() &&
                   !malformed.stopReason.empty(),
               "malformed required alignment rejected"))
    return 1;
  auto limited = options;
  limited.maxPlannerCalls = 1;
  clearGenericReuseCache();
  auto stopped = discoverGenericReuseConfigurations(analysis, config, limited);
  if (!require(!stopped.catalogueComplete && stopped.stats.plannerCalls <= 1 &&
                   stopped.stopReason == "max_generator_planner_calls" &&
                   stopped.catalogueId != first.catalogueId,
               "internal probes obey planner budget and bind admitted prefix"))
    return 1;
  auto tinyDomain = options;
  tinyDomain.maxComputeInstances = 1;
  auto domain =
      discoverGenericReuseConfigurations(analysis, config, tinyDomain);
  if (!require(
          domain.catalogueComplete && domain.proposals.empty() &&
              domain.stats.plannerCalls == 0 &&
              domain.stats.domainExclusions ==
                  domain.stats.generationAttempts &&
              domain.catalogueId != first.catalogueId,
          "common compute-instance optimizer bound screens before planner"))
    return 1;
  auto historical = options;
  historical.tileGenerator = "historical-halving";
  historical.maxTileProposals = 4;
  auto old = discoverGenericReuseConfigurations(analysis, config, historical);
  bool oldHalf = false;
  for (const auto &p : old.proposals)
    oldHalf |= p.tileSizes == std::vector<int64_t>({6, 136, 1});
  if (!require(oldHalf && old.catalogueId != first.catalogueId,
               "historical halving option retained"))
    return 1;
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks
               << ",\"configurations\":" << first.proposals.size()
               << ",\"planner_calls\":" << first.stats.plannerCalls
               << ",\"cache_hits\":" << second.stats.cacheHits << "}\n";
  return 0;
}

#include "Taskflow/GenericReuse.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace orbit::taskflow;
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  unsigned checks = 0;
  std::string error;
  auto require = [&](bool ok, const char *name) {
    if (ok)
      ++checks;
    else
      llvm::errs() << name << ": " << error << "\n";
    return ok;
  };
  auto signatures = [](const GenericAnalysis &a) {
    std::set<std::string> result;
    for (const auto &o : discoverGenericReuseOpportunities(a)) {
      std::string s = a.storages[o.storage].id + ":" + o.version + ":" +
                      (o.producer < 0 ? "input" : a.tasks[o.producer].id);
      for (auto c : o.consumers)
        s += ":" + a.tasks[c].id;
      for (auto r : o.region)
        s += ":" + std::to_string(r.lower) + "," + std::to_string(r.upper);
      result.insert(s);
    }
    return result;
  };
  for (int caseNo = 0; caseNo < 2; ++caseNo) {
    auto module =
        mlir::parseSourceFile<mlir::ModuleOp>(argv[1 + 2 * caseNo], &context);
    auto permuted =
        mlir::parseSourceFile<mlir::ModuleOp>(argv[2 + 2 * caseNo], &context);
    GenericAnalysis a, b;
    if (!require(module && permuted &&
                     analyzeGenericProgram(*module, a, error) &&
                     analyzeGenericProgram(*permuted, b, error),
                 "analyze original/permutation"))
      return 1;
    if (!require(signatures(a) == signatures(b),
                 "topological arrangement invariant opportunities"))
      return 1;
    if (!setNativeAIRTileSizes(*module, 12, 136, 1, error) ||
        !analyzeGenericProgram(*module, a, error))
      return 1;
    llvm::json::Value config = llvm::json::Object{};
    auto discovered = discoverGenericReuseProposals(a, config, 32, 4, 4);
    if (!require(!discovered.proposals.empty(), "joint proposals found"))
      return 1;
    bool capacityRejected = false, smaller = false, noncontiguous = false,
         external = false, boundary = false;
    for (const auto &r : discovered.rejected)
      if (auto *o = r.getAsObject())
        if (auto why = o->getString("reason"))
          capacityRejected |= why->contains("capacity");
    for (const auto &p : discovered.proposals) {
      for (const auto &g : p.mapping.residencyGroups) {
        for (size_t i = 1; i < g.tasks.size(); ++i)
          noncontiguous |= g.tasks[i] > g.tasks[i - 1] + 1;
        for (auto s : g.retainedStorages)
          external |= a.storages[s].role == "input";
      }
      smaller |= p.tileSizes[0] < 12 || p.tileSizes[1] < 136;
      if (auto *d = p.diagnostics.getAsObject())
        if (auto *exports = d->getArray("boundary_exports"))
          for (const auto &e : *exports)
            if (auto *x = e.getAsArray())
              boundary |= !x->empty();
    }
    if (!require(noncontiguous, "legal noncontiguous groups discovered"))
      return 1;
    if (caseNo == 0 &&
        (!require(capacityRejected && smaller,
                  "capacity feedback repairs incompatible original tile") ||
         !require(boundary, "partial consumers preserve boundary exports")))
      return 1;
    if (caseNo == 1 &&
        !require(external,
                 "pure external shared-input opportunity materialized"))
      return 1;
  }
  GenericAnalysis synthetic;
  synthetic.tasks.resize(4);
  synthetic.tasks[2].dependencies = {0};
  synthetic.tasks[3].dependencies = {2};
  std::vector<size_t> order;
  if (!require(deriveGenericResidencyOrder(synthetic, {{{0, 3}, {0}}}, order,
                                           error) == false,
               "nonconvex quotient cycle rejected"))
    return 1;
  if (!require(deriveGenericResidencyOrder(synthetic, {{{0, 2}, {0}}}, order,
                                           error) &&
                   order == std::vector<size_t>({0, 2, 1, 3}),
               "convex contraction preserves source identity"))
    return 1;
  synthetic.tasks[2].dependencies.push_back(1);
  if (!require(deriveGenericResidencyOrder(synthetic, {{{0, 2}, {0}}}, order,
                                           error) &&
                   order == std::vector<size_t>({1, 0, 2, 3}),
               "external order predecessor retained"))
    return 1;
  if (!require(!deriveGenericResidencyOrder(synthetic, {{{2, 0}, {0}}}, order,
                                            error),
               "internal order violation rejected"))
    return 1;
  if (!require(!deriveGenericResidencyOrder(
                   synthetic, {{{0, 2}, {0}}, {{2, 3}, {1}}}, order, error),
               "overlapping contraction rejected"))
    return 1;
  llvm::outs() << "{\"status\":\"passed\",\"checks\":" << checks << "}\n";
  return 0;
}

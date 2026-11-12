#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "Taskflow/ProgramPartition.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <limits>
#include <map>
using namespace mlir;
using namespace orbit::taskflow;
static const char *fixture = R"mlir(module {
  func.func @gemm_scale(%A: memref<17x8xf32> {orbit.role = "input", orbit.name = "A"}, %B: memref<8x4xf32> {orbit.role = "input", orbit.name = "B"}, %Y: memref<17x4xf32> {orbit.role = "output", orbit.name = "Y"}) attributes {orbit.noalias, llvm.emit_c_interface} {
    %C = memref.alloc() {orbit.name = "intermediate"} : memref<17x4xf32>
    %C_done = taskflow.task @gemm will_reads(%A, %B : memref<17x8xf32>, memref<8x4xf32>) will_writes(%C : memref<17x4xf32>) [original_read_memrefs(%A, %B : memref<17x8xf32>, memref<8x4xf32>), original_write_memrefs(%C : memref<17x4xf32>)] : (memref<17x8xf32>, memref<8x4xf32>, memref<17x4xf32>) -> memref<17x4xf32> {
    ^bb0(%A_b: memref<17x8xf32>, %B_b: memref<8x4xf32>, %C_b: memref<17x4xf32>):
      %zero = arith.constant 0.0 : f32
      affine.for %i = 0 to 17 {
        affine.for %j = 0 to 4 {
          affine.store %zero, %C_b[%i, %j] : memref<17x4xf32>
          affine.for %k = 0 to 8 {
            %x = affine.load %A_b[%i, %k] : memref<17x8xf32>
            %w = affine.load %B_b[%k, %j] : memref<8x4xf32>
            %old = affine.load %C_b[%i, %j] : memref<17x4xf32>
            %p = arith.mulf %x, %w : f32
            %sum = arith.addf %old, %p : f32
            affine.store %sum, %C_b[%i, %j] : memref<17x4xf32>
          }
        }
      }
      taskflow.yield done_writes(%C_b : memref<17x4xf32>)
    }
    %Y_done = taskflow.task @scale will_reads(%C_done : memref<17x4xf32>) will_writes(%Y : memref<17x4xf32>) [original_read_memrefs(%C_done : memref<17x4xf32>), original_write_memrefs(%Y : memref<17x4xf32>)] : (memref<17x4xf32>, memref<17x4xf32>) -> memref<17x4xf32> {
    ^bb0(%tmp: memref<17x4xf32>, %out: memref<17x4xf32>):
      %factor = arith.constant 0.5 : f32
      affine.for %i = 0 to 17 {
        affine.for %j = 0 to 4 {
          %x = affine.load %tmp[%i, %j] : memref<17x4xf32>
          %y = arith.mulf %x, %factor : f32
          affine.store %y, %out[%i, %j] : memref<17x4xf32>
        }
      }
      taskflow.yield done_writes(%out : memref<17x4xf32>)
    }
    memref.dealloc %C : memref<17x4xf32>
    return
  }
}
 )mlir";
std::string replaceAll(std::string value, const std::string &from,
                       const std::string &to) {
  size_t at = 0;
  while ((at = value.find(from, at)) != std::string::npos) {
    value.replace(at, from.size(), to);
    at += to.size();
  }
  return value;
}
int main() {
  DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, affine::AffineDialect,
                  arith::ArithDialect, math::MathDialect, memref::MemRefDialect,
                  func::FuncDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  unsigned checks = 0;
  bool ok = true;
  std::string error;
  auto require = [&](bool value, StringRef name) {
    if (!value) {
      llvm::errs() << "partition test failed: " << name << ": " << error
                   << "\n";
      ok = false;
    } else
      ++checks;
    return value;
  };
  auto source = parseSourceString<ModuleOp>(fixture, &context);
  GenericAnalysis analysis;
  ProgramPartitionPlan plan;
  if (!source ||
      !require(analyzeGenericProgram(*source, analysis, error),
               "analyze complete two-task source") ||
      !require(planProgramPartition(analysis, 8, plan, error),
               "derive independent whole-program axis"))
    return 1;
  require(plan.extent == 17 && plan.chunk == 8 &&
              plan.storageAxes == std::vector<int64_t>{0, -1, 0, 0},
          "sliced and replicated ABI axes");
  // No source arithmetic may be replaced by a kernel recipe. Compare actual
  // operation counts and attributes, and explicitly check every inner bound.
  auto arithmetic = [](const GenericAnalysis &a) {
    std::map<std::string, int> signature;
    for (const auto &task : a.tasks)
      task.operation->walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() != "arith")
          return;
        std::string key;
        llvm::raw_string_ostream os(key);
        os << op->getName() << " ";
        op->getAttrDictionary().print(os);
        ++signature[key];
      });
    return signature;
  };
  for (int64_t extent : {8, 1}) {
    OwningOpRef<ModuleOp> child;
    GenericAnalysis facts;
    bool good =
        materializeProgramPartition(*source, plan, extent, child, error) &&
        succeeded(verify(*child)) &&
        analyzeGenericProgram(*child, facts, error);
    if (!require(good, "materialize full and tail complete-program templates"))
      continue;
    bool shapes = true;
    for (size_t s = 0; s < facts.storages.size(); ++s) {
      auto expected = analysis.storages[s].shape;
      if (plan.storageAxes[s] >= 0)
        expected[plan.storageAxes[s]] = extent;
      shapes &= facts.storages[s].shape == expected;
    }
    require(shapes, "only declared storage dimensions shrink");
    bool domains = facts.tasks.size() == analysis.tasks.size();
    for (size_t t = 0; t < facts.tasks.size(); ++t) {
      domains &= facts.tasks[t].loops.size() == analysis.tasks[t].loops.size();
      for (size_t l = 0; l < facts.tasks[t].loops.size(); ++l) {
        const auto &x = facts.tasks[t].loops[l],
                   &y = analysis.tasks[t].loops[l];
        domains &= x.lower == y.lower && x.step == y.step &&
                   x.upper == (l == 0 ? extent : y.upper);
      }
    }
    require(domains,
            "source reduction order and every nonpartition bound preserved");
    require(arithmetic(facts) == arithmetic(analysis),
            "actual arithmetic and constants preserved");
    // Reuse generic mapping's capacity/layout/version checks on each template.
    require(distributeNativeInitialization(*child, error) &&
                analyzeGenericProgram(*child, facts, error),
            "generic initialization distribution on template");
    GenericMappingPlan mapping;
    llvm::json::Value config = llvm::json::Object{
        {"generic_air",
         llvm::json::Object{{"tile_sizes", llvm::json::Array{2, 2, 4}}}}};
    require(planGenericMapping(facts, config, mapping, error) &&
                checkGenericMapping(facts, mapping, error) &&
                mapping.residentBytes <= 24576,
            "template uses existing mapping/capacity proof");
  }
  auto manifest = serializeProgramPartition(analysis, plan);
  auto *root = manifest.getAsObject();
  auto *invocations = root->getArray("invocations");
  auto *templates = root->getArray("templates");
  require(invocations && invocations->size() == 3 && templates &&
              templates->size() == 2,
          "M17/chunk8 has two templates and three commands");
  int64_t cursor = 0;
  bool coverage = true, bindings = true;
  for (size_t i = 0; i < invocations->size(); ++i) {
    auto *inv = (*invocations)[i].getAsObject();
    int64_t lo = *inv->getInteger("lower"), hi = *inv->getInteger("upper");
    coverage &= lo == cursor && hi > lo && hi <= 17;
    cursor = hi;
    auto *deps = inv->getArray("depends_on");
    coverage &= deps->size() == (i ? 1 : 0);
    if (i)
      coverage &= (*deps)[0].getAsInteger() == int64_t(i - 1);
    for (auto &entry : *inv->getArray("bindings")) {
      auto *b = entry.getAsObject();
      size_t s = *b->getInteger("full_buffer_index");
      auto *offsets = b->getArray("offsets"), *sizes = b->getArray("sizes");
      for (size_t d = 0; d < analysis.storages[s].shape.size(); ++d) {
        bool sliced = int64_t(d) == plan.storageAxes[s];
        bindings &= (*offsets)[d].getAsInteger() == (sliced ? lo : 0);
        bindings &= (*sizes)[d].getAsInteger() ==
                    (sliced ? hi - lo : analysis.storages[s].shape[d]);
      }
    }
  }
  require(coverage && cursor == 17,
          "command intervals cover every output row exactly once");
  require(bindings, "manifest binds exact slices and full replicated inputs");
  auto rejectWitness = [&](StringRef name,
                           std::function<void(ProgramPartitionPlan &)> edit,
                           int64_t extent = 8) {
    auto bad = plan;
    edit(bad);
    OwningOpRef<ModuleOp> child;
    error.clear();
    require(!materializeProgramPartition(*source, bad, extent, child, error),
            name);
  };
  rejectWitness("wrong body fingerprint",
                [](auto &p) { p.sourceFingerprint = "forged"; });
  rejectWitness("wrong plan fingerprint", [](auto &p) { p.planId = "forged"; });
  rejectWitness("wrong witness source extent", [](auto &p) { ++p.extent; });
  rejectWitness("wrong witness chunk", [](auto &p) { ++p.chunk; });
  rejectWitness("wrong storage axis", [](auto &p) { p.storageAxes[0] = 1; });
  rejectWitness("axis index out of bounds",
                [](auto &p) { p.storageAxes[0] = 99; });
  rejectWitness("missing ABI axis", [](auto &p) { p.storageAxes.pop_back(); });
  for (int64_t extent : {-1, 0, 2, 9, 18})
    rejectWitness("invalid full/tail template extent", [](auto &) {}, extent);
  for (int64_t chunk : std::vector<int64_t>{-1, 0, INT64_MAX}) {
    ProgramPartitionPlan bad;
    error.clear();
    require(!planProgramPartition(analysis, chunk, bad, error),
            "invalid chunk rejected");
  }
  auto rejectSource = [&](StringRef name, std::string text,
                          StringRef diagnostic) {
    auto module = parseSourceString<ModuleOp>(text, &context);
    GenericAnalysis facts;
    ProgramPartitionPlan bad;
    error.clear();
    bool analyzed = module && analyzeGenericProgram(*module, facts, error);
    bool accepted = analyzed && planProgramPartition(facts, 8, bad, error);
    require(!accepted && error.find(diagnostic.str()) != std::string::npos,
            name);
  };
  rejectSource("partition IV used as a scalar value",
               replaceAll(fixture, "affine.for %i = 0 to 17 {",
                          "affine.for %i = 0 to 17 {\n %observed = "
                          "arith.index_cast %i : index to i32"),
               "axis used as a scalar value");
  rejectSource("non-independent write",
               replaceAll(fixture, "affine.store %y, %out[%i, %j]",
                          "affine.store %y, %out[0, %j] : memref<17x4xf32>\n "
                          "affine.store %y, %out[%i, %j]"),
               "write crosses independent");
  auto partial = std::string(fixture);
  auto last = partial.rfind("affine.for %j = 0 to 4");
  partial.replace(last, std::string("affine.for %j = 0 to 4").size(),
                  "affine.for %j = 0 to 3");
  rejectSource("incomplete final output", partial,
               "output not completely initialized");
  auto different = std::string(fixture);
  last = different.rfind("affine.for %i = 0 to 17");
  different.replace(last, std::string("affine.for %i = 0 to 17").size(),
                    "affine.for %i = 0 to 16");
  rejectSource("tasks with different outer domains", different,
               "output not completely initialized");
  auto square = replaceAll(replaceAll(fixture, "17x4", "17x17"), "8x4", "8x17");
  square = replaceAll(square, "0 to 4", "0 to 17");
  rejectSource("shared storage has inconsistent partition axis",
               replaceAll(square, "%tmp[%i, %j]", "%tmp[%j, %i]"),
               "shared storage uses inconsistent");
  rejectSource("out-of-bounds source access",
               replaceAll(fixture, "%A_b[%i, %k]", "%A_b[%i + 1, %k]"),
               "out-of-bounds");
  rejectSource("nonidentity layout uses existing analysis rejection",
               replaceAll(fixture, "17x8xf32>", "17x8xf32, strided<[9, 1]>>"),
               "identity-layout");
  // Independence is about source access projections, not the spelling of M
  // or a hardcoded first tensor dimension. A transposed shared axis is legal.
  auto transposed = replaceAll(replaceAll(fixture, "17x8xf32", "8x17xf32"),
                               "17x4xf32", "4x17xf32");
  transposed = replaceAll(transposed, "%A_b[%i, %k]", "%A_b[%k, %i]");
  for (auto name : {"%C_b", "%tmp", "%out"})
    transposed = replaceAll(transposed, std::string(name) + "[%i, %j]",
                            std::string(name) + "[%j, %i]");
  auto transposeModule = parseSourceString<ModuleOp>(transposed, &context);
  GenericAnalysis transposeFacts;
  ProgramPartitionPlan transposePlan;
  error.clear();
  require(transposeModule &&
              analyzeGenericProgram(*transposeModule, transposeFacts, error) &&
              planProgramPartition(transposeFacts, 8, transposePlan, error) &&
              transposePlan.storageAxes == std::vector<int64_t>{1, -1, 1, 1},
          "generic shared axis can be a nonleading storage dimension");
  OwningOpRef<ModuleOp> transposeChild;
  require(transposeModule &&
              materializeProgramPartition(*transposeModule, transposePlan, 1,
                                          transposeChild, error) &&
              succeeded(verify(*transposeChild)),
          "transposed tail template remains structurally valid");
  // Two runners remain alive together. Repeated full-sized invocations reuse
  // one runner's BOs; a distinct tail template retains its own allocations.
  auto accounting = parseSourceString<ModuleOp>(fixture, &context);
  (*accounting)
      ->setAttr("orbit.program_partition_extent",
                IntegerAttr::get(IntegerType::get(&context, 64), 8));
  llvm::json::Value cfg = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{8, 4, 8}}}}};
  error.clear();
  auto evaluated = evaluatePartitionedGenericMLIR(*accounting, cfg, error);
  auto response = evaluated.getAsObject();
  auto parent = response ? response->getObject("best_plan") : nullptr;
  require(
      parent && parent->getInteger("system_memory_peak_bytes") == 832,
      "parent BO peak sums distinct 640B full and 192B tail template runners");
  bool templateCounts =
      parent && parent->getArray("template_system_memories") &&
      parent->getArray("template_system_memories")->size() == 2;
  int64_t reportedSum = 0, weightedSum = 0;
  if (templateCounts)
    for (auto &value : *parent->getArray("template_system_memories")) {
      auto &record = *value.getAsObject();
      auto bytes = *record.getInteger("system_memory_peak_bytes");
      reportedSum += bytes;
      weightedSum += bytes * *record.getInteger("invocation_count");
      templateCounts &= record.getString("lifetime").has_value();
    }
  require(templateCounts && reportedSum == 832 && weightedSum == 1472 &&
              parent->getString("system_memory_accounting_scope").has_value() &&
              parent->getString("host_buffer_lifetime_policy").has_value(),
          "per-template lifetimes expose persistent BO reuse rather than "
          "multiplying by calls");
  llvm::json::Value otherCfg = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{4, 2, 8}}}}};
  auto other = evaluatePartitionedGenericMLIR(*accounting, otherCfg, error);
  auto otherParent = other.getAsObject()->getObject("best_plan");
  require(parent && otherParent &&
              parent->getString("partition_witness_id") ==
                  otherParent->getString("partition_witness_id") &&
              parent->getString("plan_id") !=
                  otherParent->getString("plan_id") &&
              parent->getArray("template_plans")
                      ->front()
                      .getAsObject()
                      ->getObject("plan")
                      ->getString("plan_id") !=
                  otherParent->getArray("template_plans")
                      ->front()
                      .getAsObject()
                      ->getObject("plan")
                      ->getString("plan_id"),
          "same source partition with different target tile config has "
          "distinct full mapping identity");
  int64_t metricBytes = 0;
  double metricMs = 0;
  error.clear();
  require(accumulateProgramPartitionMetrics(12, 3, 0.25, metricBytes, metricMs,
                                            error) &&
              metricBytes == 36 && metricMs == 0.75,
          "repeated template transfer and illustrative cost totals");
  auto rejectMetric = [&](int64_t bytes, int64_t count, double ms,
                          int64_t initialBytes, double initialMs,
                          StringRef name) {
    int64_t totalBytes = initialBytes;
    double totalMs = initialMs;
    error.clear();
    require(!accumulateProgramPartitionMetrics(bytes, count, ms, totalBytes,
                                               totalMs, error) &&
                totalBytes == initialBytes && totalMs == initialMs,
            name);
  };
  rejectMetric(INT64_MAX, 2, 0, 0, 0,
               "transfer multiplication overflow fails atomically");
  rejectMetric(1, 1, 0, INT64_MAX, 0,
               "transfer addition overflow fails atomically");
  rejectMetric(1, 0, 0, 0, 0, "nonpositive invocation count rejected");
  rejectMetric(-1, 1, 0, 0, 0, "negative transfer count rejected");
  rejectMetric(1, 2, std::numeric_limits<double>::max(), 0, 0,
               "theoretical cost multiplication overflow rejected");
  rejectMetric(1, 1, std::numeric_limits<double>::max(), 0,
               std::numeric_limits<double>::max(),
               "theoretical cost addition overflow rejected");
  rejectMetric(1, 1, std::numeric_limits<double>::infinity(), 0, 0,
               "nonfinite theoretical coefficient rejected");
  metricBytes = 0;
  metricMs = 0;
  require(accumulateProgramPartitionMetrics(1, 2, 0, metricBytes, metricMs,
                                            error) &&
              metricBytes == 2 && metricMs == 0,
          "unknown cost zero accumulator remains separate from uncalibrated "
          "status");
  const char *overflow =
      R"mlir(module attributes {orbit.program_partition_extent = 2 : i64} {
    func.func @huge_metadata(%B: memref<2000000000x1000000000xf32> {orbit.role = "input"}, %Y: memref<5xf32> {orbit.role = "output"}) attributes {orbit.noalias} {
      %done = taskflow.task @copy will_reads(%B : memref<2000000000x1000000000xf32>) will_writes(%Y : memref<5xf32>) [original_read_memrefs(%B : memref<2000000000x1000000000xf32>), original_write_memrefs(%Y : memref<5xf32>)] : (memref<2000000000x1000000000xf32>, memref<5xf32>) -> memref<5xf32> {
      ^bb0(%b: memref<2000000000x1000000000xf32>, %y: memref<5xf32>):
        affine.for %i = 0 to 5 {
          %v = affine.load %b[0, 0] : memref<2000000000x1000000000xf32>
          affine.store %v, %y[%i] : memref<5xf32>
        }
        taskflow.yield done_writes(%y : memref<5xf32>)
      }
      return
    }
  })mlir";
  auto huge = parseSourceString<ModuleOp>(overflow, &context);
  llvm::json::Value hugeCfg = llvm::json::Object{
      {"generic_air",
       llvm::json::Object{{"tile_sizes", llvm::json::Array{2}}}}};
  error.clear();
  auto rejected = evaluatePartitionedGenericMLIR(*huge, hugeCfg, error);
  require(rejected.getAsObject()->getString("status") == "unsupported" &&
              error.find("retained template BO system-memory sum overflow") !=
                  std::string::npos,
          "sum overflow rejected without allocating enormous metadata-only "
          "input BOs");
  auto expanded = replaceAll(replaceAll(fixture, "17x", "1048577x"), "0 to 17",
                             "0 to 1048577");
  auto expandedModule = parseSourceString<ModuleOp>(expanded, &context);
  GenericAnalysis expandedFacts;
  ProgramPartitionPlan expandedPlan;
  error.clear();
  require(expandedModule &&
              analyzeGenericProgram(*expandedModule, expandedFacts, error) &&
              !planProgramPartition(expandedFacts, 1, expandedPlan, error) &&
              error.find("host-plan invocation expansion") != std::string::npos,
          "huge invocation expansion fails before materializing host "
          "invocation JSON");
  llvm::outs() << "{\"passed\":" << (ok ? "true" : "false")
               << ",\"checks\":" << checks << "}\n";
  return ok ? 0 : 1;
}

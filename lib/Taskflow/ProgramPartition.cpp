#include "Taskflow/ProgramPartition.h"
#include "Taskflow/GenericMapping.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
using namespace mlir;
using namespace orbit::taskflow;
namespace j = llvm::json;
namespace {
constexpr int64_t maxProgramInvocations = 1048576;
bool fail(std::string &e, const std::string &s) {
  e = "program partition: " + s;
  return false;
}
std::string hash(llvm::StringRef text) {
  auto bytes = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(text.data()), text.size()));
  std::string result;
  for (auto c : bytes) {
    result += "0123456789abcdef"[c >> 4];
    result += "0123456789abcdef"[c & 15];
  }
  return result;
}
j::Array integers(const std::vector<int64_t> &values) {
  j::Array result;
  for (auto v : values)
    result.push_back(v);
  return result;
}
std::vector<size_t> abiOrder(const GenericAnalysis &a) {
  std::vector<size_t> result;
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].role != "temporary")
      result.push_back(i);
  for (size_t i = 0; i < a.storages.size(); ++i)
    if (a.storages[i].role == "temporary")
      result.push_back(i);
  return result;
}
} // namespace
bool orbit::taskflow::accumulateProgramPartitionMetrics(
    int64_t transferBytes, int64_t count, double estimatedMs,
    int64_t &totalBytes, double &totalMs, std::string &e) {
  if (transferBytes < 0 || count <= 0 || totalBytes < 0 ||
      !std::isfinite(estimatedMs) || estimatedMs < 0 ||
      !std::isfinite(totalMs) || totalMs < 0)
    return fail(e, "invalid template transfer/count/cost metric");
  if (transferBytes && count > INT64_MAX / transferBytes)
    return fail(e, "template transfer byte multiplication overflow");
  int64_t repeated = transferBytes * count;
  if (totalBytes > INT64_MAX - repeated)
    return fail(e, "template transfer byte sum overflow");
  double repeatedMs = estimatedMs * double(count), sumMs = totalMs + repeatedMs;
  if (!std::isfinite(repeatedMs) || !std::isfinite(sumMs))
    return fail(e, "template theoretical cost multiplication/sum overflow");
  totalBytes += repeated;
  totalMs = sumMs;
  return true;
}
bool orbit::taskflow::planProgramPartition(const GenericAnalysis &a,
                                           int64_t chunk,
                                           ProgramPartitionPlan &p,
                                           std::string &e) {
  p = {};
  p.chunk = chunk;
  p.sourceFingerprint = a.bodyFingerprint;
  if (chunk < 1 || chunk > INT32_MAX || a.tasks.empty())
    return fail(e, "chunk in [1,INT32_MAX] and nonempty program required");
  p.storageAxes.assign(a.storages.size(), -2);
  for (const auto &task : a.tasks) {
    if (task.loops.empty())
      return fail(e, "every task must have a common independent outer axis");
    const auto &loop = task.loops.front();
    auto outer = dyn_cast<affine::AffineForOp>(loop.operation);
    if (!outer || loop.lower != 0 || loop.step != 1 ||
        (p.extent && loop.upper != p.extent))
      return fail(
          e, "outer domains must agree and start at zero with unit stride");
    p.extent = loop.upper;
    if (p.extent <= 0 || (p.extent - 1) / p.chunk + 1 > maxProgramInvocations)
      return fail(e, "host-plan invocation expansion exceeds 1048576; this is "
                     "a host planning limit, not hardware memory capacity");
    for (auto &use : loop.induction.getUses())
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(use.getOwner()) ||
          (isa<affine::AffineStoreOp>(use.getOwner()) &&
           use.getOperandNumber() == 0))
        return fail(e, "partition axis used as a scalar value or unsupported "
                       "index expression");
    bool writes = false;
    for (const auto &access : task.accesses) {
      if (!outer->isAncestor(access.operation))
        return fail(e, "access outside common outer domain");
      AffineMap map;
      ValueRange operands;
      if (auto load = dyn_cast<affine::AffineLoadOp>(access.operation)) {
        map = load.getAffineMap();
        operands = load.getMapOperands();
      } else if (auto store =
                     dyn_cast<affine::AffineStoreOp>(access.operation)) {
        map = store.getAffineMap();
        operands = store.getMapOperands();
      } else
        return fail(e, "partition proof requires affine accesses");
      int64_t axis = -1;
      for (size_t i = 0; i < operands.size(); ++i)
        if (operands[i] == loop.induction) {
          if (i >= map.getNumDims())
            return fail(e, "partition induction cannot be an affine symbol");
          for (size_t d = 0; d < map.getNumResults(); ++d)
            if (map.getResult(d).isFunctionOfDim(i)) {
              auto dim = dyn_cast<AffineDimExpr>(map.getResult(d));
              if (!dim || dim.getPosition() != i || axis != -1)
                return fail(
                    e, "partitioned access must project the axis exactly once");
              axis = d;
            }
        }
      if (access.write) {
        writes = true;
        if (axis < 0)
          return fail(e, "write crosses independent program partitions");
      }
      if (axis >= 0 && a.storages[access.storage].shape[axis] != p.extent)
        return fail(e,
                    "partition storage dimension differs from source domain");
      auto &declared = p.storageAxes[access.storage];
      if (declared != -2 && declared != axis)
        return fail(
            e, "shared storage uses inconsistent partition axes across tasks");
      declared = axis;
    }
    if (!writes)
      return fail(e, "partition task has no independent output");
  }
  for (size_t s = 0; s < a.storages.size(); ++s) {
    if (p.storageAxes[s] == -2)
      return fail(e, "unused storage is not supported in partition ABI");
    if (p.storageAxes[s] < 0 && a.storages[s].role != "input")
      return fail(
          e,
          "only immutable input may be replicated across program partitions");
    if (a.storages[s].role == "output") {
      bool full = false;
      for (const auto &task : a.tasks)
        for (const auto &access : task.accesses)
          if (access.storage == s && access.write && access.exactRegion) {
            bool matches = access.region.size() == a.storages[s].shape.size();
            for (size_t d = 0; matches && d < access.region.size(); ++d)
              matches = access.region[d].lower == 0 &&
                        access.region[d].upper == a.storages[s].shape[d];
            full |= matches;
          }
      if (!full)
        return fail(
            e, "complete output coverage must be proven before partitioning");
    }
  }
  std::string identity = a.bodyFingerprint + ":" + std::to_string(chunk);
  auto module = a.function->getParentOfType<ModuleOp>();
  std::string source;
  llvm::raw_string_ostream os(source);
  module.print(os);
  identity += source;
  for (auto axis : p.storageAxes)
    identity += ":" + std::to_string(axis);
  p.planId = hash(identity);
  return true;
}

bool orbit::taskflow::materializeProgramPartition(ModuleOp source,
                                                  const ProgramPartitionPlan &p,
                                                  int64_t extent,
                                                  OwningOpRef<ModuleOp> &output,
                                                  std::string &e) {
  GenericAnalysis original;
  ProgramPartitionPlan checked;
  if (!analyzeGenericProgram(source, original, e) ||
      !planProgramPartition(original, p.chunk, checked, e))
    return false;
  if (checked.planId != p.planId || checked.storageAxes != p.storageAxes ||
      checked.extent != p.extent ||
      checked.sourceFingerprint != p.sourceFingerprint || extent < 1 ||
      (extent != std::min(p.chunk, p.extent) && extent != p.extent % p.chunk))
    return fail(e, "partition witness or template extent mismatch");
  output = cast<ModuleOp>(source->clone());
  output->getOperation()->removeAttr("orbit.program_partition_extent");
  GenericAnalysis a;
  if (!analyzeGenericProgram(*output, a, e))
    return false;
  auto update = [&](Value value, size_t storage) {
    int64_t axis = p.storageAxes[storage];
    if (axis < 0)
      return;
    auto type = cast<MemRefType>(value.getType());
    SmallVector<int64_t> shape(type.getShape());
    shape[axis] = extent;
    value.setType(MemRefType::get(shape, type.getElementType(),
                                  type.getLayout(), type.getMemorySpace()));
  };
  for (size_t s = 0; s < a.storages.size(); ++s)
    update(a.storages[s].root, s);
  for (const auto &task : a.tasks) {
    auto &body = task.operation->getRegion(0).front();
    for (size_t i = 0; i < task.argumentStorages.size(); ++i)
      update(body.getArgument(i), task.argumentStorages[i]);
    auto *yield = body.getTerminator();
    if (yield->getNumOperands() != task.operation->getNumResults())
      return fail(e, "unsupported task yield/result ABI");
    for (size_t i = 0; i < yield->getNumOperands(); ++i)
      task.operation->getResult(i).setType(yield->getOperand(i).getType());
    cast<affine::AffineForOp>(task.loops.front().operation)
        .setConstantUpperBound(extent);
  }
  SmallVector<Type> types;
  for (auto arg : a.function.getArguments())
    types.push_back(arg.getType());
  a.function.setType(FunctionType::get(source.getContext(), types,
                                       a.function.getResultTypes()));
  if (failed(verify(*output)))
    return fail(e, "materialized template verification failed");
  GenericAnalysis after;
  return analyzeGenericProgram(*output, after, e);
}

j::Value
orbit::taskflow::serializeProgramPartition(const GenericAnalysis &a,
                                           const ProgramPartitionPlan &p) {
  j::Array buffers, invocations, templates;
  for (size_t s = 0; s < a.storages.size(); ++s) {
    const auto &storage = a.storages[s];
    buffers.push_back(j::Object{{"name", storage.id},
                                {"role", storage.role},
                                {"dtype", storage.dtype},
                                {"shape", integers(storage.shape)},
                                {"storage_index", int64_t(s)},
                                {"partition_axis", p.storageAxes[s]}});
  }
  std::set<int64_t> extents;
  const auto order = abiOrder(a);
  for (int64_t lo = 0, index = 0; lo < p.extent; lo += p.chunk, ++index) {
    int64_t count = std::min(p.chunk, p.extent - lo);
    extents.insert(count);
    j::Array bindings, dependencies;
    if (index)
      dependencies.push_back(index - 1);
    for (size_t i = 0; i < order.size(); ++i) {
      size_t s = order[i];
      const auto &storage = a.storages[s];
      auto sizes = storage.shape;
      std::vector<int64_t> offsets(sizes.size(), 0);
      if (p.storageAxes[s] >= 0) {
        sizes[p.storageAxes[s]] = count;
        offsets[p.storageAxes[s]] = lo;
      }
      std::string mode = storage.role == "temporary" ? "temporary"
                         : storage.role == "output"  ? "slice-output"
                         : p.storageAxes[s] < 0      ? "replicated-input"
                                                     : "slice-input";
      bindings.push_back(j::Object{{"buffer_index", int64_t(i)},
                                   {"full_buffer_index", int64_t(s)},
                                   {"mode", mode},
                                   {"offsets", integers(offsets)},
                                   {"sizes", integers(sizes)}});
    }
    invocations.push_back(
        j::Object{{"index", index},
                  {"template_id", "extent" + std::to_string(count)},
                  {"lower", lo},
                  {"upper", lo + count},
                  {"depends_on", std::move(dependencies)},
                  {"bindings", std::move(bindings)}});
  }
  for (auto count : extents) {
    auto prefix = "template-" + std::to_string(count) + "/";
    templates.push_back(j::Object{{"id", "extent" + std::to_string(count)},
                                  {"extent", count},
                                  {"taskflow", prefix + "taskflow.mlir"},
                                  {"air", prefix + "program.air.mlir"},
                                  {"plan", prefix + "plan.json"},
                                  {"abi", prefix + "abi.json"}});
  }
  return j::Object{
      {"schema", "orbit.program.partition.v1"},
      {"plan_id", p.planId},
      {"max_host_invocation_expansion", maxProgramInvocations},
      {"invocation_expansion_limit_scope",
       "host planning/code generation; not physical memory capacity"},
      {"source_fingerprint", p.sourceFingerprint},
      {"partition",
       j::Object{{"source_loop", 0},
                 {"extent", p.extent},
                 {"chunk", p.chunk},
                 {"proof", "all task writes project common independent axis; "
                           "shared storage axes agree; invariant inputs "
                           "replicated; source reductions unchanged"}}},
      {"buffers", std::move(buffers)},
      {"templates", std::move(templates)},
      {"invocations", std::move(invocations)},
      {"schedule",
       "serial complete-program partitions using existing backend commands"},
      {"host_operations", "declared slicing, input staging and output copy "
                          "only; no arithmetic fallback"}};
}

j::Value orbit::taskflow::evaluatePartitionedGenericMLIR(ModuleOp module,
                                                         const j::Value &config,
                                                         std::string &e) {
  GenericAnalysis a;
  ProgramPartitionPlan p;
  auto chunk =
      module->getAttrOfType<IntegerAttr>("orbit.program_partition_extent");
  if (!chunk || !analyzeGenericProgram(module, a, e) ||
      !planProgramPartition(a, chunk.getInt(), p, e))
    return j::Object{{"status", "unsupported"}, {"reason", e}};
  auto manifest = serializeProgramPartition(a, p);
  auto &root = *manifest.getAsObject();
  double total = 0;
  bool calibrated = true;
  bool compositeChildCost = false;
  int64_t transfers = 0, peak = 0, hostPeak = 0;
  j::Array templatePlans, templateMemories;
  std::set<std::string> accountedTemplates;
  std::string mappingIdentity = "orbit.program.mapping.v1:" + p.planId;
  for (auto &t : *root.getArray("templates")) {
    auto extent = *t.getAsObject()->getInteger("extent");
    OwningOpRef<ModuleOp> child;
    GenericAnalysis facts;
    GenericMappingPlan mapping;
    if (!materializeProgramPartition(module, p, extent, child, e) ||
        !analyzeGenericProgram(*child, facts, e) ||
        !planGenericMapping(facts, config, mapping, e))
      return j::Object{{"status", "unsupported"},
                       {"reason", e},
                       {"partition", std::move(manifest)}};
    if (mapping.strategy != "serial-tiled") {
      e = "program partition templates currently require explicit native tile "
          "choices";
      return j::Object{{"status", "unsupported"}, {"reason", e}};
    }
    auto report = serializeGenericMapping(facts, mapping);
    if (llvm::any_of(mapping.taskImplementations, [](const auto &impl) {
          return impl.kind == "independent-lane-f32" ||
                 (impl.kind == "scalar-product-vector-add-f32" &&
                  impl.lanes != 8);
        }))
      return j::Object{
          {"status", "unsupported"},
          {"reason",
           "installed strict FP32 vector implementation unavailable"}};
    int64_t count = 0;
    for (auto &inv : *root.getArray("invocations"))
      count += inv.getAsObject()->getString("template_id") ==
               t.getAsObject()->getString("id");
    auto object = report.getAsObject();
    auto templateId = t.getAsObject()->getString("id")->str();
    auto templateHost = object->getInteger("system_memory_peak_bytes");
    if (!accountedTemplates.insert(templateId).second || !templateHost ||
        *templateHost < 0) {
      e = "program partition: invalid or duplicate template system-memory "
          "accounting";
      return j::Object{{"status", "unsupported"}, {"reason", e}};
    }
    // Each unique template retains one runner and its BOs until all of the
    // seed's invocations finish. Different extents do not share allocations.
    if (hostPeak > INT64_MAX - *templateHost) {
      e = "program partition: retained template BO system-memory sum overflow";
      return j::Object{{"status", "unsupported"}, {"reason", e}};
    }
    hostPeak += *templateHost;
    templateMemories.push_back(j::Object{
        {"template_id", templateId},
        {"system_memory_peak_bytes", *templateHost},
        {"invocation_count", count},
        {"lifetime", "one retained template runner and its BOs through the "
                     "complete seed/program execution; repeated invocations "
                     "reuse these allocations"}});
    auto cost = object->getNumber("estimated_ms");
    calibrated &= bool(cost);
    compositeChildCost |= mapping.directCost.enabled;
    auto childTransfers = object->getInteger("actual_dma_bytes");
    if (!childTransfers ||
        !accumulateProgramPartitionMetrics(
            *childTransfers, count, cost.value_or(0), transfers, total, e))
      return j::Object{{"status", "unsupported"}, {"reason", e}};
    mappingIdentity += ":" + std::to_string(extent) + ":" +
                       std::to_string(count) + ":" + mapping.planId;
    peak = std::max(peak, *object->getInteger("resident_bytes"));
    templatePlans.push_back(
        j::Object{{"template_id", t.getAsObject()->getString("id")->str()},
                  {"count", count},
                  {"plan", std::move(report)}});
  }
  root["template_plans"] = std::move(templatePlans);
  root["partition_witness_id"] = p.planId;
  root["plan_id"] = hash(mappingIdentity);
  root["identity_scope"] =
      "partition witness plus each unique template extent, invocation count "
      "and checked child mapping plan ID";
  root["estimated_ms"] =
      (calibrated && !compositeChildCost) ? j::Value(total) : j::Value(nullptr);
  if (compositeChildCost) {
    root["template_execution_estimated_ms"] =
        calibrated ? j::Value(total) : j::Value(nullptr);
    root["cost_status"] = "uncalibrated-complete-program-host-boundary";
    root["complete_program_missing_costs"] =
        "partition slicing, assembly and invocation control are not calibrated "
        "by the prepared single-command provider";
  }
  root["actual_dma_bytes"] = transfers;
  root["resident_bytes"] = peak;
  root["system_memory_peak_bytes"] = hostPeak;
  root["template_system_memories"] = std::move(templateMemories);
  root["system_memory_accounting_scope"] =
      "sum of unique template runner/BO tensor payloads retained concurrently "
      "for one seed/program execution; excludes full host input/assembly "
      "arrays, references, XRT context memory and compiler/runtime overhead";
  root["host_buffer_lifetime_policy"] =
      "each unique template's BOs persist until all complete-program "
      "invocations for the seed finish; reuse across invocation count, no "
      "sharing across template extents";
  root["cost_provenance"] =
      "sum of complete partition plans; serial; host slice/copy/context costs "
      "uncalibrated and excluded";
  return j::Object{{"status", "supported"}, {"best_plan", std::move(manifest)}};
}

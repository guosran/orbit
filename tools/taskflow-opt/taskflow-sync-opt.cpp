#include "Taskflow/HeterogeneousSearch.h"
#include "Taskflow/SyncBoundaryMapping.h"
#include "Taskflow/SyncCost.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace orbit::taskflow;
namespace cl = llvm::cl;
static cl::opt<std::string> input(cl::Positional, cl::Required);
static cl::opt<std::string> configPath("target-config", cl::Required);
static cl::opt<std::string> bindingPath("runtime-bindings", cl::Required);
static cl::opt<std::string> modelPath("sync-model", cl::Required);
static cl::opt<std::string> outputPath("output", cl::Required);

// A CPU-only projection of a checked successful serial plan. Runtime extents,
// aliases and observation identities must be supplied as evidence; this tool
// neither runs devices nor turns an unknown physical cost into zero.
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "ORBIT checked graph to sync API queries\n");
  auto fail = [](const std::string &why) {
    llvm::errs() << why << "\n";
    return 1;
  };
  std::string error;
  auto read = [&](const std::string &path, llvm::json::Value &value) {
    auto data = llvm::MemoryBuffer::getFile(path);
    if (!data) {
      error = data.getError().message();
      return false;
    }
    auto parsed = llvm::json::parse((*data)->getBuffer());
    if (!parsed) {
      error = llvm::toString(parsed.takeError());
      return false;
    }
    value = std::move(*parsed);
    return true;
  };
  llvm::json::Value config = nullptr, runtimeJSON = nullptr;
  if (!read(configPath, config) || !read(bindingPath, runtimeJSON))
    return fail(error);
  auto *runtimeObject = runtimeJSON.getAsObject();
  if (!runtimeObject)
    return fail("runtime bindings must be an object");
  auto runtimeId = runtimeObject->getString("runtime_identity");
  auto observerId = runtimeObject->getString("observer_identity");
  auto conditionId = runtimeObject->getString("condition_identity");
  auto *records = runtimeObject->getArray("bindings");
  if (!runtimeId || !observerId || !conditionId || !records ||
      runtimeId->empty() || observerId->empty() || conditionId->empty())
    return fail(
        "runtime/observer/condition identities and actual bindings required");
  std::vector<SyncRuntimeBOBinding> bindings;
  for (const auto &record : *records) {
    const auto *o = record.getAsObject();
    if (!o)
      return fail("binding must be an object");
    auto storage = o->getInteger("storage"),
         generation = o->getInteger("generation"),
         extent = o->getInteger("observed_extent_bytes"),
         offset = o->getInteger("storage_offset");
    auto id = o->getString("physical_id"),
         evidence = o->getString("extent_evidence");
    auto *copies = o->getArray("copies");
    if (!storage || *storage < 0 || !generation || *generation < 0 || !extent ||
        !offset || !id || !evidence || !copies)
      return fail("incomplete runtime BO binding");
    SyncRuntimeBOBinding b;
    b.storage = *storage;
    b.physicalId = id->str();
    b.generation = *generation;
    b.observedExtentBytes = *extent;
    b.storageOffset = *offset;
    b.extentEvidence = evidence->str();
    for (const auto &copy : *copies) {
      auto n = copy.getAsInteger();
      if (!n || *n < 0)
        return fail("invalid alias");
      b.copies.push_back(*n);
    }
    bindings.push_back(std::move(b));
  }
  auto modelBytes = llvm::MemoryBuffer::getFile(modelPath);
  if (!modelBytes)
    return fail(modelBytes.getError().message());
  SyncAPICostModel model;
  if (!parseSyncAPICostModel((*modelBytes)->getBuffer(), model, error))
    return fail(error);
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module)
    return fail("invalid Taskflow");
  GenericAnalysis source;
  HeterogeneousMappingChoice choice;
  HeterogeneousMappingPlan plan;
  if (!analyzeGenericProgram(*module, source, error) ||
      !readHeterogeneousChoice(*module, config, choice, error) ||
      !buildHeterogeneousMapping(source, choice, plan, error) ||
      !checkHeterogeneousMapping(source, choice, plan, error))
    return fail(error);
  MechanismRuntimeContract runtime;
  runtime.identity = runtimeId->str();
  MechanismFacts facts;
  if (!collectMechanismFacts(plan.graph, runtime, facts, error))
    return fail(error);
  SyncQueryContext queryContext;
  queryContext.callForm = SyncCallForm::ImplicitFullBuffer;
  queryContext.observerBoundary = SyncObserverBoundary::DetailedCompatibility;
  queryContext.runtimeIdentity = runtime.identity;
  queryContext.observerIdentity = observerId->str();
  SyncBoundaryTrace trace;
  if (!collectSyncBoundaryQueries(plan.graph, facts, bindings, queryContext,
                                  trace, error))
    return fail(error);
  // Derive ABI order from the very same MLIR region projection consumed by
  // heterogeneous lowering, rather than sorting logical names or report IDs.
  std::vector<HeterogeneousRegionProjection> projections(plan.slices.size());
  for (size_t i = 0; i < plan.slices.size(); ++i)
    if (plan.slices[i].device == "npu") {
      mlir::OwningOpRef<mlir::ModuleOp> region;
      if (!materializeHeterogeneousRegion(plan, i, region, projections[i],
                                          error))
        return fail(error);
    }
  llvm::json::Array evaluations;
  double known = 0;
  size_t unknown = 0;
  for (const auto &q : trace.queries) {
    SyncCostRequest request;
    request.query = q.result;
    request.direction = q.direction;
    request.conditionIdentity = conditionId->str();
    request.site = q.direction == SyncDirection::ToDevice
                       ? "installed-bound-bo-input-loop"
                       : "installed-bound-bo-readback-loop";
    for (const auto &a : projections[size_t(q.region)].arguments)
      if (a.parentStorage == size_t(q.storage))
        request.argumentOrdinal = a.childArgument;
    for (const auto &b : bindings)
      if (b.storage == size_t(q.storage))
        request.observedExtentBytes = b.observedExtentBytes;
    SyncAPICostEvaluation result;
    if (!evaluateSyncAPICost(model, request, result, error))
      return fail(error);
    if (result.durationNs)
      known += *result.durationNs;
    else
      ++unknown;
    evaluations.push_back(llvm::json::Object{
        {"mechanism_item", int64_t(q.mechanismItem)},
        {"region", q.region},
        {"storage", q.storage},
        {"request", serializeSyncCostRequest(request)},
        {"evaluation", serializeSyncAPICostEvaluation(result)}});
  }
  if (llvm::sys::fs::exists(outputPath))
    return fail("output directory must be new");
  if (auto ec = llvm::sys::fs::create_directories(outputPath))
    return fail(ec.message());
  auto write = [&](const std::string &name, const llvm::json::Value &value) {
    std::error_code ec;
    llvm::raw_fd_ostream stream(outputPath + "/" + name, ec);
    if (ec) {
      error = ec.message();
      return false;
    }
    stream << llvm::formatv("{0:2}\n", value);
    return true;
  };
  llvm::json::Value result = llvm::json::Object{
      {"schema", "orbit.sync-plan-cost.v14"},
      {"graph_id", programGraphIdentity(plan.graph)},
      {"facts_id", facts.identity},
      {"parameter_file_sha256", model.parameterFileSHA256},
      {"queries", std::move(evaluations)},
      {"known_sync_api_subtotal_ns", known},
      {"unknown_sync_queries", int64_t(unknown)},
      {"whole_program_ns", nullptr},
      {"physical_transfer_bytes", nullptr},
      {"origin", trace.origin},
      {"reason",
       "successful static plan projection; compute, internal DMA, submit/wait "
       "and other boundary APIs remain separate or unknown"}};
  if (!write("program-graph.json", serializeProgramGraph(plan.graph)) ||
      !write("mechanism-facts.json", serializeMechanismFacts(facts)) ||
      !write("sync-boundary-trace.json", serializeSyncBoundaryTrace(trace)) ||
      !write("sync-cost.json", result))
    return fail(error);
  return 0;
}

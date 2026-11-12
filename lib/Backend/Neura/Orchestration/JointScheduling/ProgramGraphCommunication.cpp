#include "Backend/Neura/Orchestration/JointScheduling/ProgramGraphCommunication.h"
#include "Taskflow/ProgramMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <set>
using namespace mlir;
namespace mlir::amoeba::neura::joint_scheduling {
namespace {
thread_local const ::orbit::taskflow::ProgramGraph *active = nullptr;
}
ScopedProgramGraphCommunication::ScopedProgramGraphCommunication(
    const ::orbit::taskflow::ProgramGraph *graph)
    : previous(active) {
  active = graph;
}
ScopedProgramGraphCommunication::~ScopedProgramGraphCommunication() {
  active = previous;
}
const ::orbit::taskflow::ProgramGraph *activeProgramGraphCommunication() {
  return active;
}
FailureOr<TaskEdgeGraph>
buildTaskEdgeGraphFromProgramGraph(func::FuncOp function,
                                   const ::orbit::taskflow::ProgramGraph &graph,
                                   std::string &error) {
  auto fail = [&](StringRef why) -> FailureOr<TaskEdgeGraph> {
    error = why.str();
    return failure();
  };
  if (!::orbit::taskflow::checkProgramGraph(graph, error))
    return failure();
  auto identity = function->getAttrOfType<StringAttr>("orbit.program_graph");
  if (!identity ||
      identity.getValue() != ::orbit::taskflow::programGraphIdentity(graph))
    return fail("Amoeba function does not bind the active common graph");
  TaskEdgeGraph result;
  result.tasks_.resize(graph.source.tasks.size());
  bool valid = true;
  function.walk([&](::mlir::taskflow::TaskflowTaskOp task) {
    auto id = task->getAttrOfType<IntegerAttr>("orbit.program_graph_task");
    if (!id || id.getInt() < 0 || size_t(id.getInt()) >= result.tasks_.size()) {
      valid = false;
      return;
    }
    size_t index = id.getInt();
    if (result.tasks_[index] ||
        task.getTaskName() != graph.source.tasks[index].id) {
      valid = false;
      return;
    }
    result.tasks_[index] = task;
  });
  if (!valid || llvm::any_of(result.tasks_, [](auto task) { return !task; }))
    return fail(
        "Amoeba source-to-lowered task correspondence is not bijective");
  std::vector<size_t> compute(graph.source.tasks.size(), SIZE_MAX);
  for (const auto &instance : graph.instances) {
    if (instance.kind != "source-task" || instance.task >= compute.size() ||
        compute[instance.task] != SIZE_MAX)
      return fail(
          "Amoeba common adapter requires one whole source instance per task");
    compute[instance.task] = instance.event;
  }
  if (llvm::any_of(compute, [](size_t event) { return event == SIZE_MAX; }))
    return fail("Amoeba common graph omits a source task instance");
  auto producerOf = [&](size_t copy) -> int64_t {
    const auto &version = graph.versions[graph.copies[copy].version];
    return version.producer < 0 ? -1
                                : int64_t(graph.events[version.producer].task);
  };
  std::set<std::pair<size_t, size_t>> dependency;
  for (size_t index = 0; index < graph.events.size(); ++index) {
    const auto &event = graph.events[index];
    if (event.kind != "transfer")
      continue;
    if (event.reads.empty() || event.writes.empty() || !event.transferRegion ||
        event.transferBytes <= 0)
      return fail("Amoeba transfer lacks exact checked copy endpoints");
    bool externalSource = producerOf(event.reads.front()) < 0;
    bool externalTarget = llvm::all_of(event.writes, [&](size_t copy) {
      return graph.allocations[graph.copies[copy].allocation].memory ==
             "external.memory";
    });
    if (externalSource || externalTarget)
      continue; // Explicitly unpriced external supply/export scope.
    size_t consumer = event.task;
    int64_t producer = producerOf(event.reads.front());
    if (producer < 0 || size_t(producer) >= compute.size() ||
        consumer >= compute.size() || producer == int64_t(consumer))
      return fail(
          "Amoeba internal copy lacks distinct source/consumer task endpoints");
    for (size_t copy : event.reads)
      if (producerOf(copy) != producer)
        return fail(
            "Amoeba internal transfer combines different producer tasks");
    for (size_t copy : event.writes) {
      const auto &allocation = graph.allocations[graph.copies[copy].allocation];
      if (allocation.memory !=
          "task." + std::to_string(consumer) + ".availability")
        return fail("Amoeba retained/shared destination copy has no supported "
                    "network endpoint");
      for (size_t user : graph.copies[copy].users)
        if (graph.events[user].kind == "compute" &&
            graph.events[user].task != consumer)
          return fail("Amoeba common route cannot multicast one created copy "
                      "across task endpoints");
    }
    if (uint64_t(event.transferBytes) > UINT64_MAX / 8)
      return fail("Amoeba transfer byte-to-bit conversion overflows");
    TaskEdge edge;
    edge.producer = result.tasks_[producer];
    edge.consumer = result.tasks_[consumer];
    edge.kind = TaskEdgeKind::Raw;
    edge.origin = TaskEdgeOrigin::Taskflow;
    edge.payload_bits = uint64_t(event.transferBytes) * 8;
    edge.programGraphEvent = index;
    edge.scope = TaskEdgeScope::TileLocal;
    edge.transfer_region_lower.append(event.transferRegion->offsets.begin(),
                                      event.transferRegion->offsets.end());
    for (size_t d = 0; d < event.transferRegion->sizes.size(); ++d)
      edge.transfer_region_upper.push_back(event.transferRegion->offsets[d] +
                                           event.transferRegion->sizes[d]);
    result.edges_.push_back(std::move(edge));
    dependency.emplace(producer, consumer);
  }
  // Data requirements without a creation transfer are readiness dependencies,
  // never duplicated traffic. Current endpoint-local subset rejects remote use.
  for (const auto &edge : graph.dataEdges) {
    if (edge.producer < 0)
      continue;
    size_t from = graph.events[edge.producer].task,
           to = graph.events[edge.consumer].task;
    if (from == to || dependency.count({from, to}))
      continue;
    const auto &copy = graph.copies[edge.copy];
    const auto &allocation = graph.allocations[copy.allocation];
    if (allocation.memory != "task." + std::to_string(to) + ".availability")
      return fail("Amoeba data copy is not available at its consumer endpoint");
    TaskEdge order;
    order.producer = result.tasks_[from];
    order.consumer = result.tasks_[to];
    order.kind = TaskEdgeKind::Control;
    order.origin = TaskEdgeOrigin::Control;
    order.payload_bits = 0;
    result.edges_.push_back(order);
    dependency.emplace(from, to);
  }
  for (size_t index = 0; index < graph.orderEdges.size(); ++index) {
    const auto &edge = graph.orderEdges[index];
    size_t from = graph.events[edge.producer].task,
           to = graph.events[edge.consumer].task;
    if (from == to)
      continue;
    TaskEdge order;
    order.producer = result.tasks_[from];
    order.consumer = result.tasks_[to];
    order.kind = TaskEdgeKind::Control;
    order.origin = TaskEdgeOrigin::Control;
    order.payload_bits = 0;
    order.programGraphOrder = index;
    result.edges_.push_back(order);
  }
  return result;
}
} // namespace mlir::amoeba::neura::joint_scheduling

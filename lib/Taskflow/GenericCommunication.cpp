#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <tuple>
#include <set>
using namespace orbit::taskflow;
llvm::StringRef orbit::taskflow::programDataPolicyName(ProgramDataPolicy policy) {
  return policy == ProgramDataPolicy::PhaseLocal ? "phase-local" : "retain-across-users";
}
bool orbit::taskflow::parseProgramDataPolicy(llvm::StringRef value,
    ProgramDataPolicy &policy, std::string &error) {
  if (value == "phase-local") policy = ProgramDataPolicy::PhaseLocal;
  else if (value == "retain-across-users") policy = ProgramDataPolicy::RetainAcrossUsers;
  else { error = "unsupported public data organization policy: " + value.str(); return false; }
  return true;
}
namespace {
std::string nativeDevice(const GenericMappingPlan &p){auto *resource=programResource(p.memoryBinding.architecture,p.memoryBinding.computeResource);return resource?resource->device:std::string{};}

bool fail(std::string &error, const std::string &message) {
  error = "program communication: " + message;
  return false;
}
MappingRegion region(const GenericTileRegion &r) {
  return {r.storage, r.offsets, r.sizes};
}
MappingRegion region(const GenericTransfer &r, const GenericAnalysis &a) {
  if (!r.sizes.empty()) return {r.storage, r.offsets, r.sizes};
  return {r.storage, std::vector<int64_t>(a.storages[r.storage].shape.size(), 0),
          a.storages[r.storage].shape};
}
bool same(const MappingRegion &a, const MappingRegion &b) {
  return a.storage == b.storage && a.offsets == b.offsets && a.sizes == b.sizes;
}
bool inside(const MappingRegion &a, const MappingRegion &b) {
  if (a.storage != b.storage || a.sizes.size() != b.sizes.size()) return false;
  for (size_t d = 0; d < a.sizes.size(); ++d)
    if (a.offsets[d] < b.offsets[d] ||
        a.offsets[d]+a.sizes[d] > b.offsets[d]+b.sizes[d]) return false;
  return true;
}
int64_t bytes(const GenericAnalysis &a, const MappingRegion &r) {
  auto type = mlir::cast<mlir::MemRefType>(a.storages[r.storage].root.getType());
  int64_t n = (type.getElementType().getIntOrFloatBitWidth()+7)/8;
  for (auto size : r.sizes) {
    if (size <= 0 || n > INT64_MAX/size) return -1;
    n *= size;
  }
  return n;
}
// This is StaticCommunication's boundary-cell construction generalized to
// arbitrary rank. Cells are data identities, not additional compute tasks.
struct Partition {
  std::vector<std::vector<std::vector<int64_t>>> cuts;
  std::vector<size_t> bases;
  bool build(const GenericAnalysis &a, const std::vector<GenericTilePhase> &phases,
             const GenericMappingPlan &p, std::vector<MappingRegion> &out,
             std::string &error) {
    cuts.resize(a.storages.size());
    for (size_t s = 0; s < a.storages.size(); ++s)
      for (auto extent : a.storages[s].shape) cuts[s].push_back({0, extent});
    auto add = [&](const MappingRegion &r) {
      for (size_t d = 0; d < r.sizes.size(); ++d) {
        cuts[r.storage][d].push_back(r.offsets[d]);
        cuts[r.storage][d].push_back(r.offsets[d]+r.sizes[d]);
      }
    };
    for (const auto &phase : phases) for (const auto &r : phase.regions) add(region(r));
    for (const auto &transfer : p.transfers) add(region(transfer, a));
    for (size_t s = 0; s < cuts.size(); ++s) {
      size_t count = 1;
      for (auto &axis : cuts[s]) {
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        if (count > 1048576/(axis.size()-1))
          return fail(error, "exact data-region partition exceeds 1048576 cells");
        count *= axis.size()-1;
      }
      if (out.size()+count > 1048576)
        return fail(error, "exact data-region partition exceeds 1048576 cells");
      bases.push_back(out.size());
      MappingRegion r;
      r.storage = s; r.offsets.resize(cuts[s].size()); r.sizes.resize(cuts[s].size());
      std::function<void(size_t)> emit = [&](size_t d) {
        if (d == cuts[s].size()) { out.push_back(r); return; }
        for (size_t i = 0; i+1 < cuts[s][d].size(); ++i) {
          r.offsets[d] = cuts[s][d][i];
          r.sizes[d] = cuts[s][d][i+1]-r.offsets[d]; emit(d+1);
        }
      };
      emit(0);
    }
    return true;
  }
  std::vector<size_t> cells(const MappingRegion &r) const {
    std::vector<size_t> result;
    std::function<void(size_t,size_t)> emit = [&](size_t d, size_t index) {
      if (d == r.sizes.size()) { result.push_back(bases[r.storage]+index); return; }
      const auto &axis = cuts[r.storage][d];
      size_t begin = std::lower_bound(axis.begin(),axis.end(),r.offsets[d])-axis.begin();
      size_t end = std::lower_bound(axis.begin(),axis.end(),r.offsets[d]+r.sizes[d])-axis.begin();
      for (size_t i = begin; i < end; ++i) emit(d+1,index*(axis.size()-1)+i);
    };
    emit(0,0); return result;
  }
};
bool phasesFor(const GenericAnalysis &a, const GenericMappingPlan &p,
               std::vector<GenericTilePhase> &resident,
               const std::vector<GenericTilePhase> *&phases, std::string &error) {
  if (p.strategy == "serial-tiled") { phases = &p.phases; return true; }
  for (size_t task : p.orderedTasks) {
    GenericTilePhase phase; phase.task = task;
    for (size_t l = 0; l < a.tasks[task].loops.size(); ++l) {
      const auto &loop = a.tasks[task].loops[l];
      phase.loops.push_back({l,loop.lower,loop.upper});
    }
    int64_t ignored;
    if (!deriveGenericPhaseRegions(a,phase,ignored,error)) return false;
    resident.push_back(std::move(phase));
  }
  phases = &resident; return true;
}
std::string poolKey(const GenericMappingPlan &p, const GenericTilePhase &phase,
                    size_t index, const GenericTileRegion &r) {
  std::string key = "storage-"+std::to_string(r.storage)+".data";
  for (const auto &scope : p.readOnlyRetentions)
    if (scope.storage == r.storage && scope.beginPhase <= index && index < scope.endPhase)
      return key+".readonly-panel";
  if (r.read && !r.write && !genericLocallyRetained(p,index,r.storage)) {
    key += ".input";
    for (auto n : r.sizes) key += "."+std::to_string(n);
    key += ".loops";
    for (auto bound : phase.loops) key += "."+std::to_string(bound.upper-bound.lower);
  }
  return key;
}
// Native tile groups acquire their input/output slots at block entry and
// release them after the last compute. Source phase indices bind instances;
// their ordinal order is not a DMA-completion barrier. The event vector below
// remains topological while deferred exports retain their producer phase.
std::vector<size_t> tileGroupPredecessors(const GenericMappingPlan &p,
    llvm::StringRef kind,size_t phase,llvm::StringRef link) {
  std::vector<size_t> result;
  const auto *scope=genericResidencyScope(p,phase);if(!scope)return result;
  const size_t width=scope->tasks.size(),begin=scope->beginPhase+(phase-scope->beginPhase)/width*width;
  auto compute=[&](size_t sourcePhase){auto e=p.phases[sourcePhase].computationEvent;if(e>=0)result.push_back(size_t(e));};
  if(kind!="transfer") {
    if(phase>begin)compute(phase-1);
    if(phase==begin){
      if(p.targetCapabilities.acquireAllInputsAtGroupEntry){
        for(const auto &transfer:p.transfers)
          if(transfer.phase>=begin&&transfer.phase<begin+width&&transfer.direction=="host_to_l1"&&transfer.event>=0)result.push_back(size_t(transfer.event));
      }
      if(begin>scope->beginPhase){
        for(const auto &transfer:p.transfers)
          if(transfer.phase>=begin-width&&transfer.phase<begin&&transfer.direction=="l1_to_host"&&transfer.event>=0)result.push_back(size_t(transfer.event));
      }
    }
  } else if(link==p.memoryBinding.outputLink)compute(begin+width-1);
  else if(begin>scope->beginPhase&&scope->inputReleasePolicy=="group-tail")compute(begin-1);
  // last-reader inputs use the per-allocation creator/all-users overwrite
  // constraints. Their earlier release must not inherit an unrelated core tail.
  return result;
}
// A configured native group has one ordered core, but independently serviced
// input and output slots. Only group transitions impose a complete control
// frontier. Within a group readiness and alias reuse derive DMA ordering.
std::vector<size_t> serialGroupPredecessors(const GenericMappingPlan &p,
    const std::vector<MappingEvent> &events,size_t before,llvm::StringRef kind,
    size_t phase,llvm::StringRef link,std::map<size_t,std::vector<size_t>> &frontiers) {
  std::vector<size_t> result=tileGroupPredecessors(p,kind,phase,link);
  auto group=std::find_if(p.executionGroups.begin(),p.executionGroups.end(),
      [&](const auto &g){return g.beginPhase<=phase&&phase<g.endPhase;});
  if(group==p.executionGroups.end())return result; // placement checker diagnoses coverage
  size_t groupIndex=group-p.executionGroups.begin();
  auto cached=frontiers.find(groupIndex);
  if(cached==frontiers.end()) {
    std::vector<size_t> control;
    auto parents=group->dependencies;parents.insert(parents.end(),group->implementationDependencies.begin(),group->implementationDependencies.end());
    for(auto parent:parents)if(parent<p.executionGroups.size()) {
      const auto &boundary=p.executionGroups[parent];std::set<size_t> frontier;
      for(size_t e=0;e<before;++e)if(boundary.beginPhase<=events[e].phase&&events[e].phase<boundary.endPhase)frontier.insert(e);
      for(size_t e=0;e<before;++e)if(boundary.beginPhase<=events[e].phase&&events[e].phase<boundary.endPhase)
        for(auto predecessor:events[e].dependencies)frontier.erase(predecessor);
      control.insert(control.end(),frontier.begin(),frontier.end());
    }
    cached=frontiers.emplace(groupIndex,std::move(control)).first;
  }
  result.insert(result.end(),cached->second.begin(),cached->second.end());
  if(kind!="transfer")for(size_t e=before;e>0;--e) {
    const auto &previous=events[e-1];
    if(previous.kind!="transfer"&&group->beginPhase<=previous.phase&&previous.phase<group->endPhase){result.push_back(e-1);break;}
  }
  return result;
}
// A serial native adapter has the same explicit latest-version, per-location
// copy state and overwrite dependencies as StaticCommunication. It does not
// assume that a BO exists implies that its contents have been produced.
struct State {
  GenericMappingPlan &p;
  const GenericAnalysis &a;
  const std::vector<GenericTilePhase> &phases;
  Partition partition;
  std::vector<std::map<size_t,size_t>> available; // allocation -> cell -> copy
  std::vector<int64_t> latest;
  std::vector<size_t> host;
  std::vector<size_t> phaseBarrier;
  std::map<size_t,std::vector<size_t>> groupFrontiers;
  size_t barrierPhase = SIZE_MAX, phaseBegin = 0;
  State(GenericMappingPlan &p, const GenericAnalysis &a,
        const std::vector<GenericTilePhase> &phases) : p(p),a(a),phases(phases) {}
  int64_t allocation(size_t storage, size_t phase, bool onHost,
                     const GenericTileRegion *demand = nullptr) const {
    const auto &c = p.communication;
    if (onHost) return host[storage] == SIZE_MAX ? -1 : int64_t(host[storage]);
    std::string key = p.strategy == "serial-tiled" && demand
        ? poolKey(p, phases[phase], phase, *demand) : "storage-"+std::to_string(storage)+".data";
    for (size_t i = 0; i < c.allocations.size(); ++i) {
      const auto &al = c.allocations[i];
      if (al.storage == storage && al.memory == p.memoryBinding.localMemory && al.key == key &&
          al.beginPhase <= phase && phase < al.endPhase) return i;
    }
    return -1;
  }
  bool setup(std::string &error) {
    auto &c = p.communication;
    if (!partition.build(a,phases,p,c.regions,error)) return false;
    available.resize(c.allocations.size()); latest.assign(c.regions.size(),-1);
    host.assign(a.storages.size(),SIZE_MAX);
    for (size_t i = 0; i < c.allocations.size(); ++i)
      if (c.allocations[i].memory == p.memoryBinding.hostMemory) host[c.allocations[i].storage] = i;
    for (size_t cell = 0; cell < c.regions.size(); ++cell) {
      size_t storage = c.regions[cell].storage;
      if (a.storages[storage].role != "input") continue;
      if (host[storage] == SIZE_MAX) return fail(error,"missing initial input location");
      size_t version = c.versions.size();
      c.versions.push_back({cell,-1}); latest[cell] = version;
      size_t copy = c.copies.size();
      c.copies.push_back({version,host[storage],-1,0,SIZE_MAX,{}});
      available[host[storage]][cell] = copy;
    }
    return true;
  }
  size_t start(std::string kind, size_t phase, size_t task, std::string link = {}) {
    auto &events = p.communication.events;
    if (p.strategy!="serial-tiled" && phase != barrierPhase) {
      std::set<size_t> frontier;
      for (size_t index = phaseBegin; index < events.size(); ++index) frontier.insert(index);
      for (size_t index = phaseBegin; index < events.size(); ++index)
        for (size_t dependency : events[index].dependencies) frontier.erase(dependency);
      phaseBarrier.assign(frontier.begin(),frontier.end());
      barrierPhase = phase;
      phaseBegin = events.size();
    }
    MappingEvent event; event.kind = std::move(kind); event.phase = phase;
    event.task = task; event.memory = event.kind == "transfer" ? "" : p.memoryBinding.localMemory; event.link = std::move(link);
    event.resource = event.kind == "transfer" ? p.memoryBinding.dmaResource : p.memoryBinding.computeResource;
    event.dependencies = p.strategy=="serial-tiled" ? serialGroupPredecessors(p,events,events.size(),event.kind,phase,event.link,groupFrontiers) : phaseBarrier;
    events.push_back(std::move(event)); return events.size()-1;
  }
  void depend(size_t event, int64_t producer) {
    if (producer >= 0 && producer != int64_t(event))
      p.communication.events[event].dependencies.push_back(producer);
  }
  void beforeOverwrite(size_t event, size_t copy) {
    const auto &value = p.communication.copies[copy];
    depend(event,value.createdBy);
    const auto &users=p.strategy=="serial-tiled"?value.users:value.lastUsers;
    for (size_t user : users) depend(event,user);
  }
  bool read(size_t event, size_t al, size_t cell, std::string &error) {
    auto found = available[al].find(cell);
    if (found == available[al].end()) return fail(error,"read before producing a location copy");
    auto &c = p.communication;
    auto &copy = c.copies[found->second];
    if (int64_t(copy.version) != latest[cell]) return fail(error,"attempt to read stale data version");
    c.events[event].reads.push_back(found->second);
    depend(event,copy.createdBy);
    copy.lastUsers = {event}; copy.users.push_back(event); return true;
  }
  void clearLocal(size_t event, size_t al) {
    if (p.communication.allocations[al].memory == p.memoryBinding.hostMemory) return;
    for (auto item : available[al]) {
      beforeOverwrite(event,item.second);
      p.communication.copies[item.second].releasedAfter = event;
    }
    available[al].clear();
  }
  void write(size_t event, size_t al, size_t cell, size_t version) {
    auto &c = p.communication;
    auto old = available[al].find(cell);
    if (old != available[al].end()) {
      beforeOverwrite(event,old->second);
      c.copies[old->second].releasedAfter = event;
    }
    size_t copy = c.copies.size();
    c.copies.push_back({version,al,int64_t(event),event,SIZE_MAX,{}});
    c.events[event].writes.push_back(copy); available[al][cell] = copy;
  }
  bool transfer(size_t index, size_t phase, std::string &error) {
    auto &t = p.transfers[index];
    bool input = t.direction == "host_to_l1";
    const auto &sourcePhase = phases[phase];
    auto demand = std::find_if(sourcePhase.regions.begin(),sourcePhase.regions.end(),
                               [&](const auto &r) { return r.storage == t.storage; });
    if (demand == sourcePhase.regions.end() && p.strategy == "serial-tiled")
      return fail(error,"DMA has no source instance access");
    int64_t local = allocation(t.storage,phase,false,
        demand == sourcePhase.regions.end() ? nullptr : &*demand);
    int64_t external = allocation(t.storage,phase,true);
    if (local < 0 || external < 0) return fail(error,"DMA allocation binding missing");
    size_t source = input ? external : local, target = input ? local : external;
    size_t event = start("transfer",phase,sourcePhase.task,input ? p.memoryBinding.inputLink : p.memoryBinding.outputLink);
    p.communication.events[event].transferRegion=region(t,a);
    p.communication.events[event].transferBytes=t.bytes;
    auto cells = partition.cells(region(t,a));
    for (size_t cell : cells) if (!read(event,source,cell,error)) return false;
    clearLocal(event,target);
    for (size_t cell : cells) write(event,target,cell,latest[cell]);
    t.event = event; t.sourceCopies = p.communication.events[event].reads;
    t.targetCopies = p.communication.events[event].writes;
    return true;
  }
  bool compute(size_t phase, int64_t initializer, size_t storage, std::string &error) {
    const auto &instance = phases[phase];
    size_t task = initializer >= 0 ? size_t(initializer) : instance.task;
    size_t event = start(initializer >= 0 ? "initialize" : "compute",phase,task);
    if (initializer < 0 && p.strategy == "serial-tiled") p.phases[phase].computationEvent = event;
    for (const auto &r : instance.regions) {
      if (initializer >= 0 && r.storage != storage) continue;
      int64_t al = allocation(r.storage,phase,false,&r);
      if (al < 0) return fail(error,"compute allocation binding missing");
      auto cells = partition.cells(region(r));
      if (r.read && initializer < 0)
        for (size_t cell : cells) if (!read(event,al,cell,error)) return false;
    }
    for (const auto &r : instance.regions) {
      if (!r.write || (initializer >= 0 && r.storage != storage)) continue;
      size_t al = allocation(r.storage,phase,false,&r);
      clearLocal(event,al);
      for (size_t cell : partition.cells(region(r))) {
        size_t version = p.communication.versions.size();
        p.communication.versions.push_back({cell,int64_t(event)});
        latest[cell] = version; write(event,al,cell,version);
      }
    }
    return true;
  }
};
// Edges describe source value consumption; transport is a separate copy-creation event.
void populateEdges(const GenericAnalysis &a, ProgramCommunicationPlan &c) {
  c.dataEdges.clear(); c.orderEdges.clear();
  for (auto &copy : c.copies) copy.consumerEdges.clear();
  std::set<std::pair<size_t,size_t>> rawTasks;
  std::map<size_t,size_t> first,last;
  for (size_t i=0;i<c.events.size();++i) {
    auto &e=c.events[i]; e.dataEdges.clear(); e.orderEdges.clear();
    if (e.kind=="transfer") continue;
    first.emplace(e.task,i); last[e.task]=i;
    for (size_t copyId:e.reads) {
      auto &copy=c.copies[copyId]; auto &v=c.versions[copy.version];
      const auto &r=c.regions[v.region];
      size_t edge=c.dataEdges.size();
      c.dataEdges.push_back({v.producer,i,v.region,copy.version,copyId,
        {c.allocations[copy.allocation].memory,copy.layout,a.storages[r.storage].dtype}});
      e.dataEdges.push_back(edge); copy.consumerEdges.push_back(edge);
      if(v.producer>=0) rawTasks.insert({c.events[v.producer].task,e.task});
    }
  }
  for(size_t task=0;task<a.tasks.size();++task) for(size_t dep:a.tasks[task].dependencies) {
    if(rawTasks.count({dep,task}) || !last.count(dep) || !first.count(task)) continue;
    c.events[first[task]].orderEdges.push_back(c.orderEdges.size());
    c.orderEdges.push_back({last[dep],first[task],"source task dependency without value consumption"});
  }
}
bool checkEdges(const GenericAnalysis &a,const ProgramCommunicationPlan &c,std::string &error) {
  if(c.bodyFingerprint!=a.bodyFingerprint)return fail(error,"source fingerprint mismatch");
  std::vector<std::vector<size_t>> users(c.copies.size()),edges(c.copies.size());
  std::set<std::pair<size_t,size_t>> rawTasks;
  std::map<size_t,size_t> first,last;size_t nextData=0,nextOrder=0;
  for(size_t i=0;i<c.events.size();++i) {
    const auto &event=c.events[i];
    for(size_t copy:event.reads)users[copy].push_back(i);
    if(event.kind=="transfer") {
      if(!event.dataEdges.empty() || !event.orderEdges.empty())return fail(error,"transfer fabricated logical edges");
      continue;
    }
    first.emplace(event.task,i);last[event.task]=i;
    if(event.dataEdges.size()!=event.reads.size())return fail(error,"consumer data edge coverage mismatch");
    for(size_t k=0;k<event.reads.size();++k) {
      size_t copyId=event.reads[k];const auto &copy=c.copies[copyId];const auto &version=c.versions[copy.version];
      if(event.dataEdges[k]!=nextData || nextData>=c.dataEdges.size())return fail(error,"data edge incidence mismatch");
      const auto &edge=c.dataEdges[nextData];const auto &r=c.regions[version.region];
      if(edge.producer!=version.producer || edge.consumer!=i || edge.region!=version.region ||
         edge.version!=copy.version || edge.copy!=copyId || edge.requirement.memory!=c.allocations[copy.allocation].memory ||
         edge.requirement.layout!=copy.layout || edge.requirement.dtype!=a.storages[r.storage].dtype)
        return fail(error,"data edge version/copy/consumer requirement mismatch");
      edges[copyId].push_back(nextData++);
      if(version.producer>=0)rawTasks.insert({c.events[version.producer].task,event.task});
    }
  }
  std::vector<std::vector<size_t>> orderIncidence(c.events.size());
  for(size_t task=0;task<a.tasks.size();++task)for(size_t dep:a.tasks[task].dependencies) {
    if(rawTasks.count({dep,task}) || !last.count(dep) || !first.count(task))continue;
    if(nextOrder>=c.orderEdges.size())return fail(error,"missing pure source order edge");
    const auto &edge=c.orderEdges[nextOrder];
    if(edge.producer!=last[dep] || edge.consumer!=first[task] || edge.producer>=edge.consumer ||
       edge.reason!="source task dependency without value consumption")return fail(error,"pure source ordering edge mismatch");
    orderIncidence[edge.consumer].push_back(nextOrder++);
  }
  if(nextData!=c.dataEdges.size() || nextOrder!=c.orderEdges.size())return fail(error,"fabricated logical edges");
  for(size_t i=0;i<c.events.size();++i)if(c.events[i].orderEdges!=orderIncidence[i])return fail(error,"order edge incidence mismatch");
  for(size_t i=0;i<c.copies.size();++i)
    if(c.copies[i].users!=users[i] || c.copies[i].consumerEdges!=edges[i] || c.copies[i].layout!="row_major")
      return fail(error,"copy user/edge/layout contract mismatch");
  return true;
}
} // namespace

bool orbit::taskflow::buildProgramCommunication(const GenericAnalysis &a,
                                                GenericMappingPlan &p,
                                                std::string &error) {
  if(p.choices.grouping=="tile-major" && (genericResidencyGroups(p.choices).empty()||llvm::any_of(genericResidencyGroups(p.choices),[](const auto &g){return g.tasks.empty()||g.retainedStorages.empty();})))return fail(error,"tile-major communication requires a nonempty task group");
  size_t previousScopeEnd=0;
  for(const auto &scope:p.residencyScopes){
    if(scope.tasks.size()<2||scope.retainedStorages.empty()||scope.beginPhase<previousScopeEnd||scope.beginPhase>=scope.endPhase||scope.endPhase>p.phases.size()||(scope.endPhase-scope.beginPhase)%scope.tasks.size()!=0)return fail(error,"invalid resolved residency scope bounds or block width");
    if(scope.inputReleasePolicy!="group-tail"&&scope.inputReleasePolicy!="last-reader")return fail(error,"unknown residency input release policy");
    previousScopeEnd=scope.endPhase;
  }
  p.communication = ProgramCommunicationPlan{};
  std::vector<GenericTilePhase> resident;
  const std::vector<GenericTilePhase> *phases;
  if (!phasesFor(a,p,resident,phases,error)) return false;
  auto &c = p.communication;
  c.bodyFingerprint = a.bodyFingerprint;
  if (!parseProgramDataPolicy(p.choices.dataOrganization,c.dataPolicy,error)) return false;
  if(!checkGenericMemory(p.memoryBinding,p.capacityBytes,error)) return false;
  c.domains=p.memoryBinding.architecture.domains;
  c.memories=p.memoryBinding.architecture.memories;
  c.resources=p.memoryBinding.architecture.resources;
  c.links=p.memoryBinding.architecture.links;
  for(const auto &t:p.transfers){
    const auto &b=p.memoryBinding;auto id=t.direction=="host_to_l1"?b.inputLink:b.outputLink;
    auto l=std::find_if(c.links.begin(),c.links.end(),[&](const auto &link){return link.id==id;});
    if(l==c.links.end() || t.bytes<l->minBytes || t.bytes>l->maxBytes)
      return fail(error,"transfer exceeds target link byte limits");
  }
  collectGenericAllocations(a,p,c.allocations);
  State state(p,a,*phases);
  if (!state.setup(error)) return false;
  if (p.strategy != "serial-tiled") {
    for (size_t i = 0; i < p.transfers.size(); ++i)
      if (p.transfers[i].direction == "host_to_l1" && !state.transfer(i,0,error)) return false;
  }
  std::map<size_t,std::vector<const GenericTileRetention *>> initializers;
  for (const auto &scope : p.retainedRegions)
    if (scope.initializerTask >= 0) initializers[scope.beginPhase].push_back(&scope);
  size_t transferIndex = 0;
  std::vector<size_t> deferredOutputs;
  std::set<size_t> groupImported;

  std::set<size_t> fused(p.fusedInitializerTasks.begin(),p.fusedInitializerTasks.end());
  for (size_t phase = 0; phase < phases->size(); ++phase) {
    if (p.strategy == "serial-tiled" && fused.count((*phases)[phase].task)) continue;
    size_t end = transferIndex;
    if(p.strategy=="serial-tiled"&&p.targetCapabilities.acquireAllInputsAtGroupEntry){
      const auto *scope=genericResidencyScope(p,phase);
      if(scope&&(phase-scope->beginPhase)%scope->tasks.size()==0){
        for(size_t t=transferIndex;t<p.transfers.size()&&p.transfers[t].phase<phase+scope->tasks.size();++t)
          if(p.transfers[t].direction=="host_to_l1"){
            if(!state.transfer(t,p.transfers[t].phase,error))return false;
            groupImported.insert(t);
          }
      }
    }
    if (p.strategy == "serial-tiled") {
      while (end < p.transfers.size() && p.transfers[end].phase == phase) ++end;
      for (size_t t = transferIndex; t < end; ++t)
        if (p.transfers[t].direction == "host_to_l1" && !groupImported.count(t) && !state.transfer(t,phase,error)) return false;
    }
    for (auto scope : initializers[phase])
      if (!state.compute(phase,scope->initializerTask,scope->storage,error)) return false;
    if (!state.compute(phase,-1,0,error)) return false;
    if (p.strategy == "serial-tiled") {
      for (size_t t = transferIndex; t < end; ++t)
        if (p.transfers[t].direction == "l1_to_host") {
          if(genericResidencyScope(p,phase))deferredOutputs.push_back(t);
          else if(!state.transfer(t,phase,error))return false;
        }
      const auto *scope=genericResidencyScope(p,phase);
      if(scope && (phase+1-scope->beginPhase)%scope->tasks.size()==0) {
        for(auto t:deferredOutputs)if(!state.transfer(t,p.transfers[t].phase,error))return false;
        deferredOutputs.clear();
      }
      transferIndex = end;
    }
  }
  if (p.strategy != "serial-tiled")
    for (size_t i = 0; i < p.transfers.size(); ++i)
      if (p.transfers[i].direction == "l1_to_host" && !state.transfer(i,phases->size()-1,error)) return false;
  for (auto &copy : c.copies) {
    if (copy.releasedAfter == SIZE_MAX) copy.releasedAfter = c.events.size();
    const auto &allocation = c.allocations[copy.allocation];
    if (allocation.memory != p.memoryBinding.hostMemory) {
      auto end = std::find_if(c.events.begin(),c.events.end(),
          [&](const auto &event) { return event.phase >= allocation.endPhase; });
      copy.releasedAfter = std::min(copy.releasedAfter,size_t(end-c.events.begin()));
    }
    // An event reads its old copies before writing replacement copies. Public
    // validity is half-open, so that final read belongs to [available, event+1).
    if(!copy.users.empty() && copy.users.back()==copy.releasedAfter && copy.releasedAfter<c.events.size())
      ++copy.releasedAfter;
  }
  for (auto &event : c.events) {
    auto &deps = event.dependencies;
    std::sort(deps.begin(),deps.end());
    deps.erase(std::unique(deps.begin(),deps.end()),deps.end());
  }
  populateEdges(a,c);
  for(const auto &transfer:p.transfers) {
    if(transfer.bytes<0 || c.transferBytes>INT64_MAX-transfer.bytes) return fail(error,"transfer total overflow");
    c.transferBytes+=transfer.bytes;
  }
  MappingExecutionRegion call;call.id="program-call";
  call.placement=nativeDevice(p);call.invocationLifecycle="host-device";call.submissionResource=p.memoryBinding.submissionResource;call.endEvent=c.events.size();
  for(size_t stage=0;stage<std::max<size_t>(1,p.hardwareStages.size());++stage)call.deviceStages.push_back(stage);
  c.executionRegions.push_back(std::move(call));
  if(!finalizeProgramGraph(a,c,&p,error))return false;
  p.executionGroups=c.executionGroups;
  return true;
}

bool orbit::taskflow::checkProgramCommunication(const GenericAnalysis &a,
    const GenericMappingPlan &p, std::string &error) {
  if(p.choices.grouping=="tile-major" && (genericResidencyGroups(p.choices).empty()||llvm::any_of(genericResidencyGroups(p.choices),[](const auto &g){return g.tasks.empty()||g.retainedStorages.empty();})))return fail(error,"tile-major communication requires a nonempty task group");
  size_t previousScopeEnd=0;
  for(const auto &scope:p.residencyScopes){
    if(scope.tasks.size()<2||scope.retainedStorages.empty()||scope.beginPhase<previousScopeEnd||scope.beginPhase>=scope.endPhase||scope.endPhase>p.phases.size()||(scope.endPhase-scope.beginPhase)%scope.tasks.size()!=0)return fail(error,"invalid resolved residency scope bounds or block width");
    if(scope.inputReleasePolicy!="group-tail"&&scope.inputReleasePolicy!="last-reader")return fail(error,"unknown residency input release policy");
    previousScopeEnd=scope.endPhase;
  }
  const auto &c = p.communication;
  if(c.executionGroups.size()!=p.executionGroups.size())return fail(error,"public execution group witness coverage mismatch");
  for(size_t i=0;i<c.executionGroups.size();++i){const auto &x=c.executionGroups[i],&y=p.executionGroups[i];if(std::tie(x.beginPhase,x.endPhase,x.resource,x.memory,x.dependencies,x.memberEvents,x.spatialOwner,x.implementationDependencies)!=std::tie(y.beginPhase,y.endPhase,y.resource,y.memory,y.dependencies,y.memberEvents,y.spatialOwner,y.implementationDependencies))return fail(error,"public execution group differs from native witness");}
  std::map<size_t,std::string> inputs,outputs;
  for(size_t storage=0;storage<a.storages.size();++storage) {
    if(a.storages[storage].role=="input")inputs[storage]=p.memoryBinding.hostMemory;
    if(a.storages[storage].role=="output")outputs[storage]=p.memoryBinding.hostMemory;
  }
  if(c.inputLocations!=inputs || c.outputLocations!=outputs)
    return fail(error,"native graph input/output boundary differs from source roles and host binding");
  if(c.clockMHz || c.allocationLifetime!="phase-pool")return fail(error,"native graph requires unknown clock and phase-pool allocation lifetime");
  if(c.executionRegions.size()!=1)return fail(error,"one checked AIR program-call region required");
  const auto &call=c.executionRegions.front();std::vector<size_t> stages;
  for(size_t i=0;i<std::max<size_t>(1,p.hardwareStages.size());++i)stages.push_back(i);
  if(call.id!="program-call" || call.placement!=nativeDevice(p) || call.invocationLifecycle!="host-device" || call.submissionResource!=p.memoryBinding.submissionResource ||
     call.beginEvent!=0 || call.endEvent!=c.events.size() || call.submissionCount!=1 || call.deviceStages!=stages)
    return fail(error,"program-call coverage/submission/stage contract mismatch");
  ProgramDataPolicy policy;
  if (!parseProgramDataPolicy(p.choices.dataOrganization,policy,error) || c.dataPolicy!=policy || !c.placements.empty())
    return fail(error,"data policy differs from checked mapping choice");
  std::vector<GenericTilePhase> resident;
  const std::vector<GenericTilePhase> *phases;
  if (!phasesFor(a,p,resident,phases,error)) return false;
  std::vector<MappingRegion> regions;
  Partition partition;
  if (!partition.build(a,*phases,p,regions,error)) return false;
  if (regions.size() != c.regions.size()) return fail(error,"logical region coverage mismatch");
  for (size_t i = 0; i < regions.size(); ++i)
    if (!same(regions[i],c.regions[i])) return fail(error,"logical region is not derived from source accesses");
  std::vector<MappingAllocation> allocations;
  collectGenericAllocations(a,p,allocations);
  if (allocations.size() != c.allocations.size()) return fail(error,"allocation coverage mismatch");
  for (size_t i = 0; i < allocations.size(); ++i) {
    const auto &x = allocations[i], &y = c.allocations[i];
    if (x.storage != y.storage || x.memory != y.memory || x.key != y.key ||
        x.bytes != y.bytes || x.beginPhase != y.beginPhase || x.endPhase != y.endPhase || x.addressing != y.addressing || x.spatialOwner != y.spatialOwner)
      return fail(error,"allocation placement, size or lifetime mismatch");
  }
  if(!checkGenericMemory(p.memoryBinding,p.capacityBytes,error)) return false;
  StaticConfig actual;actual.memories=c.memories;actual.resources=c.resources;actual.links=c.links;
  actual.domains=c.domains;
  if(staticFingerprint(serializeMemoryArchitecture(actual)) !=
     staticFingerprint(serializeMemoryArchitecture(p.memoryBinding.architecture)))
    return fail(error,"target memory/resource/path contract changed");
  // Check lifetimes as actual storage allocation intervals, independent of
  // data liveness. Dormant full/tail template pools still consume capacity.
  std::map<std::string,std::map<size_t,int64_t>> deltas;
  for (const auto &allocation : allocations) {
    if (allocation.bytes <= 0 || allocation.beginPhase >= allocation.endPhase)
      return fail(error,"invalid memory allocation lifetime");
    auto &starts = deltas[allocation.memory][allocation.beginPhase];
    auto &ends = deltas[allocation.memory][allocation.endPhase];
    if (starts > INT64_MAX-allocation.bytes) return fail(error,"memory sum overflow");
    starts += allocation.bytes; ends -= allocation.bytes;
  }
  for (const auto &memory : c.memories) {
    int64_t current = 0;
    for (auto delta : deltas[memory.id]) {
      if (delta.second > 0 && current > INT64_MAX-delta.second)
        return fail(error,"memory peak overflow");
      current += delta.second;
      if (current < 0 || current > memory.capacityBytes)
        return fail(error,"memory allocation capacity exceeded");
    }
  }
  std::vector<int64_t> latest(regions.size(),-1);
  std::vector<std::map<size_t,size_t>> available(allocations.size());
  std::vector<size_t> host(a.storages.size(),SIZE_MAX);
  for (size_t i = 0; i < allocations.size(); ++i)
    if (allocations[i].memory == p.memoryBinding.hostMemory) host[allocations[i].storage] = i;
  auto local = [&](size_t phase, const GenericTileRegion &r) -> int64_t {
    std::string key = p.strategy == "serial-tiled" ? poolKey(p,(*phases)[phase],phase,r)
                  : "storage-"+std::to_string(r.storage)+".data";
    for (size_t i = 0; i < allocations.size(); ++i)
      if (allocations[i].memory == p.memoryBinding.localMemory && allocations[i].key == key &&
          allocations[i].storage == r.storage && allocations[i].beginPhase <= phase &&
          phase < allocations[i].endPhase) return i;
    return -1;
  };
  std::vector<bool> seenVersions(c.versions.size()), seenCopies(c.copies.size());
  std::vector<size_t> expectedRelease(c.copies.size(),c.events.size());
  std::vector<std::vector<size_t>> expectedUsers(c.copies.size()), allUsers(c.copies.size());
  for (size_t v = 0; v < c.versions.size(); ++v) {
    const auto &version = c.versions[v];
    if (version.region >= regions.size() || version.producer < -1 ||
        version.producer >= int64_t(c.events.size())) return fail(error,"invalid version producer/region");
    if (version.producer >= 0) continue;
    size_t cell = version.region;
    if (a.storages[regions[cell].storage].role != "input" || latest[cell] >= 0)
      return fail(error,"fabricated or duplicate initial version");
    latest[cell] = v; seenVersions[v] = true;
  }
  for (size_t copy = 0; copy < c.copies.size(); ++copy) {
    const auto &value = c.copies[copy];
    if (value.version >= c.versions.size() || value.allocation >= allocations.size() ||
        value.createdBy < -1 || value.createdBy >= int64_t(c.events.size()))
      return fail(error,"invalid copy identity/producer/allocation");
    size_t cell = c.versions[value.version].region;
    const auto &allocation = allocations[value.allocation];
    if (regions[cell].storage != allocation.storage || bytes(a,regions[cell]) > allocation.bytes)
      return fail(error,"copy region exceeds its physical allocation");
    if (value.createdBy >= 0) continue;
    if (c.versions[value.version].producer != -1 || value.availableFrom != 0 ||
        allocation.memory != p.memoryBinding.hostMemory ||
        available[value.allocation].count(cell))
      return fail(error,"fabricated or duplicate initial location copy");
    available[value.allocation][cell] = copy; seenCopies[copy] = true;
  }
  for (size_t cell = 0; cell < regions.size(); ++cell)
    if (a.storages[regions[cell].storage].role == "input" &&
        (latest[cell] < 0 || !available[host[regions[cell].storage]].count(cell)))
      return fail(error,"missing initial input version/copy");
  std::map<size_t,size_t> transferAt;
  for (size_t i = 0; i < p.transfers.size(); ++i) {
    const auto &transfer = p.transfers[i];
    if (transfer.event < 0 || transfer.event >= int64_t(c.events.size()) ||
        !transferAt.emplace(transfer.event,i).second)
      return fail(error,"missing or duplicate transfer event binding");
  }
  std::set<size_t> fused(p.fusedInitializerTasks.begin(),p.fusedInitializerTasks.end());
  std::vector<bool> computations(phases->size());
  std::set<std::tuple<size_t,size_t,size_t>> initializers;
  for (const auto &scope : p.retainedRegions)
    if (scope.initializerTask >= 0)
      initializers.emplace(scope.beginPhase,scope.initializerTask,scope.storage);
  size_t previousPhase = 0;
  std::map<size_t,std::vector<size_t>> groupFrontiers;
  std::set<size_t> phaseBarrier;
  size_t phaseBegin = 0;
  for (size_t index = 0; index < c.events.size(); ++index) {
    const auto &event = c.events[index];
    if (event.phase >= phases->size() || (p.choices.grouping!="tile-major" && event.phase < previousPhase))
      return fail(error,"event execution schedule violates task instance order");
    if (p.strategy!="serial-tiled" && index && event.phase != previousPhase) {
      phaseBarrier.clear();
      for (size_t prior = phaseBegin; prior < index; ++prior) phaseBarrier.insert(prior);
      for (size_t prior = phaseBegin; prior < index; ++prior)
        for (size_t parent : c.events[prior].dependencies) phaseBarrier.erase(parent);
      phaseBegin = index;
    }
    previousPhase = event.phase;
    const auto &phase = (*phases)[event.phase];
    std::set<size_t> dependency;
    auto depend = [&](int64_t producer) {
      if (producer >= 0 && producer != int64_t(index)) dependency.insert(producer);
    };
    if(p.strategy=="serial-tiled") {auto group=serialGroupPredecessors(p,c.events,index,event.kind,event.phase,event.link,groupFrontiers);for(auto parent:group)if(parent<index)dependency.insert(parent);}
    else dependency.insert(phaseBarrier.begin(),phaseBarrier.end());
    auto beforeOverwrite = [&](size_t copy) {
      depend(c.copies[copy].createdBy);
      const auto &users=p.strategy=="serial-tiled"?allUsers[copy]:expectedUsers[copy];
      for (auto user : users) depend(user);
    };
    using Key = std::pair<size_t,size_t>; // allocation, logical cell
    std::set<Key> expectedReads, expectedWrites;
    bool transfer = event.kind == "transfer";
    size_t transferIndex = SIZE_MAX;
    if (transfer) {
      auto found = transferAt.find(index);
      if (found == transferAt.end()) return fail(error,"unplanned transfer event");
      transferIndex = found->second;
      const auto &t = p.transfers[transferIndex];
      if(!event.transferRegion || !same(*event.transferRegion,region(t,a)) || event.transferBytes!=t.bytes)
        return fail(error,"common transfer payload differs from native DMA plan");
      if ((p.strategy == "serial-tiled" && t.phase != event.phase) || event.task != phase.task ||
          event.link != (t.direction=="host_to_l1"?p.memoryBinding.inputLink:p.memoryBinding.outputLink) || event.resource != p.memoryBinding.dmaResource ||
          event.reads != t.sourceCopies || event.writes != t.targetCopies)
        return fail(error,"DMA event differs from actual source/destination binding");
      GenericTileRegion demand;
      demand.storage = t.storage;
      auto foundRegion = std::find_if(phase.regions.begin(),phase.regions.end(),
          [&](const auto &r) { return r.storage == t.storage; });
      int64_t al = foundRegion == phase.regions.end()
          ? local(event.phase,demand) : local(event.phase,*foundRegion);
      if (al < 0 || host[t.storage] == SIZE_MAX) return fail(error,"DMA placement inaccessible");
      bool input = t.direction == "host_to_l1";
      int64_t countBytes = 0;
      for (size_t cell : partition.cells(region(t,a))) {
        expectedReads.emplace(input ? host[t.storage] : size_t(al),cell);
        expectedWrites.emplace(input ? size_t(al) : host[t.storage],cell);
        countBytes += bytes(a,regions[cell]);
      }
      if (countBytes != t.bytes) return fail(error,"DMA cell bytes differ from planned transfer");
      auto link=std::find_if(c.links.begin(),c.links.end(),[&](const auto &l){return l.id==event.link;});
      if(link==c.links.end() || t.bytes<link->minBytes || t.bytes>link->maxBytes)
        return fail(error,"DMA violates target transfer size capability");
    } else {
      if(event.transferRegion || event.transferBytes)return fail(error,"compute event cannot carry transfer payload");
      if (event.resource != p.memoryBinding.computeResource || !event.link.empty())
        return fail(error,"compute event uses an invalid execution resource");
      int64_t initializedStorage = -1;
      if (event.kind == "initialize") {
        auto found = std::find_if(initializers.begin(),initializers.end(),[&](const auto &key) {
          return std::get<0>(key) == event.phase && std::get<1>(key) == event.task;
        });
        if (found == initializers.end()) return fail(error,"unproved/duplicate local initializer event");
        initializedStorage = std::get<2>(*found); initializers.erase(found);
      } else if (event.kind != "compute" || event.task != phase.task ||
                 fused.count(phase.task) || computations[event.phase] ||
                 (p.strategy == "serial-tiled" && p.phases[event.phase].computationEvent != int64_t(index)))
        return fail(error,"missing, duplicate or incorrect source task instance event");
      else computations[event.phase] = true;
      for (const auto &r : phase.regions) {
        if (initializedStorage >= 0 && r.storage != size_t(initializedStorage)) continue;
        int64_t al = local(event.phase,r);
        if (al < 0) return fail(error,"source task has no local allocation");
        for (size_t cell : partition.cells(region(r))) {
          if (r.read && initializedStorage < 0) expectedReads.emplace(al,cell);
          if (r.write) expectedWrites.emplace(al,cell);
        }
      }
    }
    std::set<Key> actualReads, actualWrites;
    for (size_t copy : event.reads) {
      if (copy >= c.copies.size() || !seenCopies[copy]) return fail(error,"copy used before creation");
      const auto &value = c.copies[copy];
      size_t cell = c.versions[value.version].region;
      const auto &al = allocations[value.allocation];
      auto active = available[value.allocation].find(cell);
      if (active == available[value.allocation].end() || active->second != copy ||
          int64_t(value.version) != latest[cell] || al.beginPhase > event.phase ||
          event.phase >= al.endPhase || !actualReads.emplace(value.allocation,cell).second)
        return fail(error,"read of stale, overwritten, released or duplicate copy");
      expectedUsers[copy] = {index};allUsers[copy].push_back(index);
      depend(value.createdBy);
    }
    if (actualReads != expectedReads) return fail(error,"task/transfer reads do not cover actual source access versions");
    std::set<size_t> reset;
    for (size_t copy : event.writes) {
      if (copy >= c.copies.size() || seenCopies[copy]) return fail(error,"duplicate or missing copy creation");
      const auto &value = c.copies[copy];
      size_t cell = c.versions[value.version].region;
      const auto &al = allocations[value.allocation];
      if (value.createdBy != int64_t(index) || value.availableFrom != index ||
          event.phase < al.beginPhase || event.phase >= al.endPhase ||
          !actualWrites.emplace(value.allocation,cell).second)
        return fail(error,"copy producer, validity or write coverage mismatch");
      if (transfer) {
        if (int64_t(value.version) != latest[cell]) return fail(error,"transfer fabricates or copies a stale version");
      } else {
        if (seenVersions[value.version] || c.versions[value.version].producer != int64_t(index))
          return fail(error,"task fails to produce its own unique source version");
        seenVersions[value.version] = true; latest[cell] = value.version;
      }
      if (al.memory != p.memoryBinding.hostMemory && reset.insert(value.allocation).second) {
        for (auto old : available[value.allocation]) {
          beforeOverwrite(old.second); expectedRelease[old.second] = index;
        }
        available[value.allocation].clear();
      }
      auto old = available[value.allocation].find(cell);
      if (old != available[value.allocation].end()) {
        beforeOverwrite(old->second); expectedRelease[old->second] = index;
      }
      available[value.allocation][cell] = copy; seenCopies[copy] = true;
    }
    if (actualWrites != expectedWrites) return fail(error,"task/transfer output copies miss actual written regions");
    if (event.dependencies != std::vector<size_t>(dependency.begin(),dependency.end()))
      return fail(error,"missing/fabricated group-control, core-order, copy-producer or overwrite dependency");
  }
  if (!initializers.empty()) return fail(error,"missing local source initializer");
  for (size_t phase = 0; phase < phases->size(); ++phase)
    if (!fused.count((*phases)[phase].task) && !computations[phase])
      return fail(error,"source task instance has no checked execution event");
  for (size_t i = 0; i < c.copies.size(); ++i)
  {
    const auto &allocation = allocations[c.copies[i].allocation];
    if (allocation.memory != p.memoryBinding.hostMemory) {
      auto end = std::find_if(c.events.begin(),c.events.end(),
          [&](const auto &event) { return event.phase >= allocation.endPhase; });
      expectedRelease[i] = std::min(expectedRelease[i],size_t(end-c.events.begin()));
    }
    // Replay the ordered read-before-write event boundary independently.
    if(!expectedUsers[i].empty() && expectedUsers[i].back()==expectedRelease[i] && expectedRelease[i]<c.events.size())
      ++expectedRelease[i];
    if (!seenCopies[i] || c.copies[i].releasedAfter != expectedRelease[i] ||
        c.copies[i].lastUsers != expectedUsers[i])
      return fail(error,"copy lifetime, final user or release/reuse mismatch");
  }
  if (std::find(seenVersions.begin(),seenVersions.end(),false) != seenVersions.end())
    return fail(error,"unproduced or unused fabricated version");
  for (size_t cell = 0; cell < regions.size(); ++cell) {
    size_t storage = regions[cell].storage;
    if (a.storages[storage].role != "output") continue;
    auto found = available[host[storage]].find(cell);
    if (latest[cell] < 0 || found == available[host[storage]].end() ||
        int64_t(c.copies[found->second].version) != latest[cell])
      return fail(error,"final source output version is not available at declared host boundary");
  }
  int64_t transferBytes=0;
  for(const auto &transfer:p.transfers) {
    if(transfer.bytes<0 || transferBytes>INT64_MAX-transfer.bytes)return fail(error,"transfer total overflow");
    transferBytes+=transfer.bytes;
  }
  if(c.transferBytes!=transferBytes || c.estimatedMakespanMs || !c.computeEstimates.empty() || !c.peakMemoryBytes.empty())return fail(error,"native uncalibrated cost or traffic mismatch");
  for(const auto &event:c.events)if(event.memory!=(event.kind=="transfer"?"":p.memoryBinding.localMemory))return fail(error,"event execution memory mismatch");
  if(!checkEdges(a,c,error))return false;
  size_t instance=0;
  for(size_t event=0;event<c.events.size();++event)if(c.events[event].kind!="transfer") {
    if(instance>=c.instances.size())return fail(error,"missing common native task instance");
    const auto &value=c.instances[instance++];const auto &execution=c.events[event];const auto &phase=(*phases)[execution.phase];
    auto implementation=std::find_if(p.taskImplementations.begin(),p.taskImplementations.end(),
        [&](const auto &candidate){return candidate.task==execution.task;});
    if(implementation==p.taskImplementations.end())return fail(error,"common native instance has no selected implementation");
    if(value.task!=execution.task || value.event!=event || value.phase!=execution.phase || value.domainTask!=phase.task ||
       value.kind!=(execution.kind=="initialize"?"source-initializer":"source-task") ||
       value.implementation!=implementation->kind || value.lanes!=implementation->lanes ||
       value.parallelLoop!=implementation->parallelLoop || value.domain.size()!=phase.loops.size())
      return fail(error,"common native instance source/phase/implementation mismatch");
    for(size_t loop=0;loop<phase.loops.size();++loop){const auto &bound=value.domain[loop];const auto &expected=phase.loops[loop];
      if(expected.loop>=a.tasks[phase.task].loops.size() || bound.loop!=expected.loop || bound.lower!=expected.lower || bound.upper!=expected.upper ||
         bound.step!=a.tasks[phase.task].loops[expected.loop].step)
        return fail(error,"common native instance loop domain differs from chosen source tile");}
  }
  if(instance!=c.instances.size())return fail(error,"fabricated common native task instance");
  return checkProgramGraphSource(a,c,error) && checkProgramGraph(c,error);
}

llvm::json::Value orbit::taskflow::serializeProgramCommunication(const GenericMappingPlan &p) {
  return serializeProgramCommunication(p.communication);
}
llvm::json::Value orbit::taskflow::serializeProgramCommunication(const ProgramCommunicationPlan &c) {
  bool compact = c.events.size() > 16384;
  llvm::MD5 hash;
  auto number = [&](int64_t n) { hash.update(std::to_string(n)); hash.update(";"); };
  auto word = [&](const std::string &s) { number(s.size()); hash.update(s); };
  auto indices = [&](const std::vector<size_t> &values) {
    number(values.size()); llvm::json::Array a;
    for (auto value : values) { number(value); if (!compact) a.push_back(int64_t(value)); }
    return a;
  };
  llvm::json::Array memories,resources,links,allocations,regions,versions,copies,events,dataEdges,orderEdges,estimates,placements,executionRegions;
  llvm::json::Object peaks;
  for(const auto &peak:c.peakMemoryBytes){word(peak.first);number(peak.second);peaks[peak.first]=peak.second;}
  word(programDataPolicyName(c.dataPolicy).str()); word(c.bodyFingerprint);
  number(c.transferBytes);
  auto jsonWord=[&](const llvm::json::Value &v){std::string value;llvm::raw_string_ostream stream(value);stream<<v;word(value);};
  jsonWord(c.estimatedMakespanMs ? llvm::json::Value(*c.estimatedMakespanMs) : llvm::json::Value(nullptr));
  for(const auto &estimate:c.computeEstimates)jsonWord(serializeMappingComputeEstimate(estimate));
  for(const auto &region:c.executionRegions){
    llvm::json::Array stages;for(auto stage:region.deviceStages)stages.push_back(int64_t(stage));
    llvm::json::Value record=llvm::json::Object{{"id",region.id},{"placement",region.placement},{"submission_resource",region.submissionResource},
      {"begin_event",int64_t(region.beginEvent)},{"end_event_exclusive",int64_t(region.endEvent)},{"device_stages",std::move(stages)},
      {"host_submission_count",region.submissionCount},{"submission_kind",region.submissionCount?"host_submission":"none"},
      {"cost_ownership",region.submissionCount?"prepared-region composite includes submission, device compute/transfers/wait and readback once":"host graph execution; no device submission charge"}};
    if(!region.invocationLifecycle.empty())(*record.getAsObject())["invocation_lifecycle"]=region.invocationLifecycle;
    jsonWord(record);executionRegions.push_back(std::move(record));
  }
  for(const auto &placement:c.placements){number(placement.task);word(placement.resource);word(placement.memory);word(placement.implementation);
    llvm::json::Object entry{{"task",int64_t(placement.task)},{"resource",placement.resource},{"memory",placement.memory},{"implementation",placement.implementation}};
    if(placement.resourceCount!=1 || !placement.resourceShape.empty()){number(placement.resourceCount);for(auto n:placement.resourceShape)number(n);entry["resource_count"]=placement.resourceCount;llvm::json::Array shape;for(auto n:placement.resourceShape)shape.push_back(n);entry["resource_shape"]=std::move(shape);}
    placements.push_back(std::move(entry));}
  StaticConfig architecture;architecture.domains=c.domains;architecture.memories=c.memories;architecture.resources=c.resources;architecture.links=c.links;
  auto architectureJSON=serializeMemoryArchitecture(architecture);jsonWord(architectureJSON);
  for (const auto &memory : c.memories) {
    word(memory.id); number(memory.capacityBytes);
    number(memory.accessibleDevices.size());for(const auto &device:memory.accessibleDevices)word(device);
    llvm::json::Object entry{{"id",memory.id},{"capacity_bytes",memory.capacityBytes}};
    if(!memory.capacityKnown){entry["capacity_bytes"]=nullptr;entry["capacity_known"]=false;}
    if(memory.alignmentBytes!=1)entry["alignment_bytes"]=memory.alignmentBytes;
    memories.push_back(std::move(entry));
  }
  for (const auto &resource : c.resources) {
    word(resource.id); word(resource.kind); word(resource.device); number(resource.capacity);
    resources.push_back(llvm::json::Object{{"id",resource.id},{"kind",resource.kind},
      {"device",resource.device},{"capacity",resource.capacity}});
  }
  for (const auto &link : c.links) {
    word(link.id); word(link.source); word(link.target); word(link.resource);
    jsonWord(link.latencyMs);jsonWord(link.bandwidthBytesPerMs);word(link.provenance.kind);word(link.provenance.source);
    word(link.sourceLayout);word(link.targetLayout);word(link.sourceDtype);word(link.targetDtype);
    number(link.minBytes);number(link.maxBytes);number(link.temporaryBytes);
    word(link.sourceDevice);word(link.targetDevice);word(link.memory);
    links.push_back(llvm::json::Object{{"id",link.id},{"source",link.source},
      {"target",link.target},{"resource",link.resource},
      {"latency_ms",link.latencyMs},{"bandwidth_bytes_per_ms",link.bandwidthBytesPerMs},
      {"provenance",link.provenance.source},{"cost_status",link.bandwidthBytesPerMs>0?"target_estimate":"uncalibrated"}});
  }
  for (const auto &allocation : c.allocations) {
    number(allocation.storage); word(allocation.memory); word(allocation.key);
    number(allocation.bytes); number(allocation.beginPhase); number(allocation.endPhase);
    allocations.push_back(llvm::json::Object{{"storage",int64_t(allocation.storage)},
      {"memory",allocation.memory},{"key",allocation.key},{"bytes",allocation.bytes},
      {"begin_phase",int64_t(allocation.beginPhase)},
      {"end_phase_exclusive",int64_t(allocation.endPhase)}});
  }
  for (const auto &r : c.regions) {
    number(r.storage); number(r.sizes.size()); llvm::json::Array offsets,sizes;
    for (auto n : r.offsets) { number(n); if (!compact) offsets.push_back(n); }
    for (auto n : r.sizes) { number(n); if (!compact) sizes.push_back(n); }
    if (!compact) regions.push_back(llvm::json::Object{{"storage",int64_t(r.storage)},
        {"offsets",std::move(offsets)},{"sizes",std::move(sizes)}});
  }
  for (const auto &v : c.versions) {
    number(v.region); number(v.producer);
    if (!compact) versions.push_back(llvm::json::Object{{"region",int64_t(v.region)}, {"producer",v.producer}});
  }
  for (const auto &copy : c.copies) {
    number(copy.version); number(copy.allocation); number(copy.createdBy);
    number(copy.availableFrom); number(copy.releasedAfter);
    auto users = indices(copy.lastUsers);
    auto allUsers=indices(copy.users), consumerEdges=indices(copy.consumerEdges); word(copy.layout);
    if (!compact) copies.push_back(llvm::json::Object{{"version",int64_t(copy.version)},
      {"allocation",int64_t(copy.allocation)},{"created_by",copy.createdBy},
      {"available_from",int64_t(copy.availableFrom)},
      {"released_after",int64_t(copy.releasedAfter)},{"last_users",std::move(users)},{"users",std::move(allUsers)},
      {"consumer_edges",std::move(consumerEdges)},{"layout",copy.layout}});
  }
  for (const auto &event : c.events) {
    word(event.kind); number(event.phase); number(event.task); word(event.resource); word(event.link);
    auto deps = indices(event.dependencies), reads = indices(event.reads), writes = indices(event.writes);
    auto de=indices(event.dataEdges),oe=indices(event.orderEdges); word(event.memory);
    if (!compact) events.push_back(llvm::json::Object{{"kind",event.kind},
      {"phase",int64_t(event.phase)},{"task",int64_t(event.task)},
      {"resource",event.resource},{"link",event.link},{"depends_on",std::move(deps)},
      {"reads",std::move(reads)},{"writes",std::move(writes)},
      {"data_edges",std::move(de)},{"order_edges",std::move(oe)},{"memory",event.memory}});
  }
  for(const auto &edge:c.dataEdges) {
    number(edge.producer); number(edge.consumer); number(edge.region); number(edge.version); number(edge.copy);
    word(edge.requirement.memory); word(edge.requirement.layout); word(edge.requirement.dtype);
    if(!compact) dataEdges.push_back(llvm::json::Object{{"producer",edge.producer},
      {"consumer",int64_t(edge.consumer)},{"region",int64_t(edge.region)},
      {"version",int64_t(edge.version)},{"copy",int64_t(edge.copy)},
      {"required_memory",edge.requirement.memory},{"required_layout",edge.requirement.layout},
      {"required_dtype",edge.requirement.dtype}});
  }
  for(const auto &edge:c.orderEdges) {
    number(edge.producer);number(edge.consumer);word(edge.reason);
    if(!compact) orderEdges.push_back(llvm::json::Object{{"producer",int64_t(edge.producer)},
      {"consumer",int64_t(edge.consumer)},{"reason",edge.reason}});
  }
  for(const auto &estimate:c.computeEstimates) estimates.push_back(serializeMappingComputeEstimate(estimate));
  llvm::MD5::MD5Result result; hash.final(result);
  llvm::SmallString<32> digest; llvm::MD5::stringifyResult(result,digest);
  return llvm::json::Object{{"schema","orbit.program.communication.v2"},
    {"data_organization",programDataPolicyName(c.dataPolicy).str()},{"body_fingerprint",c.bodyFingerprint},
    {"data_edges",std::move(dataEdges)},{"order_edges",std::move(orderEdges)},
    {"data_edge_count",int64_t(c.dataEdges.size())},{"order_edge_count",int64_t(c.orderEdges.size())},
    {"task_placements",std::move(placements)},{"compute_estimates",std::move(estimates)},
    {"estimated_makespan_ms",c.estimatedMakespanMs ? llvm::json::Value(*c.estimatedMakespanMs) : llvm::json::Value(nullptr)},
    {"transfer_bytes",c.transferBytes},
    {"adapter",c.placements.empty()?"native arbitrary-rank adapter of Static allocation/version/location-copy semantics":"whole-task exact-region adapter of actual Taskflow analysis to Static target contracts"},
    {"memory_peak_bytes",std::move(peaks)},
    {"memory_lifetime_scope",c.placements.empty()?"native physical pool phase intervals; copies have event validity":"event intervals count distinct allocations once; phase fields are conservative summaries"},
    {"schedule",c.placements.empty()?"explicit group control, core order and copy producer/overwrite dependencies; ordinals are a topological witness, not completion barriers":"serial topological whole-task events; compute and direct transfers charged once"},
    {"resource_capacity_scope","logical issue witness, not a measured DMA-engine count or a promise of serialized physical transfers"},
    {"system_capacity_scope","allocation accounting limit, not measured available host RAM"},
    {"execution_regions",std::move(executionRegions)},
    {"memory_architecture",std::move(architectureJSON)},
    {"memories",std::move(memories)},{"resources",std::move(resources)},
    {"links",std::move(links)},{"allocations",std::move(allocations)},
    {"logical_regions",std::move(regions)},{"versions",std::move(versions)},
    {"copies",std::move(copies)},{"events",std::move(events)},
    {"logical_region_count",int64_t(c.regions.size())},{"version_count",int64_t(c.versions.size())},
    {"copy_count",int64_t(c.copies.size())},{"event_count",int64_t(c.events.size())},
    {"expanded_details_omitted",compact},{"semantic_digest",digest.str().str()}};
}

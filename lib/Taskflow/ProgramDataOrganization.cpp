#include "Taskflow/ProgramMapping.h"
#include "Taskflow/GenericAnalysis.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
using namespace orbit::taskflow;
namespace {
bool reject(std::string &error, const std::string &message) {
  error="program data organization: "+message; return false;
}
struct Effects { std::set<size_t> reads,writes; };
bool sourceEffects(const GenericAnalysis &a,std::vector<Effects> &effects,std::string &error) {
  // This finite whole-task adapter requires exact whole-storage access sets.
  // Other native tiling adapters retain their finer arbitrary-rank cell model.
  effects.resize(a.tasks.size());
  for(size_t t=0;t<a.tasks.size();++t) for(const auto &access:a.tasks[t].accesses) {
    if(!access.exactRegion || access.region.size()!=a.storages[access.storage].shape.size())
      return reject(error,"whole-task adapter requires exact source access coverage");
    for(size_t d=0;d<access.region.size();++d)
      if(access.region[d].lower!=0 || access.region[d].upper!=a.storages[access.storage].shape[d])
        return reject(error,"whole-task adapter requires full-storage source regions");
    if(access.read) effects[t].reads.insert(access.storage);
    if(access.write) effects[t].writes.insert(access.storage);
  }
  return true;
}
int64_t storageBytes(const GenericStorage &s) {
  auto type=mlir::cast<mlir::MemRefType>(s.root.getType());
  int64_t n=(type.getElementType().getIntOrFloatBitWidth()+7)/8;
  for(int64_t x:s.shape) {if(x<=0 || n>INT64_MAX/x) return -1;n*=x;} return n;
}
bool placements(const GenericAnalysis &a,const StaticConfig &target,
    const ProgramDataOrganizationChoice &choice,std::vector<MappingTaskPlacement> &out,std::string &error) {
  if(choice.placements.size()!=a.tasks.size())return reject(error,"one explicit placement per source task required");
  if(!validateMemoryArchitecture(target,error))return false;
  out.resize(a.tasks.size());std::set<size_t> seen;
  for(const auto &p:choice.placements) {
    if(p.task>=a.tasks.size() || !seen.insert(p.task).second)return reject(error,"invalid or duplicate task placement");
    auto resource=std::find_if(target.resources.begin(),target.resources.end(),[&](const auto &r){return r.id==p.resource;});
    auto memory=std::find_if(target.memories.begin(),target.memories.end(),[&](const auto &m){return m.id==p.memory;});
    if(resource==target.resources.end() || resource->capacity<1 || resource->kind!="compute" ||
       memory==target.memories.end() || memory->capacityBytes<=0 ||
       !memoryAccessible(target,p.memory,p.resource))
      return reject(error,"task resource cannot access selected memory");
    out[p.task]=p;
  } return true;
}
const StaticTransfer *linkFor(const StaticConfig &target,const GenericStorage &s,
    const std::string &source,const std::string &dest,int64_t bytes) {
  for(const auto &l:target.links) if(l.source==source && l.target==dest &&
      (l.sourceLayout.empty() || l.sourceLayout=="row_major") &&
      (l.targetLayout.empty() || l.targetLayout=="row_major") &&
      (l.sourceDtype.empty() || l.sourceDtype==s.dtype) &&
      (l.targetDtype.empty() || l.targetDtype==s.dtype) &&
      bytes>=l.minBytes && bytes<=l.maxBytes && l.temporaryBytes==0) return &l;
  return nullptr;
}
} // namespace
bool orbit::taskflow::planProgramDataOrganization(const GenericAnalysis &a,
    const StaticConfig &target,const ProgramDataOrganizationChoice &choice,
    ProgramCommunicationPlan &c,std::string &error) {
  c={}; c.dataPolicy=choice.policy;c.bodyFingerprint=a.bodyFingerprint;
  c.domains=target.domains;c.memories=target.memories;c.resources=target.resources;c.links=target.links;
  if(!placements(a,target,choice,c.placements,error))return false;
  std::vector<Effects> effects;if(!sourceEffects(a,effects,error))return false;
  c.computeEstimates=choice.computeEstimates;
  std::map<size_t,double> computeMs;bool known=true;double duration=0;
  for(const auto &estimate:choice.computeEstimates) {
    double ms;
    if(estimate.task>=a.tasks.size() || computeMs.count(estimate.task) ||
       estimate.resource!=c.placements[estimate.task].resource ||
       estimate.implementation!=c.placements[estimate.task].implementation ||
       !convertComputeEstimateToMs(estimate,choice.clockMHz,ms,error))return reject(error,"invalid compute estimate: "+error);
    auto resource=std::find_if(target.resources.begin(),target.resources.end(),[&](const auto &r){return r.id==estimate.resource;});
    if(resource==target.resources.end() || estimate.resourceCount>resource->capacity)return reject(error,"compute estimate exceeds resource capacity");
    computeMs[estimate.task]=ms;
  }
  for(size_t s=0;s<a.storages.size();++s) {
    if(storageBytes(a.storages[s])<0)return reject(error,"storage byte count overflow");
    c.regions.push_back({s,std::vector<int64_t>(a.storages[s].shape.size(),0),a.storages[s].shape});
  }
  std::vector<int64_t> latest(a.storages.size(),-1);
  std::map<std::pair<size_t,std::string>,size_t> available;
  auto allocation=[&](size_t storage,const std::string &memory,size_t phase) {
    size_t i=c.allocations.size();c.allocations.push_back({storage,memory,"copy-pool-"+std::to_string(i),storageBytes(a.storages[storage]),phase,phase+1});return i;
  };
  auto create=[&](size_t version,size_t al,int64_t event) {
    size_t i=c.copies.size();MappingCopy copy;copy.version=version;copy.allocation=al;copy.createdBy=event;
    copy.availableFrom=event<0?0:size_t(event);c.copies.push_back(copy);
    available[{c.regions[c.versions[version].region].storage,c.allocations[al].memory}]=i;
    if(event>=0)c.events[event].writes.push_back(i);return i;
  };
  for(size_t s=0;s<a.storages.size();++s) if(a.storages[s].role=="input") {
    auto loc=target.assumptions.initialLocations.find(a.storages[s].id);
    if(loc==target.assumptions.initialLocations.end())return reject(error,"missing initial input location");
    if(std::none_of(target.memories.begin(),target.memories.end(),[&](const auto &m){return m.id==loc->second;}))return reject(error,"unknown initial memory");
    size_t v=c.versions.size();c.versions.push_back({s,-1});latest[s]=v;
    create(v,allocation(s,loc->second,0),-1);
  }
  auto start=[&](std::string kind,size_t task,std::string memory,std::string resource,std::string link) {
    MappingEvent e;e.kind=kind;e.task=task;e.phase=task;e.memory=memory;e.resource=resource;e.link=link;
    if(!c.events.empty())e.dependencies.push_back(c.events.size()-1);c.events.push_back(e);return c.events.size()-1;
  };
  auto read=[&](size_t copy,size_t event,bool data) {
    auto &v=c.copies[copy];c.events[event].reads.push_back(copy);v.users.push_back(event);v.lastUsers={event};
    if(v.createdBy>=0)c.events[event].dependencies.push_back(v.createdBy);
    if(data) {size_t edge=c.dataEdges.size();auto &version=c.versions[v.version];size_t s=c.regions[version.region].storage;
      c.dataEdges.push_back({version.producer,event,version.region,v.version,copy,{c.allocations[v.allocation].memory,v.layout,a.storages[s].dtype}});
      c.events[event].dataEdges.push_back(edge);v.consumerEdges.push_back(edge);}
  };
  std::function<bool(size_t,const std::string &,size_t,size_t &)> ensure;
  ensure=[&](size_t storage,const std::string &memory,size_t task,size_t &copy) {
    auto found=available.find({storage,memory});
    if(found!=available.end() && c.copies[found->second].version==size_t(latest[storage])) {copy=found->second;return true;}
    if(latest[storage]<0)return reject(error,"source read has no initialized version");
    size_t source=SIZE_MAX;std::vector<std::string> path;
    StaticConfig routes=target;routes.links.clear();int64_t b=storageBytes(a.storages[storage]);
    // Filter semantic constraints before connectivity, not after choosing a route.
    for(const auto &l:target.links)if(linkFor(target,a.storages[storage],l.source,l.target,b)==&l)routes.links.push_back(l);
    for(const auto &entry:available)if(entry.first.first==storage && c.copies[entry.second].version==size_t(latest[storage])) {
      std::vector<std::string> candidate;std::string ignored;
      if(findMemoryPath(routes,entry.first.second,memory,candidate,ignored) &&
         (source==SIZE_MAX || candidate.size()<path.size())){source=entry.second;path=std::move(candidate);}
    }
    if(source==SIZE_MAX || path.empty())return reject(error,"no legal transfer path for required copy");
    for(const auto &id:path) {
      auto link=std::find_if(target.links.begin(),target.links.end(),[&](const auto &l){return l.id==id;});
      size_t event=start("transfer",task,link->target,link->resource,link->id);read(source,event,false);
      copy=create(latest[storage],allocation(storage,link->target,task),event);
      if(c.transferBytes>INT64_MAX-b)return reject(error,"traffic overflow");c.transferBytes+=b;
      if(std::isfinite(link->bandwidthBytesPerMs) && std::isfinite(link->latencyMs) && link->bandwidthBytesPerMs>0 && link->latencyMs>=0 && !link->provenance.source.empty())duration+=link->latencyMs+b/link->bandwidthBytesPerMs;else known=false;
      source=copy;
    }
    return true;
  };
  for(size_t task=0;task<a.tasks.size();++task) {
    for(size_t dep:a.tasks[task].dependencies)if(dep>=task)return reject(error,"whole-task adapter requires source topological order");
    const auto &place=c.placements[task];std::vector<size_t> reads;
    for(size_t s:effects[task].reads) {size_t copy;if(!ensure(s,place.memory,task,copy))return false;reads.push_back(copy);}
    size_t event=start("compute",task,place.memory,place.resource,"");
    for(size_t copy:reads)read(copy,event,true);
    for(size_t s:effects[task].writes) {
      size_t v=c.versions.size();c.versions.push_back({s,int64_t(event)});latest[s]=v;
      create(v,allocation(s,place.memory,task),event);
    }
    if(computeMs.count(task))duration+=computeMs[task];else known=false;
    for(size_t dep:a.tasks[task].dependencies) {
      bool raw=false;for(size_t id:c.events[event].dataEdges){auto producer=c.dataEdges[id].producer;if(producer>=0 && c.events[producer].task==dep)raw=true;}
      if(!raw) {auto prior=std::find_if(c.events.rbegin(),c.events.rend(),[&](const auto &e){return e.kind=="compute" && e.task==dep;});
        if(prior==c.events.rend())return reject(error,"missing order predecessor");
        c.events[event].orderEdges.push_back(c.orderEdges.size());c.orderEdges.push_back({size_t(c.events.rend()-prior-1),event,"source task dependency without value consumption"});}
    }
    if(choice.policy==ProgramDataPolicy::PhaseLocal) {
      // Every imported location is task-local, including intermediate copies
      // along a multi-hop route. Only external and compute-produced copies
      // remain valid sources for a later task under this policy.
      for(auto it=available.begin();it!=available.end();) {
        auto creator=c.copies[it->second].createdBy;
        if(creator>=0 && c.events[creator].kind=="transfer")it=available.erase(it);
        else ++it;
      }
    }
  }
  for(size_t s=0;s<a.storages.size();++s)if(a.storages[s].role=="output") {
    auto dest=target.assumptions.outputLocations.find(a.storages[s].id);
    if(dest==target.assumptions.outputLocations.end())return reject(error,"missing output boundary location");
    size_t copy;if(!ensure(s,dest->second,a.tasks.size()-1,copy))return false;
  }
  for(auto &copy:c.copies) {
    copy.releasedAfter=copy.users.empty()?copy.availableFrom+1:copy.users.back()+1;
    auto &al=c.allocations[copy.allocation];
    if(copy.createdBy>=0)al.beginPhase=c.events[copy.createdBy].phase;
    al.endPhase=copy.users.empty()?al.beginPhase+1:c.events[copy.users.back()].phase+1;
    size_t storage=c.regions[c.versions[copy.version].region].storage;
    if(a.storages[storage].role=="output" && target.assumptions.outputLocations.at(a.storages[storage].id)==al.memory && copy.version==size_t(latest[storage])) {
      al.endPhase=a.tasks.size();copy.releasedAfter=c.events.size();
    }
  }
  for(auto &e:c.events){std::sort(e.dependencies.begin(),e.dependencies.end());e.dependencies.erase(std::unique(e.dependencies.begin(),e.dependencies.end()),e.dependencies.end());}
  for(const auto &memory:target.memories)for(size_t event=0;event<c.events.size();++event) {
    int64_t live=0;std::set<size_t> counted;
    for(const auto &copy:c.copies)if(copy.availableFrom<=event && event<copy.releasedAfter && counted.insert(copy.allocation).second) {
      const auto &al=c.allocations[copy.allocation];if(al.memory!=memory.id)continue;
      if(al.bytes>INT64_MAX-live)return reject(error,"memory accounting overflow");live+=al.bytes;
    }
    c.peakMemoryBytes[memory.id]=std::max(c.peakMemoryBytes[memory.id],live);
  }
  if(known && std::isfinite(duration))c.estimatedMakespanMs=duration;
  return checkProgramDataOrganization(a,target,choice,c,error);
}
bool orbit::taskflow::checkProgramDataOrganization(const GenericAnalysis &a,
    const StaticConfig &target,const ProgramDataOrganizationChoice &choice,
    const ProgramCommunicationPlan &c,std::string &error) {
  std::vector<Effects> effects;std::vector<MappingTaskPlacement> place;
  if(!sourceEffects(a,effects,error) || !placements(a,target,choice,place,error))return false;
  for(const auto &storage:a.storages)if(storageBytes(storage)<=0)
    return reject(error,"source storage byte count is invalid or overflows int64");
  if(!c.executionRegions.empty())return reject(error,"whole-task CPU planner has no declared offload submission boundary");
  if(c.dataPolicy!=choice.policy || c.bodyFingerprint!=a.bodyFingerprint ||
     c.regions.size()!=a.storages.size() || c.placements.size()!=place.size())return reject(error,"source/policy/placement coverage mismatch");
  for(size_t t=0;t<place.size();++t)if(c.placements[t].task!=t || c.placements[t].memory!=place[t].memory ||
      c.placements[t].resource!=place[t].resource || c.placements[t].implementation!=place[t].implementation)
    return reject(error,"source task placement changed");
  for(size_t s=0;s<a.storages.size();++s)if(c.regions[s].storage!=s ||
      c.regions[s].sizes!=a.storages[s].shape || c.regions[s].offsets!=std::vector<int64_t>(a.storages[s].shape.size(),0))
    return reject(error,"logical region differs from source");
  StaticConfig copyTarget;copyTarget.memories=c.memories;copyTarget.resources=c.resources;copyTarget.links=c.links;copyTarget.domains=c.domains;
  if(staticFingerprint(serializeMemoryArchitecture(copyTarget))!=staticFingerprint(serializeMemoryArchitecture(target)))
    return reject(error,"target architecture contract changed");
  // Target tables are contracts, not candidate-mutable resources.
  if(c.memories.size()!=target.memories.size() || c.resources.size()!=target.resources.size() || c.links.size()!=target.links.size())
    return reject(error,"target contract coverage changed");
  for(size_t i=0;i<c.memories.size();++i)if(c.memories[i].id!=target.memories[i].id ||
      c.memories[i].capacityBytes!=target.memories[i].capacityBytes || c.memories[i].accessibleDevices!=target.memories[i].accessibleDevices)
    return reject(error,"memory contract changed");
  for(size_t i=0;i<c.resources.size();++i)if(c.resources[i].id!=target.resources[i].id || c.resources[i].kind!=target.resources[i].kind ||
      c.resources[i].device!=target.resources[i].device || c.resources[i].capacity!=target.resources[i].capacity)
    return reject(error,"resource contract changed");
  for(size_t i=0;i<c.links.size();++i) {
    const auto &x=c.links[i],&y=target.links[i];
    if(x.id!=y.id || x.source!=y.source || x.target!=y.target || x.resource!=y.resource || x.latencyMs!=y.latencyMs ||
       x.bandwidthBytesPerMs!=y.bandwidthBytesPerMs || x.minBytes!=y.minBytes || x.maxBytes!=y.maxBytes ||
       x.sourceLayout!=y.sourceLayout || x.targetLayout!=y.targetLayout || x.sourceDtype!=y.sourceDtype || x.targetDtype!=y.targetDtype ||
       x.sourceDevice!=y.sourceDevice || x.targetDevice!=y.targetDevice || x.memory!=y.memory || x.temporaryBytes!=y.temporaryBytes ||
       x.provenance.kind!=y.provenance.kind || x.provenance.source!=y.provenance.source)
      return reject(error,"link contract changed");
  }
  std::set<std::string> keys;
  for(const auto &al:c.allocations) {
    if(al.storage>=a.storages.size() || al.bytes<=0 || al.bytes!=storageBytes(a.storages[al.storage]) || !keys.insert(al.key).second ||
       al.beginPhase>=al.endPhase || al.endPhase>a.tasks.size() ||
       std::none_of(target.memories.begin(),target.memories.end(),[&](const auto &m){return m.id==al.memory;}))
      return reject(error,"invalid allocation region/size/location/lifetime");
  }
  std::vector<bool> seenCopies(c.copies.size()),seenVersions(c.versions.size()),seenTasks(a.tasks.size());
  std::vector<int64_t> latest(a.storages.size(),-1);
  std::vector<std::vector<size_t>> users(c.copies.size()),consumerEdges(c.copies.size());
  std::vector<size_t> creatorCount(c.allocations.size());
  for(size_t i=0;i<c.versions.size();++i)if(c.versions[i].region>=c.regions.size() || c.versions[i].producer< -1 ||
      c.versions[i].producer>=int64_t(c.events.size()))return reject(error,"version has invalid region or producer");
  for(size_t i=0;i<c.copies.size();++i) {
    const auto &copy=c.copies[i];
    if(copy.version>=c.versions.size() || copy.allocation>=c.allocations.size() || copy.createdBy< -1 ||
       copy.createdBy>=int64_t(c.events.size()) || copy.layout!="row_major" ||
       copy.availableFrom>copy.releasedAfter || copy.releasedAfter>c.events.size() ||
       c.allocations[copy.allocation].storage!=c.regions[c.versions[copy.version].region].storage)
      return reject(error,"invalid copy version/allocation/layout/lifetime");
    ++creatorCount[copy.allocation];
    if(copy.createdBy<0) {
      if(copy.createdBy!=-1)return reject(error,"invalid external copy creator");
      auto s=c.regions[c.versions[copy.version].region].storage;
      auto loc=target.assumptions.initialLocations.find(a.storages[s].id);
      if(a.storages[s].role!="input" || c.versions[copy.version].producer!=-1 || latest[s]>=0 ||
         copy.availableFrom!=0 || loc==target.assumptions.initialLocations.end() || loc->second!=c.allocations[copy.allocation].memory)
        return reject(error,"fabricated initial copy");
      latest[s]=copy.version;seenCopies[i]=true;seenVersions[copy.version]=true;
    }
  }
  if(std::find(creatorCount.begin(),creatorCount.end(),size_t(0))!=creatorCount.end() ||
     std::find_if(creatorCount.begin(),creatorCount.end(),[](size_t n){return n!=1;})!=creatorCount.end())
    return reject(error,"allocation must own exactly one created copy in whole-task adapter");
  size_t nextTask=0,nextData=0,nextOrder=0;int64_t traffic=0;double duration=0;bool known=true;
  std::map<size_t,double> estimates;
  if(c.computeEstimates.size()!=choice.computeEstimates.size())return reject(error,"compute estimate coverage changed");
  for(size_t i=0;i<choice.computeEstimates.size();++i) {
    const auto &est=choice.computeEstimates[i];double ms;
    std::string lhs,rhs;llvm::raw_string_ostream lo(lhs),ro(rhs);
    lo<<serializeMappingComputeEstimate(est);ro<<serializeMappingComputeEstimate(c.computeEstimates[i]);
    if(lhs!=rhs || est.task>=a.tasks.size() || estimates.count(est.task) ||
       est.resource!=place[est.task].resource || est.implementation!=place[est.task].implementation ||
       !convertComputeEstimateToMs(est,choice.clockMHz,ms,error))return reject(error,"compute estimate contract changed or invalid");
    auto resource=std::find_if(target.resources.begin(),target.resources.end(),[&](const auto &r){return r.id==est.resource;});
    if(resource==target.resources.end() || est.resourceCount>resource->capacity)return reject(error,"compute estimate exceeds resource capacity");
    estimates[est.task]=ms;
  }
  for(size_t i=0;i<c.events.size();++i) {
    const auto &e=c.events[i];
    if(e.task>=a.tasks.size() || e.phase!=e.task ||
       (i && std::find(e.dependencies.begin(),e.dependencies.end(),i-1)==e.dependencies.end()))
      return reject(error,"serial event schedule lacks previous completion");
    for(size_t dep:e.dependencies)if(dep>=i)return reject(error,"event dependence is not acyclic");
    for(size_t id:e.reads) {
      if(id>=c.copies.size() || !seenCopies[id])return reject(error,"read before copy creation");
      const auto &copy=c.copies[id];size_t s=c.regions[c.versions[copy.version].region].storage;
      if(copy.version!=size_t(latest[s]) || copy.availableFrom>i || copy.releasedAfter<=i ||
         (copy.createdBy>=0 && std::find(e.dependencies.begin(),e.dependencies.end(),size_t(copy.createdBy))==e.dependencies.end()))
        return reject(error,"stale version, use after release, or missing copy creation dependency");
      users[id].push_back(i);
    }
    if(e.kind=="transfer") {
      if(e.reads.size()!=1 || e.writes.size()!=1 || !e.dataEdges.empty() || !e.orderEdges.empty())return reject(error,"transfer cannot fabricate source data/order edges");
      size_t id=e.writes[0];if(id>=c.copies.size())return reject(error,"invalid transfer target");
      const auto &src=c.copies[e.reads[0]],&dst=c.copies[id];
      auto s=c.allocations[src.allocation].storage;
      auto link=linkFor(target,a.storages[s],c.allocations[src.allocation].memory,c.allocations[dst.allocation].memory,storageBytes(a.storages[s]));
      if(!link || link->id!=e.link || link->resource!=e.resource || e.memory!=link->target ||
         dst.version!=src.version || dst.createdBy!=int64_t(i) || seenCopies[id])return reject(error,"transfer source/destination/version/link mismatch");
      auto res=std::find_if(target.resources.begin(),target.resources.end(),[&](const auto &r){return r.id==e.resource;});
      if(res==target.resources.end() || res->capacity<1)return reject(error,"missing transfer resource");
      seenCopies[id]=true;
      int64_t b=storageBytes(a.storages[s]);if(traffic>INT64_MAX-b)return reject(error,"traffic overflow");traffic+=b;
      if(std::isfinite(link->bandwidthBytesPerMs) && std::isfinite(link->latencyMs) && link->bandwidthBytesPerMs>0 && link->latencyMs>=0 && !link->provenance.source.empty())duration+=link->latencyMs+b/link->bandwidthBytesPerMs;else known=false;
    } else if(e.kind=="compute") {
      if(e.task!=nextTask++ || seenTasks[e.task] || e.memory!=place[e.task].memory || e.resource!=place[e.task].resource || !e.link.empty())
        return reject(error,"compute instance/placement mismatch");
      seenTasks[e.task]=true;std::set<size_t> reads,writes;std::set<size_t> raw;
      if(e.dataEdges.size()!=e.reads.size())return reject(error,"missing value consumer edge");
      for(size_t k=0;k<e.reads.size();++k) {
        size_t copyId=e.reads[k];const auto &copy=c.copies[copyId];const auto &v=c.versions[copy.version];size_t s=c.regions[v.region].storage;
        if(!reads.insert(s).second || c.allocations[copy.allocation].memory!=e.memory || e.dataEdges[k]!=nextData || nextData>=c.dataEdges.size())return reject(error,"compute reads wrong region/location");
        const auto &edge=c.dataEdges[nextData];
        if(edge.producer!=v.producer || edge.consumer!=i || edge.region!=v.region || edge.version!=copy.version || edge.copy!=copyId ||
           edge.requirement.memory!=e.memory || edge.requirement.layout!=copy.layout || edge.requirement.dtype!=a.storages[s].dtype)
          return reject(error,"consumer edge requirement/version/copy mismatch");
        consumerEdges[copyId].push_back(nextData++);if(v.producer>=0)raw.insert(c.events[v.producer].task);
      }
      for(size_t id:e.writes) {
        if(id>=c.copies.size())return reject(error,"invalid compute result copy");
        const auto &copy=c.copies[id];const auto &v=c.versions[copy.version];size_t s=c.regions[v.region].storage;
        if(seenCopies[id] || seenVersions[copy.version] || !writes.insert(s).second || v.producer!=int64_t(i) ||
           copy.createdBy!=int64_t(i) || c.allocations[copy.allocation].memory!=e.memory)
          return reject(error,"compute produces invalid version/location copy");
        seenCopies[id]=true;seenVersions[copy.version]=true;latest[s]=copy.version;
      }
      if(reads!=effects[e.task].reads || writes!=effects[e.task].writes)return reject(error,"compute effects differ from actual MLIR");
      size_t localOrder=0;
      for(size_t dep:a.tasks[e.task].dependencies) if(!raw.count(dep)) {
        if(localOrder>=e.orderEdges.size() || e.orderEdges[localOrder++]!=nextOrder || nextOrder>=c.orderEdges.size())return reject(error,"missing pure order edge");
        const auto &edge=c.orderEdges[nextOrder++];
        if(edge.consumer!=i || edge.producer>=i || c.events[edge.producer].kind!="compute" || c.events[edge.producer].task!=dep ||
           edge.reason!="source task dependency without value consumption")return reject(error,"invalid pure order edge");
      }
      if(localOrder!=e.orderEdges.size())return reject(error,"fabricated pure order edge");
      if(estimates.count(e.task))duration+=estimates[e.task];else known=false;
    } else return reject(error,"unsupported event kind");
    for(size_t id:e.writes)if(c.copies[id].availableFrom!=i)return reject(error,"copy availability differs from creation event");
  }
  if(nextTask!=a.tasks.size() || nextData!=c.dataEdges.size() || nextOrder!=c.orderEdges.size() ||
     std::find(seenCopies.begin(),seenCopies.end(),false)!=seenCopies.end() ||
     std::find(seenVersions.begin(),seenVersions.end(),false)!=seenVersions.end())return reject(error,"missing/fabricated task, edge, version or copy");
  std::set<std::pair<size_t,std::string>> retainedImports;
  for(size_t i=0;i<c.copies.size();++i) {
    const auto &copy=c.copies[i];const auto &al=c.allocations[copy.allocation];size_t s=al.storage;
    auto expectedLast=users[i].empty()?std::vector<size_t>{}:std::vector<size_t>{users[i].back()};
    size_t begin=copy.createdBy<0?0:c.events[copy.createdBy].phase;
    size_t end=users[i].empty()?begin+1:c.events[users[i].back()].phase+1;
    size_t release=users[i].empty()?copy.availableFrom+1:users[i].back()+1;
    auto out=target.assumptions.outputLocations.find(a.storages[s].id);
    bool final=a.storages[s].role=="output" && out!=target.assumptions.outputLocations.end() && out->second==al.memory && copy.version==size_t(latest[s]);
    if(final){end=a.tasks.size();release=c.events.size();}
    if(copy.createdBy>=0 && c.events[copy.createdBy].kind=="transfer") {
      if(users[i].empty() && !final)return reject(error,"unneeded creation transfer");
      if(choice.policy==ProgramDataPolicy::RetainAcrossUsers && !retainedImports.insert({copy.version,al.memory}).second)
        return reject(error,"retained policy duplicates a valid location copy");
    }
    if(copy.users!=users[i] || copy.lastUsers!=expectedLast || copy.consumerEdges!=consumerEdges[i] ||
       copy.releasedAfter!=release || al.beginPhase!=begin || al.endPhase!=end)
      return reject(error,"copy users/release/allocation lifetime mismatch");
    if(choice.policy==ProgramDataPolicy::PhaseLocal && copy.createdBy>=0 && c.events[copy.createdBy].kind=="transfer") {
      // A transit copy can be consumed only by transfers, so compute-user
      // counting alone would silently permit retention across later tasks.
      // Bind all uses to the importing task, independently of the builder.
      std::set<size_t> consumers{c.events[copy.createdBy].task};
      for(auto u:users[i])consumers.insert(c.events[u].task);
      if(consumers.size()>1)return reject(error,"phase-local imported copy retained across task/transfer consumers");
    }
  }
  // Event intervals are the valid serial schedule's actual allocation lifetime;
  // phase bounds separately remain a conservative human-readable summary.
  std::map<std::string,int64_t> peaks;
  for(const auto &memory:target.memories)for(size_t event=0;event<c.events.size();++event) {
    int64_t live=0;std::set<size_t> counted;
    for(const auto &copy:c.copies)if(copy.availableFrom<=event && event<copy.releasedAfter && counted.insert(copy.allocation).second) {
      const auto &al=c.allocations[copy.allocation];if(al.memory!=memory.id)continue;
      if(memory.capacityBytes<0 || live>memory.capacityBytes || al.bytes>memory.capacityBytes-live)
        return reject(error,"memory capacity exceeded by event-interval copy lifetimes");
      live+=al.bytes;
    }
    peaks[memory.id]=std::max(peaks[memory.id],live);
  }
  if(peaks!=c.peakMemoryBytes)return reject(error,"event memory peak report mismatch");
  for(size_t s=0;s<a.storages.size();++s)if(a.storages[s].role=="output") {
    auto loc=target.assumptions.outputLocations.find(a.storages[s].id);bool found=false;
    if(loc!=target.assumptions.outputLocations.end())for(const auto &copy:c.copies)if(c.allocations[copy.allocation].storage==s &&
       c.allocations[copy.allocation].memory==loc->second && copy.version==size_t(latest[s]) && copy.releasedAfter==c.events.size())found=true;
    if(!found)return reject(error,"final source output missing at declared boundary");
  }
  if(traffic!=c.transferBytes || (known && std::isfinite(duration))!=c.estimatedMakespanMs.has_value() ||
     (c.estimatedMakespanMs && *c.estimatedMakespanMs!=duration))return reject(error,"traffic/full-graph cost differs from checked event charges");
  return true;
}

#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramPartition.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <set>

using namespace orbit::taskflow;
namespace j=llvm::json;
namespace {
bool fail(std::string &e,const std::string &s){e="public joint domain: "+s;return false;}
int64_t bytes(const GenericAnalysis &a,const MappingRegion &r) {
  auto type=mlir::cast<mlir::MemRefType>(a.storages[r.storage].root.getType());
  int64_t n=(type.getElementType().getIntOrFloatBitWidth()+7)/8;
  for(auto d:r.sizes){if(d<=0||n>INT64_MAX/d)return -1;n*=d;}return n;
}
bool parameter(const ProgramMechanismModel &m,const std::string &specific,
    const std::string &fallback,const std::string &unit,double &value,std::string &e) {
  auto p=m.parameters.find(specific);if(p==m.parameters.end())p=m.parameters.find(fallback);
  if(p==m.parameters.end()||!p->second.value||p->second.unit!=unit||
      p->second.source.empty()||(m.mode=="strict"&&p->second.kind=="assumed"))
    return fail(e,"unpriced/unsupported mechanism: "+specific);
  value=*p->second.value;
  return std::isfinite(value)&&value>=0?true:fail(e,"invalid mechanism quantity");
}
bool quantize(double value,int64_t &n,std::string &e) {
  if(!std::isfinite(value)||value<0||value>double(INT64_MAX/128))return fail(e,"unsafe nanosecond range");
  n=int64_t(std::ceil(value));return true;
}
bool model(const j::Value &config,ProgramMechanismModel &out,std::string &e) {
  auto root=config.getAsObject();auto air=root?root->getObject("generic_air"):nullptr;
  auto object=air?air->getObject("cost_model"):nullptr;
  if(!object||object->getString("kind")!="event-mechanism"||
      object->getString("schema")!="orbit.direct-air.cost.v1")
    return fail(e,"explicit public event-mechanism model required");
  out.id=staticFingerprint(j::Value(j::Object(*object)));out.mode=object->getString("mode").value_or("").str();
  out.boundary=object->getString("boundary").value_or("").str();
  if((out.mode!="strict"&&out.mode!="estimate")||out.boundary.empty()||object->getString("model_id").value_or("").empty())return fail(e,"explicit cost mode and boundary required");
  auto params=object->getObject("parameters");if(!params)return fail(e,"mechanism parameters required");
  for(const auto &entry:*params) {
    auto p=entry.second.getAsObject();auto provenance=p?p->getObject("provenance"):nullptr;
    if(!p||!provenance)return fail(e,"parameter provenance required");
    ProgramMechanismParameter value;value.value=p->getNumber("value");
    value.unit=p->getString("unit").value_or("").str();
    value.kind=provenance->getString("kind").value_or("").str();
    value.source=provenance->getString("source").value_or("").str();
    if(value.source.empty()||(value.kind!="assumed"&&value.kind!="architecture"&&value.kind!="calibrated"))return fail(e,"invalid parameter provenance");
    out.parameters[entry.first.str()]=value;
  }
  out.implementations={"scalar"};return true;
}
bool compute(const ProgramJointDomain &d,const ProgramGraph &g,
    const MappingTaskInstance &instance,int64_t &ns,std::string &e) {
  double startup=0,operation=0,access=0;
  if(!parameter(d.model,"compute.scalar.startup","compute.startup","ns",startup,e)||
      !parameter(d.model,"compute.scalar.operation","compute.operation","ns",operation,e)||
      !parameter(d.model,"compute.scalar.memory_access","compute.memory_access","ns",access,e))return false;
  double work=startup;
  const auto &task=g.source.tasks[instance.task];
  task.operation->walk([&](mlir::Operation *op){
    auto name=op->getName().getStringRef();
    bool memory=name=="affine.load"||name=="affine.store"||name=="memref.load"||name=="memref.store";
    bool arith=(name.starts_with("arith.")||name.starts_with("math."))&&name!="arith.constant";
    if(!memory&&!arith)return;
    double multiplicity=1;
    for(const auto &bound:instance.domain)if(task.loops[bound.loop].operation->isProperAncestor(op))
      multiplicity*=double((bound.upper-bound.lower-1)/bound.step+1);
    work+=multiplicity*(memory?access:operation);
  });
  return quantize(work,ns,e);
}
bool sourceVariant(const GenericAnalysis &a,ProgramJointDomain &d,int64_t chunk,
                   ProgramJointVariant &v,std::string &e) {
  ProgramPartitionPlan proof;if(!planProgramPartition(a,chunk,proof,e))return false;
  for(auto axis:proof.storageAxes)if(axis!=0)return fail(e,"current backend domain needs every storage partitioned on axis zero");
  auto rows=proof.extent;int64_t slices=(rows+chunk-1)/chunk;
  auto &g=v.graph;v.chunk=chunk;g.domains=d.target.domains;g.resources=d.target.resources;
  g.memories=d.target.memories;g.links=d.target.links;g.schedule.policy="dependency-order-witness";
  g.targetIdentity=staticFingerprint(serializeMemoryArchitecture(d.target));
  std::map<std::pair<size_t,size_t>,size_t> cells;
  std::vector<int64_t> latest;
  std::vector<size_t> hostAlloc(a.storages.size());
  for(size_t st=0;st<a.storages.size();++st) {
    MappingRegion full{st,std::vector<int64_t>(a.storages[st].shape.size(),0),a.storages[st].shape};
    auto b=bytes(a,full);if(b<1)return fail(e,"invalid source storage size");
    hostAlloc[st]=g.allocations.size();MappingAllocation allocation;
    allocation.storage=st;allocation.memory=d.hostMemory;allocation.key="source-bo-"+std::to_string(st);
    allocation.bytes=b;allocation.beginPhase=0;allocation.endPhase=a.tasks.size()*slices;
    allocation.addressing="region-addressed";g.allocations.push_back(allocation);
    for(int64_t sh=0;sh<slices;++sh) {
      auto r=full;r.offsets[0]=sh*chunk;r.sizes[0]=std::min(chunk,rows-r.offsets[0]);
      cells[{st,size_t(sh)}]=g.regions.size();g.regions.push_back(r);latest.push_back(-1);
      if(a.storages[st].role=="input") {
        auto version=g.versions.size();g.versions.push_back({g.regions.size()-1,-1});latest.back()=version;
        MappingCopy copy;copy.version=version;copy.allocation=hostAlloc[st];copy.createdBy=-1;g.copies.push_back(copy);
      }
    }
  }
  std::map<size_t,size_t> versionCopy;
  for(size_t i=0;i<g.copies.size();++i)versionCopy[g.copies[i].version]=i;
  GenericMappingPlan adapter;adapter.strategy="serial-tiled";
  adapter.memoryBinding.hostMemory=d.hostMemory;
  for(size_t task=0;task<a.tasks.size();++task)adapter.taskImplementations.push_back({task,"scalar",-1,1});
  std::vector<std::vector<size_t>> taskEvents(a.tasks.size());
  for(size_t task=0;task<a.tasks.size();++task)for(int64_t sh=0;sh<slices;++sh) {
    size_t event=g.events.size();GenericTilePhase phase;phase.task=task;
    for(size_t l=0;l<a.tasks[task].loops.size();++l) {
      auto lower=a.tasks[task].loops[l].lower,upper=a.tasks[task].loops[l].upper;
      if(l==0){lower=sh*chunk;upper=std::min(rows,lower+chunk);}
      phase.loops.push_back({l,lower,upper});
    }
    int64_t payload=0;if(!deriveGenericPhaseRegions(a,phase,payload,e))return false;
    MappingEvent compute;compute.kind="compute";compute.task=task;compute.phase=event;
    auto host=std::find_if(d.target.resources.begin(),d.target.resources.end(),[&](const auto &r){return r.kind=="compute"&&memoryAccessible(d.target,d.hostMemory,r.id);});
    if(host==d.target.resources.end())return fail(e,"source reference graph requires a host-accessible compute resource");
    compute.resource=host->id;compute.memory=d.hostMemory;
    std::set<size_t> reads,writes;
    for(const auto &r:phase.regions){auto cell=cells.at({r.storage,size_t(sh)});if(r.read)reads.insert(cell);if(r.write)writes.insert(cell);}
    for(auto cell:reads) {
      if(latest[cell]<0)return fail(e,"source read has no initialized version");
      auto version=size_t(latest[cell]),copy=versionCopy.at(version);
      compute.reads.push_back(copy);auto producer=g.versions[version].producer;
      if(producer>=0)compute.dependencies.push_back(producer);
      compute.dataEdges.push_back(g.dataEdges.size());
      g.dataEdges.push_back({producer,event,cell,version,copy,{d.hostMemory,"row_major",a.storages[g.regions[cell].storage].dtype}});
    }
    for(auto cell:writes) {
      size_t version=g.versions.size();g.versions.push_back({cell,int64_t(event)});latest[cell]=version;
      MappingCopy copy;copy.version=version;copy.allocation=hostAlloc[g.regions[cell].storage];copy.createdBy=event;copy.availableFrom=event;
      compute.writes.push_back(g.copies.size());versionCopy[version]=g.copies.size();g.copies.push_back(copy);
    }
    taskEvents[task].push_back(event);g.events.push_back(compute);adapter.phases.push_back(phase);
  }
  for(size_t task=0;task<a.tasks.size();++task)for(auto parent:a.tasks[task].dependencies) {
    bool raw=false;
    for(const auto &edge:g.dataEdges)if(edge.producer>=0&&g.events[edge.consumer].task==task&&g.events[edge.producer].task==parent)raw=true;
    if(!raw) {
      auto from=taskEvents[parent].back(),to=taskEvents[task].front();g.events[to].dependencies.push_back(from);
      g.events[to].orderEdges.push_back(g.orderEdges.size());g.orderEdges.push_back({from,to,"source pure ordering"});
    }
  }
  for(size_t i=0;i<g.events.size();++i) {
    auto &event=g.events[i];std::sort(event.dependencies.begin(),event.dependencies.end());event.dependencies.erase(std::unique(event.dependencies.begin(),event.dependencies.end()),event.dependencies.end());
    for(auto copy:event.reads)g.copies[copy].users.push_back(i);
    for(auto edge:event.dataEdges)g.copies[g.dataEdges[edge].copy].consumerEdges.push_back(edge);
  }
  for(auto &copy:g.copies) {
    if(!copy.users.empty())copy.lastUsers={copy.users.back()};
    copy.releasedAfter=g.events.size();
  }
  MappingExecutionRegion host;host.id="source-reference";host.placement="host";host.submissionCount=0;host.endEvent=g.events.size();g.executionRegions={host};
  if(!finalizeProgramGraph(a,g,&adapter,e))return false;
  g.regionEvents.clear();g.executionRegions={host};
  if(!checkProgramGraph(g,e))return false;
  v.producers.resize(g.versions.size(),-1);v.readers.resize(g.versions.size());
  for(size_t value=0;value<g.versions.size();++value) {
    v.producers[value]=g.versions[value].producer;
    v.valueBytes.push_back(bytes(a,g.regions[g.versions[value].region]));
  }
  for(size_t i=0;i<g.instances.size();++i) {
    int64_t duration=0;if(!compute(d,g,g.instances[i],duration,e))return false;v.computeNs.push_back(duration);
    for(auto copy:g.events[g.instances[i].event].reads)v.readers[g.copies[copy].version].push_back(i);
  }
  return true;
}
}

bool orbit::taskflow::programJointTransferNs(const ProgramJointDomain &d,size_t vi,
    size_t value,size_t li,int64_t &ns,std::string &e) {
  if(vi>=d.variants.size()||li>=d.target.links.size()||value>=d.variants[vi].valueBytes.size())return fail(e,"invalid transfer reference");
  const auto &v=d.variants[vi];const auto &l=d.target.links[li];auto n=v.valueBytes[value];
  auto storage=v.graph.regions[v.graph.versions[value].region].storage;auto dtype=v.graph.source.storages[storage].dtype;
  if(n<l.minBytes||n>l.maxBytes||l.temporaryBytes||(!l.sourceDtype.empty()&&l.sourceDtype!=dtype)||(!l.targetDtype.empty()&&l.targetDtype!=dtype)||
      (!l.sourceLayout.empty()&&l.sourceLayout!="row_major")||(!l.targetLayout.empty()&&l.targetLayout!="row_major"))return fail(e,"unsupported transfer layout/dtype/size");
  double startup=0,bw=0;
  auto prefix="link."+l.id;
  if(d.model.parameters.count(prefix+".startup")||!l.bandwidthBytesPerMs) {
    if(!parameter(d.model,prefix+".startup","communication.startup","ns",startup,e)||
       !parameter(d.model,prefix+".bandwidth","communication.bandwidth","bytes/ns",bw,e))return false;
  } else {if(l.provenance.source.empty()||(d.model.mode=="strict"&&l.provenance.kind=="assumed"))return fail(e,"unpriced or assumed link in strict mode");startup=l.latencyMs*1e6;bw=l.bandwidthBytesPerMs/1e6;}
  if(bw<=0)return fail(e,"positive explicit link bandwidth required");
  return quantize(startup+double(n)/bw,ns,e);
}

bool orbit::taskflow::programJointGroupFits(const ProgramJointDomain &d,size_t gi,
    size_t pi,int64_t &n,std::string &e) {
  n=0;if(gi>=d.groups.size()||pi>=d.places.size())return fail(e,"invalid group/place");
  const auto &group=d.groups[gi];const auto &place=d.places[pi];
  auto memory=programMemory(d.target,place.binding.memory);
  if(!memory||memory->alignmentBytes<1||!memory->capacityKnown||!memoryAccessible(d.target,memory->id,place.binding.resource))return fail(e,"inaccessible group memory");
  if(group.inputs.size()>size_t(d.maxGroupInputs)||group.outputs.size()>size_t(d.maxGroupOutputs))return fail(e,"group stream limit");
  for(auto value:group.values) {
    if(group.variant>=d.variants.size()||value>=d.variants[group.variant].valueBytes.size())return fail(e,"invalid local version reference");
    auto bytes=d.variants[group.variant].valueBytes[value];if(bytes<1||bytes>INT64_MAX-memory->alignmentBytes)return fail(e,"allocation alignment overflow");auto rem=bytes%memory->alignmentBytes;
    if(rem)bytes+=memory->alignmentBytes-rem;
    if(bytes<1||n>INT64_MAX-bytes)return fail(e,"allocation overflow");n+=bytes;
  }
  if(n>memory->capacityBytes)return fail(e,"static local payload exceeds capacity");
  for(bool input:{true,false})for(auto value:(input?group.inputs:group.outputs)) {
    bool found=false;for(auto link:(input?place.inputLinks:place.outputLinks)){int64_t cost;std::string ignored;if(programJointTransferNs(d,group.variant,value,link,cost,ignored))found=true;}
    if(!found)return fail(e,"missing supported transfer route");
  }
  return true;
}

bool orbit::taskflow::buildProgramJointDomain(const GenericAnalysis &a,const j::Value &config,
    ProgramJointDomain &d,std::string &e) {
  auto start=std::chrono::steady_clock::now();d={};
  if(a.tasks.empty()||a.tasks.size()>512)return fail(e,"source task domain bound exceeded");
  auto root=config.getAsObject();auto options=root?root->getObject("program_joint"):nullptr;
  GenericTargetCapabilities capabilities;GenericMemoryBinding binding;
  if(!getGenericTargetCapabilities(config,capabilities,e)||!configureGenericMemory(config,capabilities.capacityBytes,binding,e)||!model(config,d.model,e))return false;
  d.target=binding.architecture;d.hostMemory=binding.hostMemory;
  if(options)if(auto architecture=options->get("architecture"))if(!parseMemoryArchitecture(*architecture,d.target,e))return false;
  if(options)if(auto host=options->getString("host_memory"))d.hostMemory=host->str();
  auto integer=[&](const char *name,int64_t fallback){return options?options->getInteger(name).value_or(fallback):fallback;};
  auto maxInstances=integer("max_instances",256),maxGroups=integer("max_groups",128),maxGroupTasks=integer("max_group_tasks",4);
  d.maxConfigurations=integer("max_configurations",8);
  d.maxGroupInputs=integer("max_group_inputs",capabilities.maxGroupInputs);d.maxGroupOutputs=integer("max_group_outputs",capabilities.maxGroupOutputs);
  d.maxConfigurationInputs=integer("max_configuration_inputs",2);d.maxConfigurationOutputs=integer("max_configuration_outputs",2);
  if(maxInstances<1||maxInstances>1024||maxGroups<1||maxGroups>4096||maxGroupTasks<1||maxGroupTasks>16||d.maxConfigurations<1||d.maxConfigurations>128||
     d.maxGroupInputs<1||d.maxGroupOutputs<1||d.maxConfigurationInputs<1||d.maxConfigurationOutputs<1)return fail(e,"invalid finite-domain limits");
  double c=0;
  if(!parameter(d.model,"configuration.startup","configuration.startup","ns",c,e)||!quantize(c,d.configurationNs,e)||
     !parameter(d.model,"host.submit","host.submit","ns",c,e)||!quantize(c,d.submitNs,e)||
     !parameter(d.model,"host.complete","host.complete","ns",c,e)||!quantize(c,d.completeNs,e))return false;
  // First supported realization: exact identity-grid, unique writers, no in-place updates.
  auto shape=a.storages.front().shape;
  if(shape.empty()||shape.size()>2)return fail(e,"current executable domain requires one/two-dimensional pointwise source");
  std::vector<int64_t> writers(a.storages.size(),-1);
  for(size_t t=0;t<a.tasks.size();++t) {
    if(a.tasks[t].loops.size()!=shape.size())return fail(e,"source loop rank differs from pointwise domain");
    for(size_t l=0;l<shape.size();++l)if(a.tasks[t].loops[l].lower!=0||a.tasks[t].loops[l].step!=1||a.tasks[t].loops[l].upper!=shape[l])return fail(e,"source tasks do not share the independent grid");
    std::set<size_t> written;
    for(const auto &access:a.tasks[t].accesses) {
      if(!access.exactRegion||a.storages[access.storage].shape!=shape||access.read&&access.write)return fail(e,"nonexact, differently shaped or in-place source access");
      if(access.write)written.insert(access.storage);
    }
    for(auto storage:written){if(writers[storage]>=0)return fail(e,"multiple source writers are outside this realization");writers[storage]=t;}
  }
  for(size_t p=0;p<d.target.resources.size();++p) {
    const auto &r=d.target.resources[p];if(r.kind!="compute")continue;
    for(const auto &m:d.target.memories) {
      if(m.id==d.hostMemory||!memoryAccessible(d.target,m.id,r.id))continue;
      ProgramJointPlace place;place.binding={0,r.id,m.id,"scalar"};
      for(size_t li=0;li<d.target.links.size();++li) {
        const auto &link=d.target.links[li];if(link.source==d.hostMemory&&link.target==m.id)place.inputLinks.push_back(li);
        if(link.source==m.id&&link.target==d.hostMemory)place.outputLinks.push_back(li);
      }
      if(!place.inputLinks.empty()&&!place.outputLinks.empty())d.places.push_back(place);
    }
  }
  if(d.places.empty()||d.places.size()>16)return fail(e,"one-hop host/local placement domain missing or too large");
  std::set<int64_t> chunks;
  if(options)if(auto values=options->getArray("chunks"))for(const auto &value:*values){auto n=value.getAsInteger();if(!n||*n<1||*n>shape[0])return fail(e,"invalid independent row chunk");chunks.insert(*n);}
  if(chunks.empty()){chunks.insert(shape[0]);if(d.places.size()>1)chunks.insert((shape[0]+d.places.size()-1)/d.places.size());}
  if(chunks.size()>8)return fail(e,"tile-variant limit exceeded");
  for(auto chunk:chunks) {
    if(a.tasks.size()*size_t((shape[0]+chunk-1)/chunk)>size_t(maxInstances)){d.complete=false;continue;}
    ProgramJointVariant variant;if(!sourceVariant(a,d,chunk,variant,e))return false;
    auto vi=d.variants.size();d.variants.push_back(std::move(variant));const auto &v=d.variants.back();
    auto count=v.graph.instances.size();std::vector<std::set<size_t>> adjacency(count);
    for(size_t value=0;value<v.readers.size();++value) {
      auto users=v.readers[value];if(v.producers[value]>=0)users.push_back(v.producers[value]);
      for(auto x:users)for(auto y:users)if(x!=y)adjacency[x].insert(y);
    }
    auto slices=(shape[0]+chunk-1)/chunk;
    std::set<std::vector<size_t>> seen;std::deque<std::vector<size_t>> queue;
    auto addGroup=[&](std::vector<size_t> members) {
      std::sort(members.begin(),members.end());if(!seen.insert(members).second)return;
      if(members.size()>1){std::set<size_t> connected{members.front()};bool grew=true;while(grew){grew=false;for(auto member:members)if(!connected.count(member))for(auto parent:connected)if(adjacency[member].count(parent)){connected.insert(member);grew=true;break;}}if(connected.size()!=members.size())return;}
      auto slice=members.front()%slices;
      for(auto member:members)if(member%slices!=slice)return;
      for(const auto &edge:v.graph.orderEdges)if(std::binary_search(members.begin(),members.end(),edge.producer)&&std::binary_search(members.begin(),members.end(),edge.consumer))return;
      ProgramJointGroup group;group.variant=vi;group.slice=slice;group.instances=members;
      std::set<size_t> values,inputs,outputs;
      for(auto member:members) {
        const auto &event=v.graph.events[v.graph.instances[member].event];
        for(auto copy:event.reads){auto value=v.graph.copies[copy].version;values.insert(value);auto producer=v.producers[value];if(producer<0||!std::binary_search(members.begin(),members.end(),size_t(producer)))inputs.insert(value);}
        for(auto copy:event.writes){auto value=v.graph.copies[copy].version;values.insert(value);auto st=v.graph.regions[v.graph.versions[value].region].storage;
          bool external=a.storages[st].role=="output";for(auto reader:v.readers[value])external|=!std::binary_search(members.begin(),members.end(),reader);if(external)outputs.insert(value);}
      }
      group.values={values.begin(),values.end()};group.inputs={inputs.begin(),inputs.end()};group.outputs={outputs.begin(),outputs.end()};
      if(d.groups.size()>=size_t(maxGroups)){d.complete=false;return;}
      d.groups.push_back(group);if(members.size()<size_t(maxGroupTasks))queue.push_back(members);
    };
    for(size_t i=0;i<count;++i)addGroup({i});
    // A whole connected row group is one local option, not a whole-program plan.
    for(int64_t sh=0;sh<slices;++sh){std::vector<size_t> all;for(size_t t=0;t<a.tasks.size();++t)all.push_back(t*slices+sh);if(all.size()>1&&all.size()<=size_t(maxGroupTasks))addGroup(all);}
    while(!queue.empty()&&d.groups.size()<size_t(maxGroups)) {
      auto current=queue.front();queue.pop_front();
      for(auto member:current)for(auto next:adjacency[member])if(!std::binary_search(current.begin(),current.end(),next)){auto grown=current;grown.push_back(next);addGroup(grown);}
    }
    if(!queue.empty())d.complete=false;
  }
  if(d.variants.empty()||d.groups.empty())return fail(e,"finite source domain is empty");
  d.identity=staticFingerprint(serializeProgramJointDomain(d));
  d.generationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return true;
}

llvm::json::Value orbit::taskflow::serializeProgramJointDomain(const ProgramJointDomain &d) {
  j::Array variants,groups,places;
  for(const auto &v:d.variants){j::Array compute,values;for(auto n:v.computeNs)compute.push_back(n);for(auto n:v.valueBytes)values.push_back(n);
    variants.push_back(j::Object{{"chunk",v.chunk},{"source_graph",serializeProgramGraph(v.graph)},{"compute_ns",std::move(compute)},{"version_bytes",std::move(values)}});}
  auto list=[](const auto &items){j::Array a;for(auto x:items)a.push_back(int64_t(x));return a;};
  for(const auto &g:d.groups)groups.push_back(j::Object{{"variant",int64_t(g.variant)},{"slice",int64_t(g.slice)},{"instances",list(g.instances)},{"values",list(g.values)},{"inputs",list(g.inputs)},{"outputs",list(g.outputs)}});
  for(const auto &p:d.places)places.push_back(j::Object{{"resource",p.binding.resource},{"memory",p.binding.memory},{"input_links",list(p.inputLinks)},{"output_links",list(p.outputLinks)}});
  return j::Object{{"schema","orbit.program-joint.domain.v23"},{"architecture",serializeMemoryArchitecture(d.target)},
    {"variants",std::move(variants)},{"local_groups",std::move(groups)},{"placements",std::move(places)},
    {"model_id",d.model.id},{"host_memory",d.hostMemory},{"configuration_ns",d.configurationNs},{"submit_ns",d.submitNs},{"complete_ns",d.completeNs},
    {"max_configurations",d.maxConfigurations},{"max_group_inputs",d.maxGroupInputs},{"max_group_outputs",d.maxGroupOutputs},
    {"max_configuration_inputs",d.maxConfigurationInputs},{"max_configuration_outputs",d.maxConfigurationOutputs},{"domain_complete",d.complete}};
}

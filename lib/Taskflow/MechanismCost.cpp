#include "Taskflow/MechanismCost.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

using namespace orbit::taskflow;
namespace {
using O=llvm::json::Object;using A=llvm::json::Array;using J=llvm::json::Value;
bool fail(std::string &error,const std::string &why){error="mechanism cost: "+why;return false;}
A ids(const std::vector<size_t> &values){A result;for(auto x:values)result.push_back(int64_t(x));return result;}
bool add(int64_t &total,int64_t value){if(value<0 || total>INT64_MAX-value)return false;total+=value;return true;}
bool digest(const std::string &text){return text.size()==64 && std::all_of(text.begin(),text.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
int64_t bytes(const GenericStorage &storage){
 auto type=mlir::dyn_cast<mlir::MemRefType>(storage.root.getType());if(!type)return -1;
 int64_t n=(type.getElementType().getIntOrFloatBitWidth()+7)/8;
 for(auto extent:storage.shape){if(extent<=0 || n>INT64_MAX/extent)return -1;n*=extent;}return n;
}
J optional(std::optional<double> value){return value?J(*value):J(nullptr);}
bool chargeable(const std::string &kind){return kind=="host.input_reset" || kind=="host.output_copy" || kind=="bo.sync_to_device" || kind=="bo.sync_from_device" || kind=="npu.submit";}
using Key=std::tuple<std::string,std::string,std::string,std::string,std::string,std::string>;
Key key(const MechanismParameter &p){return {p.mechanism,p.path,p.state,p.resource,p.boundary,p.runtimeIdentity};}
bool parameters(const std::vector<MechanismParameter> &values,std::string &error){
 for(size_t i=0;i<values.size();++i){const auto &p=values[i];
  if(!chargeable(p.mechanism))return fail(error,"opaque compute/wait composites and descriptive movement cannot receive independent fixed-byte charges");
  if(p.path.empty() || p.state.empty() || p.resource.empty() || p.boundary.empty() || p.runtimeIdentity.empty() || p.runtimeIdentity=="unbound" ||
     p.provenance.empty() || p.measurementBoundary!=p.boundary || !digest(p.rawManifestSHA256) || p.minBytes<0 || p.maxBytes<p.minBytes ||
     !std::isfinite(p.alphaMs) || !std::isfinite(p.betaMsPerByte) || p.alphaMs<0 || p.betaMsPerByte<0 || (p.alphaMs==0 && p.betaMsPerByte==0))
   return fail(error,"invalid mechanism measurement identity, boundary, byte domain or coefficient");
  for(size_t j=0;j<i;++j)if(key(p)==key(values[j]) && std::max(p.minBytes,values[j].minBytes)<=std::min(p.maxBytes,values[j].maxBytes))
   return fail(error,"duplicate overlapping mechanism parameter domain");
 }
 return true;
}
J parameterJSON(const MechanismParameter &p){return O{{"mechanism",p.mechanism},{"path",p.path},{"state",p.state},{"resource",p.resource},{"boundary",p.boundary},{"runtime_identity",p.runtimeIdentity},
 {"alpha_ms",p.alphaMs},{"beta_ms_per_byte",p.betaMsPerByte},{"min_bytes",p.minBytes},{"max_bytes",p.maxBytes},{"provenance",p.provenance},{"measurement_boundary",p.measurementBoundary},{"raw_manifest_sha256",p.rawManifestSHA256}};}
} // namespace

bool orbit::taskflow::collectMechanismFacts(const ProgramGraph &graph,const MechanismRuntimeContract &runtime,MechanismFacts &facts,std::string &error){
 facts={};
 if(!checkProgramGraph(graph,error))return false;
 if(graph.regionEvents.empty() || runtime.identity.empty() || runtime.state!="prepared-hot-bound-bo" ||
    !runtime.boundBuffers || !runtime.inputReset || !runtime.finalOutputCopy || !runtime.outputPoisoned ||
    runtime.syncToDevice!="all-abi" || runtime.syncFromDevice!="outputs")
  return fail(error,"checked heterogeneous regions and explicit prepared bound-BO runtime contract required");
 for(const auto &alias:runtime.resourceAliases)
  if(alias.first!="host.thread" && alias.first!="host.memory")
   return fail(error,"unsupported runtime resource alias");
 facts.graphId=programGraphIdentity(graph);facts.bodyFingerprint=graph.bodyFingerprint;facts.runtime=runtime;
 std::set<size_t> used,mapped;std::vector<size_t> eventOwner(graph.events.size(),SIZE_MAX);
 std::set<std::pair<size_t,int64_t>> syncedEpochs;
 for(size_t region=0;region<graph.executionRegions.size();++region){const auto &r=graph.executionRegions[region];
  for(auto event:r.memberEvents)eventOwner[event]=region;
  for(const auto *list:{&r.inputCopies,&r.outputCopies})for(auto copy:*list){
   const auto &allocation=graph.allocations[graph.copies[copy].allocation];
   if(allocation.memory!=r.endpointMemory)return fail(error,"ABI copy is not the declared host endpoint");
   used.insert(allocation.storage);if(r.placement=="npu")mapped.insert(allocation.storage);
  }
 }
 auto append=[&](MechanismItem item){item.id=facts.items.size();facts.items.push_back(std::move(item));return facts.items.size()-1;};
 MechanismItem outer;outer.mechanism="host.call_overhead";outer.path="outer-dispatch-bookkeeping";
 outer.state=runtime.state;outer.resource="host.thread";outer.boundary="outside-recorded-API-calls";
 outer.composite=true;append(outer);
 auto hostcopy=[&](size_t storage,bool input){MechanismItem item;item.storage=storage;
  item.mechanism=input?"host.input_reset":"host.output_copy";
  item.path=input?(mapped.count(storage)?"numpy-to-mapped-bo":"numpy-to-host"):(mapped.count(storage)?"mapped-bo-to-numpy":"host-to-numpy");
  item.state="prepared-hot";item.resource="host.memory";item.boundary=input?"numpy.copyto-call":"executor.final-output-acquisition";
  item.bytes=bytes(graph.source.storages[storage]);return item;
 };
 if(runtime.inputReset)for(auto storage:used)if(graph.source.storages[storage].role=="input"){
  auto item=hostcopy(storage,true);if(item.bytes<0)return fail(error,"host input byte count overflow");
  for(size_t copy=0;copy<graph.copies.size();++copy)if(graph.copies[copy].createdBy<0 &&
      graph.allocations[graph.copies[copy].allocation].storage==storage)item.copies.push_back(copy);
  append(item);
 }
 for(size_t region=0;region<graph.executionRegions.size();++region){const auto &r=graph.executionRegions[region];
  std::vector<size_t> lifecycle;
  for(size_t i=0;i<graph.regionEvents.size();++i)if(graph.regionEvents[i].region==region)lifecycle.push_back(i);
  auto item=[&](const std::string &mechanism,const std::string &path,const std::string &resource,const std::string &boundary){
   MechanismItem value;value.region=region;value.mechanism=mechanism;value.path=path;value.resource=resource;value.boundary=boundary;
   value.state=runtime.state;value.sourceTasks=r.sourceTasks;value.publicEvents=r.memberEvents;
   std::string publicKind=mechanism=="bo.sync_to_device"?"visible-to-device":
       mechanism=="bo.sync_from_device"?"visible-to-host":mechanism=="npu.submit"?"submit":mechanism=="npu.wait"?"complete":"";
   for(auto id:lifecycle)if(mechanism=="cpu.call" || graph.regionEvents[id].kind==publicKind)value.regionEvents.push_back(id);
   return value;
  };
  if(r.placement=="npu"){
   std::set<size_t> abi(r.inputCopies.begin(),r.inputCopies.end());abi.insert(r.outputCopies.begin(),r.outputCopies.end());
   for(auto copy:abi){const auto &allocation=graph.allocations[graph.copies[copy].allocation];
    auto sync=item("bo.sync_to_device","pyxrt.bo.sync.TO_DEVICE","NPU.hostcall","pyxrt.bo.sync-call");sync.storage=allocation.storage;sync.copies={copy};sync.bytes=allocation.bytes;
    sync.state="host-written";
    bool output=std::find(r.outputCopies.begin(),r.outputCopies.end(),copy)!=r.outputCopies.end();sync.runtimeOnly=output;
    sync.publicEvents.clear();
    for(auto event:r.memberEvents)if(graph.events[event].kind=="transfer" &&
        std::find(graph.events[event].reads.begin(),graph.events[event].reads.end(),copy)!=graph.events[event].reads.end())sync.publicEvents.push_back(event);
    if(output)sync.regionEvents.clear(); // buffer sync, not a logical future-version read
    else sync.regionEvents.erase(std::remove_if(sync.regionEvents.begin(),sync.regionEvents.end(),[&](size_t id){
      const auto &copies=graph.regionEvents[id].copies;return std::find(copies.begin(),copies.end(),copy)==copies.end();
    }),sync.regionEvents.end());
    int64_t writeEpoch=0; // input reset / output poison precedes the call
    if(!output){auto producer=graph.versions[graph.copies[copy].version].producer;
     if(producer>=0)writeEpoch=producer+1;
     if(producer>=0 && graph.executionRegions[eventOwner[size_t(producer)]].placement=="npu")sync.state="device-completed-host-visible";
    }
    if(!syncedEpochs.insert({allocation.storage,writeEpoch}).second)sync.state="host-visible-after-sync";
    append(sync);
   }
   append(item("npu.submit","pyxrt.run.start","NPU.submit","pyxrt.run.start-call"));
   auto wait=item("npu.wait","installed-runner.wait","NPU.submit","installed-wait-call");wait.composite=true;
   size_t waitId=append(wait);
   for(auto event:r.memberEvents)if(graph.events[event].kind=="transfer"){
    const auto &transfer=graph.events[event];auto movement=item("native.dma","checked-host-local-DMA",transfer.resource,"inside-npu-wait-composite");
    movement.descriptive=true;movement.coveredBy=waitId;movement.bytes=transfer.transferBytes;movement.publicEvents={event};
    movement.copies=transfer.reads;movement.copies.insert(movement.copies.end(),transfer.writes.begin(),transfer.writes.end());
    movement.storage=transfer.transferRegion->storage;append(movement);
   }
   for(auto copy:r.outputCopies){const auto &allocation=graph.allocations[graph.copies[copy].allocation];
    auto sync=item("bo.sync_from_device","pyxrt.bo.sync.FROM_DEVICE","NPU.hostcall","pyxrt.bo.sync-call");
    sync.storage=allocation.storage;sync.copies={copy};sync.bytes=allocation.bytes;sync.state="device-completed";
    sync.publicEvents.clear();for(auto event:r.memberEvents)if(graph.events[event].kind=="transfer" &&
        std::find(graph.events[event].writes.begin(),graph.events[event].writes.end(),copy)!=graph.events[event].writes.end())sync.publicEvents.push_back(event);
    sync.regionEvents.erase(std::remove_if(sync.regionEvents.begin(),sync.regionEvents.end(),[&](size_t id){
      const auto &copies=graph.regionEvents[id].copies;return std::find(copies.begin(),copies.end(),copy)==copies.end();
    }),sync.regionEvents.end());
    append(sync);
   }
  }else if(r.placement=="cpu"){
   auto call=item("cpu.call","compiled-ciface-region","CPU.compute","compiled-region-call-with-descriptor-setup");call.composite=true;size_t owner=append(call);
   for(bool read:{true,false}){
    int64_t total=0;
    for(auto task:r.sourceTasks)for(const auto &access:graph.source.tasks[task].accesses)if(read?access.read:access.write){
     int64_t count=1;for(const auto &loop:graph.source.tasks[task].loops)if(loop.operation->isProperAncestor(access.operation)){
      int64_t iterations=(loop.upper-loop.lower-1)/loop.step+1;
      if(iterations<=0 || count>INT64_MAX/iterations)return fail(error,"CPU logical access count overflow");count*=iterations;
     }
     auto type=mlir::cast<mlir::MemRefType>(graph.source.storages[access.storage].root.getType());int64_t width=(type.getElementType().getIntOrFloatBitWidth()+7)/8;
     if(count>INT64_MAX/width || !add(total,count*width))return fail(error,"CPU logical access byte overflow");
    }
    auto movement=item(read?"cpu.logical_read":"cpu.logical_write","actual-source-load-store","CPU.compute","inside-cpu-call-composite");
    movement.bytes=total;movement.descriptive=true;movement.coveredBy=owner;append(movement);
   }
  }else return fail(error,"unsupported region execution mechanism");
  auto overhead=item("runtime.region_overhead","region-dispatch-bookkeeping","host.thread","outside-recorded-region-API-calls");
  overhead.composite=true;append(overhead);
 }
 if(runtime.finalOutputCopy){
  MechanismItem acquisition;bool have=false,allMapped=true,allHost=true;
  for(auto storage:used)if(graph.source.storages[storage].role=="output"){
   auto item=hostcopy(storage,false);if(item.bytes<0)return fail(error,"host output byte count overflow");
   if(!have){acquisition=item;acquisition.bytes=0;have=true;}else acquisition.storage=-1;
   if(!add(acquisition.bytes,item.bytes))return fail(error,"final output acquisition byte sum overflow");
   allMapped &= mapped.count(storage)!=0;allHost &= mapped.count(storage)==0;
   for(size_t copy=0;copy<graph.copies.size();++copy) {
    const auto &value=graph.copies[copy];const auto &allocation=graph.allocations[value.allocation];
    if(allocation.storage==storage && graph.outputLocations.at(storage)==allocation.memory && value.releasedAfter==graph.events.size())acquisition.copies.push_back(copy);
   }
  }
  if(have){acquisition.path=allMapped?"mapped-bo-to-numpy":allHost?"host-to-numpy":"mixed-host-output-acquisition";append(acquisition);}
 }
 for(const auto &item:facts.items) {
  if(facts.runtimeResourceBindings.count(item.resource))continue;
  auto alias=runtime.resourceAliases.find(item.resource);
  std::string target=alias==runtime.resourceAliases.end()?item.resource:alias->second;
  auto resource=std::find_if(graph.resources.begin(),graph.resources.end(),[&](const auto &value){return value.id==target;});
  if(resource==graph.resources.end() || resource->capacity<=0)
   return fail(error,"runtime mechanism resource has no declared positive-capacity target binding: "+item.resource);
  bool hostAlias=item.resource=="host.thread" || item.resource=="host.memory";
  if(hostAlias && (alias==runtime.resourceAliases.end() || resource->kind!="compute" || resource->device!="host"))
   return fail(error,"host runtime alias must bind a declared host compute worker");
  MechanismResourceBinding binding;binding.targetResource=target;binding.targetCapacity=resource->capacity;
  binding.semantics=hostAlias?
      "serialized host worker occupancy; not an independent memory port, DRAM bandwidth, or extra concurrent resource":
      "same declared public target resource; listed mechanisms execute serially";
  facts.runtimeResourceBindings[item.resource]=std::move(binding);
 }
 auto view=serializeMechanismFacts(facts);facts.identity=staticFingerprint(view);return true;
}

bool orbit::taskflow::parseMechanismParameters(const J &value,std::vector<MechanismParameter> &result,std::string &error){
 result.clear();auto object=value.getAsObject();auto rows=object?object->getArray("parameters"):nullptr;
 if(!object || object->getString("schema")!="orbit.mechanism-cost-parameters.v13" || !rows)return fail(error,"invalid mechanism parameter schema");
 for(const auto &row:*rows){auto item=row.getAsObject();if(!item)return fail(error,"mechanism parameter must be an object");MechanismParameter p;
  for(auto entry:{std::make_pair("mechanism",&p.mechanism),{"path",&p.path},{"state",&p.state},{"resource",&p.resource},{"boundary",&p.boundary},{"runtime_identity",&p.runtimeIdentity},{"provenance",&p.provenance},{"measurement_boundary",&p.measurementBoundary},{"raw_manifest_sha256",&p.rawManifestSHA256}}){
   auto text=item->getString(entry.first);if(!text)return fail(error,"missing string parameter field: "+std::string(entry.first));*entry.second=text->str();
  }
  auto minimum=item->getInteger("min_bytes"),maximum=item->getInteger("max_bytes");auto alpha=item->getNumber("alpha_ms"),beta=item->getNumber("beta_ms_per_byte");
  if(!minimum || !maximum || !alpha || !beta)return fail(error,"explicit integer byte domain and numeric coefficients required");
  p.minBytes=*minimum;p.maxBytes=*maximum;p.alphaMs=*alpha;p.betaMsPerByte=*beta;result.push_back(std::move(p));
 }
 return parameters(result,error);
}

bool orbit::taskflow::evaluateMechanismCost(const MechanismFacts &facts,const std::vector<MechanismParameter> &values,MechanismCostEvaluation &result,std::string &error){
 result={};if(!parameters(values,error))return false;
 auto copy=facts;copy.identity.clear();
 if(facts.identity.empty() || facts.identity!=staticFingerprint(serializeMechanismFacts(copy)))return fail(error,"mechanism facts identity changed");
 result.factsIdentity=facts.identity;A rows;for(const auto &p:values)rows.push_back(parameterJSON(p));result.parameterIdentity=staticFingerprint(J(std::move(rows)));
 bool complete=true,prefixKnown=true;double end=0;
 for(size_t index=0;index<facts.items.size();++index){const auto &item=facts.items[index];MechanismPricedItem price;price.item=index;
  auto binding=facts.runtimeResourceBindings.find(item.resource);
  if(item.id!=index || item.bytes<0 || binding==facts.runtimeResourceBindings.end() ||
     binding->second.targetResource.empty() || binding->second.targetCapacity<=0)
   return fail(error,"invalid mechanism identity, payload, or public target resource binding");
  if(item.descriptive){
   if(item.coveredBy<0 || size_t(item.coveredBy)>=index || !facts.items[size_t(item.coveredBy)].composite)
    return fail(error,"descriptive movement lacks its earlier composite owner");
   price.status="included-in-composite";price.reason="descriptive payload is not an additional elapsed-time charge";result.items.push_back(price);continue;
  }
  const MechanismParameter *parameter=nullptr;
  if(!item.composite && facts.runtime.identity!="unbound")for(const auto &p:values)
   if(p.mechanism==item.mechanism && p.path==item.path && p.state==item.state && p.resource==item.resource && p.boundary==item.boundary &&
      p.runtimeIdentity==facts.runtime.identity && item.bytes>=p.minBytes && item.bytes<=p.maxBytes){if(parameter)return fail(error,"multiple matching mechanism parameters");parameter=&p;}
  if(parameter){double duration=parameter->alphaMs+parameter->betaMsPerByte*double(item.bytes);
   if(!std::isfinite(duration) || duration<=0 || !std::isfinite(result.knownSubtotalMs+duration))return fail(error,"mechanism cost overflow or nonpositive duration");
   price.durationMs=duration;price.status="calibrated-api-model";result.knownSubtotalMs+=duration;
   if(prefixKnown){price.startMs=end;end+=duration;price.endMs=end;}
  }else {complete=false;prefixKnown=false;price.status="unknown";price.reason=item.composite?"opaque compute/memory/runtime composite has no separable mechanism calibration":"no exact runtime/path/state/resource/boundary and byte-domain calibration";}
  result.items.push_back(std::move(price));
 }
 if(complete)result.totalMs=result.knownSubtotalMs;return true;
}

J orbit::taskflow::serializeMechanismFacts(const MechanismFacts &facts){
 O aliases,bindings;
 for(const auto &entry:facts.runtime.resourceAliases)aliases[entry.first]=entry.second;
 for(const auto &entry:facts.runtimeResourceBindings)bindings[entry.first]=O{
  {"target_resource",entry.second.targetResource},{"target_capacity",entry.second.targetCapacity},
  {"semantics",entry.second.semantics}};
 A rows;for(const auto &item:facts.items)rows.push_back(O{{"id",int64_t(item.id)},{"region",item.region},{"storage",item.storage},{"mechanism",item.mechanism},{"path",item.path},{"state",item.state},{"resource",item.resource},{"boundary",item.boundary},{"bytes",item.bytes},
  {"copies",ids(item.copies)},{"source_tasks",ids(item.sourceTasks)},{"public_events",ids(item.publicEvents)},{"region_events",ids(item.regionEvents)},
  {"composite",item.composite},{"descriptive_only",item.descriptive},{"runtime_only",item.runtimeOnly},{"covered_by",item.coveredBy}});
 O result{{"schema","orbit.mechanism-facts.v13"},{"program_graph_id",facts.graphId},{"source_fingerprint",facts.bodyFingerprint},
  {"runtime",O{{"identity",facts.runtime.identity},{"state",facts.runtime.state},{"bound_buffers",facts.runtime.boundBuffers},{"sync_to_device",facts.runtime.syncToDevice},{"sync_from_device",facts.runtime.syncFromDevice},{"input_reset",facts.runtime.inputReset},{"final_output_copy",facts.runtime.finalOutputCopy},{"output_poisoned",facts.runtime.outputPoisoned},{"resource_aliases",std::move(aliases)}}},
  {"runtime_resource_bindings",std::move(bindings)},
  {"items",std::move(rows)},{"byte_scope","logical API-size arguments; BO sync is not a new logical copy or measured physical DDR transfer"},
  {"schedule_scope","serial listed API/composite mechanisms; no unknown duration is replaced by zero"}};
 if(!facts.identity.empty())result["id"]=facts.identity;return result;
}
J orbit::taskflow::serializeMechanismCost(const MechanismCostEvaluation &cost){
 A rows;for(const auto &item:cost.items)rows.push_back(O{{"item",int64_t(item.item)},{"status",item.status},{"reason",item.reason},{"duration_ms",optional(item.durationMs)},{"start_ms",optional(item.startMs)},{"end_ms",optional(item.endMs)}});
 return O{{"schema","orbit.mechanism-cost.v13"},{"facts_id",cost.factsIdentity},{"parameter_id",cost.parameterIdentity},{"known_subtotal_ms",cost.knownSubtotalMs},{"complete_ms",optional(cost.totalMs)},
  {"items",std::move(rows)},{"status",cost.totalMs?"complete-api-model":"incomplete-unknown-composites-or-domain"},{"scope","analytical API mechanisms only; opaque compute/DMA/runtime cannot be split or double charged; no work-element regression"}};
}

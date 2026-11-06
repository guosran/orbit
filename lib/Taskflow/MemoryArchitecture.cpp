#include "Taskflow/GenericMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <queue>
#include <set>
using namespace orbit::taskflow;
namespace {
using O=llvm::json::Object;using A=llvm::json::Array;using V=llvm::json::Value;
bool fail(std::string &e,const std::string &s){e="memory architecture: "+s;return false;}
const StaticMemory *memory(const StaticConfig &c,llvm::StringRef id){for(const auto &m:c.memories)if(m.id==id)return &m;return nullptr;}
const StaticResource *resource(const StaticConfig &c,llvm::StringRef id){for(const auto &r:c.resources)if(r.id==id)return &r;return nullptr;}
A strings(const std::vector<std::string> &v){A a;for(const auto &x:v)a.push_back(x);return a;}
bool textFields(const O &o,std::initializer_list<llvm::StringRef> keys,std::string &e){
  for(auto key:keys)if(auto v=o.get(key))if(!v->getAsString())return fail(e,"expected string "+key.str());return true;
}
bool readStrings(const O &o,llvm::StringRef key,std::vector<std::string> &out,std::string &e){
  auto value=o.get(key);if(!value)return true;auto a=value->getAsArray();if(!a)return fail(e,"expected array "+key.str());
  for(const auto &v:*a){auto s=v.getAsString();if(!s || s->empty())return fail(e,"invalid string in "+key.str());out.push_back(s->str());}return true;
}
std::string string(const O &o,llvm::StringRef key){return o.getString(key).value_or("").str();}
StaticConfig npuArchitecture(int64_t capacity){
  StaticConfig c;c.domains={{"system","","system"},{"npu","system","device"},{"npu.tile(0,2)","npu","tile",{{"column",0},{"row",2}}}};
  c.resources={{"npu.compute","compute","npu",1,"npu.tile(0,2)"},{"npu.dma","transfer","npu",1,"npu"},{"host-submission","submission","host",1,"system"},{"host.worker","compute","host",1,"system"}};
  c.memories={{"system_memory_bo",INT64_MAX,{"host"},"system","system-bo",{"host.worker"}},
              {"aie_l1",capacity,{"npu"},"npu.tile(0,2)","local-sram",{"npu.compute"}}};
  for(bool input:{true,false}){StaticTransfer t;t.id=input?"host_to_l1":"l1_to_host";
    t.source=input?"system_memory_bo":"aie_l1";t.target=input?"aie_l1":"system_memory_bo";t.resource="npu.dma";t.domain="npu";
    t.provenance={"target_contract","serial explicit DMA; latency/bandwidth uncalibrated"};c.links.push_back(t);}
  return c;
}
}
bool orbit::taskflow::memoryAccessible(const StaticConfig &c,llvm::StringRef mid,llvm::StringRef rid){
  auto m=memory(c,mid);auto r=resource(c,rid);if(!m || !r || r->kind!="compute")return false;
  if(!m->accessibleResources.empty())return std::find(m->accessibleResources.begin(),m->accessibleResources.end(),rid.str())!=m->accessibleResources.end();
  return std::find(m->accessibleDevices.begin(),m->accessibleDevices.end(),r->device)!=m->accessibleDevices.end();
}
bool orbit::taskflow::validateMemoryArchitecture(const StaticConfig &c,std::string &e){
  std::map<std::string,std::string> domains;std::set<std::string> ids;
  for(const auto &d:c.domains){for(const auto &axis:d.coordinates)if(axis.first.empty())return fail(e,"empty coordinate axis");if(d.id.empty() || d.kind.empty() || !domains.emplace(d.id,d.parent).second)return fail(e,"invalid/duplicate domain");}
  for(const auto &d:c.domains){std::set<std::string> seen;std::string id=d.id;while(!id.empty()){
    if(!domains.count(id) || !seen.insert(id).second)return fail(e,"unknown parent or cyclic domain ownership");id=domains.at(id);}}
  auto domain=[&](const std::string &d){return d.empty() || domains.count(d);};
  for(const auto &r:c.resources)if(r.id.empty() || !ids.insert(r.id).second || r.capacity<=0 || (r.kind=="compute" && r.device.empty()) || !domain(r.domain) ||
      (r.kind!="compute" && r.kind!="communication" && r.kind!="transfer" && r.kind!="submission" && r.kind!="shared"))return fail(e,"invalid resource/domain/capacity");
  std::map<std::string,int> sharedState;
  std::function<bool(const StaticResource &)> sharedVisit=[&](const auto &r) {
    if(sharedState[r.id]==1)return false;if(sharedState[r.id]==2)return true;sharedState[r.id]=1;
    for(const auto &entry:r.sharedResources){auto target=resource(c,entry.first);if(!target || entry.second<=0 || entry.second>target->capacity || !sharedVisit(*target))return false;}
    sharedState[r.id]=2;return true;
  };
  for(const auto &r:c.resources)if(!sharedVisit(r))return fail(e,"invalid/cyclic shared resource reference");
  ids.clear();for(const auto &m:c.memories){if(m.id.empty() || !ids.insert(m.id).second || m.capacityBytes<=0 || m.alignmentBytes<=0 || !domain(m.domain) ||
       (m.accessibleDevices.empty() && m.accessibleResources.empty()))return fail(e,"invalid memory/domain/capacity/access");
    if(!m.capacityKnown && m.capacityBytes!=INT64_MAX)return fail(e,"unknown memory capacity must use the arithmetic accounting ceiling");
    std::set<std::string> seen;for(const auto &id:m.accessibleResources){auto r=resource(c,id);if(!r || r->kind!="compute" || !seen.insert(id).second)return fail(e,"memory has unknown/duplicate compute accessor");}}
  if(c.memories.empty() || c.resources.empty())return fail(e,"memory and resource instances required");
  ids.clear();for(const auto &l:c.links){auto r=resource(c,l.resource);if(l.id.empty() || !ids.insert(l.id).second || !memory(c,l.source) || !memory(c,l.target) ||
      l.source==l.target || !r || (r->kind!="transfer" && r->kind!="communication") || !domain(l.domain) || l.minBytes<0 || l.maxBytes<l.minBytes || l.temporaryBytes<0 ||
      !std::isfinite(l.latencyMs) || l.latencyMs<0 || !std::isfinite(l.bandwidthBytesPerMs) || l.bandwidthBytesPerMs<0)return fail(e,"invalid transfer path/resource/domain/cost");}
  return true;
}
bool orbit::taskflow::findMemoryPath(const StaticConfig &c,llvm::StringRef source,llvm::StringRef target,std::vector<std::string> &path,std::string &e){
  path.clear();if(!validateMemoryArchitecture(c,e))return false;if(!memory(c,source) || !memory(c,target))return fail(e,"unknown path endpoint");
  std::queue<std::string> q;std::map<std::string,std::pair<std::string,std::string>> prev;q.push(source.str());prev[source.str()]={"",""};
  while(!q.empty()){auto at=q.front();q.pop();if(at==target){std::string cur=at;while(cur!=source){path.push_back(prev.at(cur).second);cur=prev.at(cur).first;}std::reverse(path.begin(),path.end());return true;}
    for(const auto &l:c.links)if(l.source==at && !prev.count(l.target)){prev[l.target]={at,l.id};q.push(l.target);}}
  return fail(e,"no transfer path from "+source.str()+" to "+target.str());
}
llvm::json::Value orbit::taskflow::serializeMemoryArchitecture(const StaticConfig &c){
  A domains,resources,memories,links;
  for(const auto &d:c.domains){O x{{"id",d.id},{"parent",d.parent},{"kind",d.kind}};if(!d.coordinates.empty()){O coords;for(auto &a:d.coordinates)coords[a.first]=a.second;x["coordinates"]=std::move(coords);}domains.push_back(std::move(x));}
  for(const auto &r:c.resources){O x{{"id",r.id},{"kind",r.kind},{"device",r.device},{"capacity",r.capacity},{"domain",r.domain}};if(!r.sharedResources.empty()){O shares;for(auto &a:r.sharedResources)shares[a.first]=a.second;x["shared_resources"]=std::move(shares);}resources.push_back(std::move(x));}
  for(const auto &m:c.memories){O entry{{"id",m.id},{"capacity_bytes",m.capacityBytes},{"accessible_devices",strings(m.accessibleDevices)},{"accessible_resources",strings(m.accessibleResources)},{"domain",m.domain},{"kind",m.kind}};
    if(!m.capacityKnown){entry["capacity_bytes"]=nullptr;entry["capacity_known"]=false;}
    if(m.alignmentBytes!=1)entry["alignment_bytes"]=m.alignmentBytes;memories.push_back(std::move(entry));}
  for(const auto &l:c.links)links.push_back(O{{"id",l.id},{"source",l.source},{"target",l.target},{"resource",l.resource},{"domain",l.domain},
    {"latency_ms",l.latencyMs},{"bandwidth_bytes_per_ms",l.bandwidthBytesPerMs},{"min_bytes",l.minBytes},{"max_bytes",l.maxBytes},{"temporary_bytes",l.temporaryBytes},
    {"source_layout",l.sourceLayout},{"target_layout",l.targetLayout},{"source_dtype",l.sourceDtype},{"target_dtype",l.targetDtype},
    {"memory",l.memory},{"source_device",l.sourceDevice},{"target_device",l.targetDevice},
    {"provenance",O{{"kind",l.provenance.kind},{"source",l.provenance.source}}},{"cost_status",l.bandwidthBytesPerMs>0&&!l.provenance.source.empty()?"target_estimate":"uncalibrated"}});
  return O{{"domains",std::move(domains)},{"resources",std::move(resources)},{"memories",std::move(memories)},{"links",std::move(links)}};
}
bool orbit::taskflow::parseMemoryArchitecture(const V &v,StaticConfig &c,std::string &e){
  c={};auto o=v.getAsObject();if(!o)return fail(e,"expected target object");
  if(auto v=o->get("domains"))if(!v->getAsArray())return fail(e,"domains must be array");
  if(auto ds=o->getArray("domains"))for(const auto &v:*ds){auto x=v.getAsObject();if(!x || !textFields(*x,{"id","parent","kind"},e))return fail(e,"invalid domain");StaticResourceDomain d{string(*x,"id"),string(*x,"parent"),string(*x,"kind")};if(auto value=x->get("coordinates")){auto coords=value->getAsObject();if(!coords)return fail(e,"coordinates must be integer object");for(const auto &entry:*coords){auto n=entry.second.getAsInteger();if(!n || entry.first.str().empty())return fail(e,"invalid coordinate axis/value");d.coordinates[entry.first.str()]=*n;}}c.domains.push_back(std::move(d));}
  auto rs=o->getArray("resources"),ms=o->getArray("memories"),ls=o->getArray("links");if(!rs || !ms || !ls)return fail(e,"resources/memories/links arrays required");
  for(const auto &v:*rs){auto x=v.getAsObject();if(!x || !textFields(*x,{"id","kind","device","domain"},e))return fail(e,"invalid resource");auto n=x->getInteger("capacity");if(!n)return fail(e,"resource capacity required");StaticResource r{string(*x,"id"),string(*x,"kind"),string(*x,"device"),*n,string(*x,"domain")};if(auto value=x->get("shared_resources")){auto shares=value->getAsObject();if(!shares)return fail(e,"shared_resources must be integer object");for(const auto &entry:*shares){auto n=entry.second.getAsInteger();if(!n)return fail(e,"invalid shared resource units");r.sharedResources[entry.first.str()]=*n;}}c.resources.push_back(std::move(r));}
  for(const auto &v:*ms){auto x=v.getAsObject();if(!x || !textFields(*x,{"id","kind","domain"},e))return fail(e,"invalid memory");StaticMemory m;m.id=string(*x,"id");m.domain=string(*x,"domain");m.kind=string(*x,"kind");
    if(auto known=x->get("capacity_known")){auto value=known->getAsBoolean();if(!value)return fail(e,"capacity_known must be boolean");m.capacityKnown=*value;}
    auto n=x->getInteger("capacity_bytes");
    if(m.capacityKnown){if(!n)return fail(e,"known memory capacity required");m.capacityBytes=*n;}
    else {auto value=x->get("capacity_bytes");if(!value || !value->getAsNull())return fail(e,"unknown memory capacity requires null bytes");m.capacityBytes=INT64_MAX;}
    if(auto value=x->get("alignment_bytes")){auto alignment=value->getAsInteger();if(!alignment)return fail(e,"integer memory alignment required");m.alignmentBytes=*alignment;}
    if(!readStrings(*x,"accessible_devices",m.accessibleDevices,e) || !readStrings(*x,"accessible_resources",m.accessibleResources,e))return false;c.memories.push_back(m);}
  for(const auto &v:*ls){auto x=v.getAsObject();if(!x || !textFields(*x,{"id","source","target","resource","domain","memory","source_device","target_device","source_layout","target_layout","source_dtype","target_dtype"},e))return fail(e,"invalid link");StaticTransfer l;l.id=string(*x,"id");l.source=string(*x,"source");l.target=string(*x,"target");l.resource=string(*x,"resource");l.domain=string(*x,"domain");
    for(auto entry:{std::make_pair("latency_ms",&l.latencyMs),std::make_pair("bandwidth_bytes_per_ms",&l.bandwidthBytesPerMs)})if(const V *p=x->get(entry.first)){auto n=p->getAsNumber();if(!n)return fail(e,"invalid link cost");*entry.second=*n;}
    for(auto entry:{std::make_pair("min_bytes",&l.minBytes),std::make_pair("max_bytes",&l.maxBytes),std::make_pair("temporary_bytes",&l.temporaryBytes)})if(const V *p=x->get(entry.first)){auto n=p->getAsInteger();if(!n)return fail(e,"invalid link byte constraint");*entry.second=*n;}
    l.sourceLayout=string(*x,"source_layout");l.targetLayout=string(*x,"target_layout");l.sourceDtype=string(*x,"source_dtype");l.targetDtype=string(*x,"target_dtype");
    l.memory=string(*x,"memory");l.sourceDevice=string(*x,"source_device");l.targetDevice=string(*x,"target_device");
    if(auto p=x->getObject("provenance"))l.provenance={string(*p,"kind"),string(*p,"source")};c.links.push_back(l);}
  return validateMemoryArchitecture(c,e);
}
GenericMemoryBinding::GenericMemoryBinding():architecture(npuArchitecture(24576)){}
llvm::json::Value orbit::taskflow::serializeGenericMemory(const GenericMemoryBinding &b){
  return O{{"architecture",serializeMemoryArchitecture(b.architecture)},{"binding",O{{"host_memory",b.hostMemory},{"local_memory",b.localMemory},{"compute_resource",b.computeResource},{"dma_resource",b.dmaResource},{"input_link",b.inputLink},{"output_link",b.outputLink},{"submission_resource",b.submissionResource},{"host_air_memory_space",0},{"local_air_memory_space",2}}}};
}
bool orbit::taskflow::checkGenericMemory(const GenericMemoryBinding &b,int64_t capacity,std::string &e){
  ProgramTargetBindingView view{b.hostMemory,b.localMemory,b.computeResource,b.dmaResource,b.inputLink,b.outputLink,b.submissionResource};
  return checkProgramTargetBinding(b.architecture,view,capacity,e);
}
bool orbit::taskflow::configureGenericMemory(const V &config,int64_t capacity,GenericMemoryBinding &b,std::string &e){
  b=GenericMemoryBinding{};b.architecture=npuArchitecture(capacity);auto object=config.getAsObject();auto cfg=object?object->getObject("generic_air"):nullptr;
  if(cfg){if(auto v=cfg->get("memory_architecture"))if(!parseMemoryArchitecture(*v,b.architecture,e))return false;
    if(auto v=cfg->get("memory_binding")){auto o=v->getAsObject();if(!o)return fail(e,"expected memory_binding object");
      for(auto entry:{std::make_pair("host_memory",&b.hostMemory),std::make_pair("local_memory",&b.localMemory),std::make_pair("compute_resource",&b.computeResource),std::make_pair("dma_resource",&b.dmaResource),std::make_pair("input_link",&b.inputLink),std::make_pair("output_link",&b.outputLink),std::make_pair("submission_resource",&b.submissionResource)})if(const V *v=o->get(entry.first)){auto s=v->getAsString();if(!s || s->empty())return fail(e,"invalid memory binding identifier");*entry.second=s->str();}}}
  return checkGenericMemory(b,capacity,e);
}
llvm::json::Value orbit::taskflow::serializeGenericCapabilities(const GenericTargetCapabilities &c){
  A align,impl;for(auto n:c.tileAlignment)align.push_back(n);for(const auto &i:c.implementations)impl.push_back(O{{"kind",i.kind},{"lanes",i.lanes}});
  return O{{"capacity_bytes",c.capacityBytes},{"max_group_inputs",c.maxGroupInputs},{"max_group_outputs",c.maxGroupOutputs},{"acquire_all_inputs_at_group_entry",c.acquireAllInputsAtGroupEntry},{"tile_alignment",std::move(align)},{"implementations",std::move(impl)}};
}
bool orbit::taskflow::checkGenericCapabilities(const GenericMemoryBinding &b,const GenericTargetCapabilities &c,std::string &e){
  if(c.maxGroupInputs<0 || c.maxGroupOutputs<0 || c.tileAlignment.size()!=3 || c.implementations.empty() || !checkGenericMemory(b,c.capacityBytes,e))return fail(e,"invalid target capability coverage");
  for(auto n:c.tileAlignment)if(n<=0)return fail(e,"positive tile alignment required");
  std::set<std::pair<std::string,int64_t>> seen;
  for(const auto &i:c.implementations)if(!((i.kind=="scalar" && i.lanes==1)||(i.kind=="scalar-product-vector-add-f32"&&i.lanes==8)) ||
      !seen.emplace(i.kind,i.lanes).second)return fail(e,"unsupported/duplicate target implementation");
  if(c.identity!=staticFingerprint(O{{"memory",serializeGenericMemory(b)},{"capabilities",serializeGenericCapabilities(c)}}))return fail(e,"target capability identity changed");
  return true;
}
bool orbit::taskflow::getGenericTargetCapabilities(const V &config,GenericTargetCapabilities &c,std::string &e){
  c={};auto object=config.getAsObject();auto cfg=object?object->getObject("generic_air"):nullptr;
  if(cfg)if(auto v=cfg->get("capacity_bytes")){auto n=v->getAsInteger();if(!n)return fail(e,"integer capacity required");c.capacityBytes=*n;}
  GenericMemoryBinding b;if(!configureGenericMemory(config,c.capacityBytes,b,e))return false;
  if(cfg)if(auto v=cfg->get("capabilities")){
    auto cap=v->getAsObject();if(!cap)return fail(e,"capabilities must be object");
    for(auto entry:{std::make_pair("max_group_inputs",&c.maxGroupInputs),std::make_pair("max_group_outputs",&c.maxGroupOutputs)})if(auto v=cap->get(entry.first)){auto n=v->getAsInteger();if(!n || *n<0)return fail(e,"nonnegative group stream limit required");*entry.second=*n;}
    if(auto v=cap->get("acquire_all_inputs_at_group_entry")){auto b=v->getAsBoolean();if(!b)return fail(e,"input acquisition policy must be boolean");c.acquireAllInputsAtGroupEntry=*b;}
    if(auto v=cap->get("tile_alignment")){auto a=v->getAsArray();if(!a || a->size()!=3)return fail(e,"three tile alignments required");c.tileAlignment.clear();for(const auto &v:*a){auto n=v.getAsInteger();if(!n || *n<=0)return fail(e,"positive tile alignment required");c.tileAlignment.push_back(*n);}}
    if(auto v=cap->get("implementations")){auto a=v->getAsArray();if(!a)return fail(e,"implementation capabilities must be array");c.implementations.clear();for(const auto &v:*a){auto o=v.getAsObject();if(!o)return fail(e,"invalid implementation capability");GenericTaskImplementation i;i.kind=string(*o,"kind");
      i.lanes=i.kind=="scalar"?1:8;if(auto v=o->get("lanes")){auto n=v->getAsInteger();if(!n)return fail(e,"integer vector lanes required");i.lanes=*n;}c.implementations.push_back(i);}}
  }
  c.identity=staticFingerprint(O{{"memory",serializeGenericMemory(b)},{"capabilities",serializeGenericCapabilities(c)}});
  return checkGenericCapabilities(b,c,e);
}

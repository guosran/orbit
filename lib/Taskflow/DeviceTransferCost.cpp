#include "Taskflow/DeviceTransferCost.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>
using namespace orbit::taskflow;
namespace {
using J=llvm::json::Value;using O=llvm::json::Object;using A=llvm::json::Array;
bool fail(std::string &e,const std::string &s){e="device transfer cost: "+s;return false;}
bool digest(llvm::StringRef s){return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
bool add(int64_t &a,int64_t b){if(b<0 || a>INT64_MAX-b)return false;a+=b;return true;}
A ids(const std::vector<size_t> &v){A a;for(auto x:v)a.push_back(int64_t(x));return a;}
A strings(const std::vector<std::string> &v){A a;for(const auto &x:v)a.push_back(x);return a;}
std::string text(const J &v){std::string s;llvm::raw_string_ostream out(s);out<<v;return out.str();}
J descriptors(const std::vector<DeviceTransferDescriptor> &v){A a;for(const auto &d:v)a.push_back(O{
  {"id",d.id},{"offset_bytes",d.offsetBytes},{"elements",d.elements},
  {"repetitions",d.repetitions},{"element_bytes",d.elementBytes}});return a;}
bool copyBody(const GenericAnalysis &a,size_t &input,size_t &output,int64_t &elements,std::string &e) {
  if(a.tasks.size()!=1 || a.storages.size()!=2 || !a.function->hasAttr("orbit.noalias"))
    return fail(e,"initial implementation requires one actual noalias copy task and two storages");
  input=output=SIZE_MAX;
  for(size_t i=0;i<a.storages.size();++i) {
    const auto &s=a.storages[i];auto type=mlir::dyn_cast<mlir::MemRefType>(s.root.getType());
    if(s.dtype!="f32" || s.shape.size()!=2 || s.shape[0]!=1 || s.shape[1]<=0 ||
       !type || !type.getLayout().isIdentity())return fail(e,"copy requires static identity-layout [1,E] f32");
    if(s.role=="input")input=i;else if(s.role=="output")output=i;
  }
  if(input==SIZE_MAX || output==SIZE_MAX || a.storages[input].shape!=a.storages[output].shape)
    return fail(e,"copy input/output shape or role differs");
  elements=a.storages[input].shape[1];
  if(elements>INT64_MAX/8 || elements%16)return fail(e,"copy total payload must fit int64 and be 64-byte aligned");
  const auto &task=a.tasks[0];
  if(task.loops.size()!=2 || task.accesses.size()!=2 || task.loops[0].lower!=0 || task.loops[0].upper!=1 ||
     task.loops[1].lower!=0 || task.loops[1].upper!=elements || task.loops[0].step!=1 || task.loops[1].step!=1 ||
     !task.loops[0].operation->isProperAncestor(task.loops[1].operation))
    return fail(e,"copy requires the exact two-dimensional unit-step source nest");
  mlir::affine::AffineLoadOp load;mlir::affine::AffineStoreOp store;unsigned loads=0,stores=0;bool allowed=true;
  task.operation->walk([&](mlir::Operation *op){
    if(op==task.operation)return;
    if(auto x=mlir::dyn_cast<mlir::affine::AffineLoadOp>(op)){load=x;++loads;return;}
    if(auto x=mlir::dyn_cast<mlir::affine::AffineStoreOp>(op)){store=x;++stores;return;}
    auto name=op->getName().getStringRef();
    if(name!="affine.for" && name!="affine.yield" && name!="taskflow.yield")allowed=false;
  });
  if(!allowed || loads!=1 || stores!=1 || store.getValue()!=load.getResult())
    return fail(e,"body is not an actual arithmetic-free affine load-to-store copy");
  auto identity=[&](auto op){auto map=op.getAffineMap();auto operands=op.getMapOperands();
    return map.isIdentity() && map.getNumDims()==2 && map.getNumSymbols()==0 && operands.size()==2 &&
      operands[0]==task.loops[0].induction && operands[1]==task.loops[1].induction;};
  if(!identity(load) || !identity(store) || !task.loops[1].operation->isProperAncestor(load.getOperation()) ||
     !task.loops[1].operation->isProperAncestor(store.getOperation()))return fail(e,"copy accesses are not direct matching source induction axes");
  bool read=false,write=false;
  for(const auto &access:task.accesses){
    read |= access.operation==load.getOperation() && access.storage==input && access.read && !access.write;
    write |= access.operation==store.getOperation() && access.storage==output && access.write && !access.read;
  }
  return read&&write?true:fail(e,"actual copy storage/access binding differs");
}
bool bindingIdentity(const DeviceTransferBackendBinding &b,std::string &e) {
  if(b.graphId.empty() || b.nativePlanId.empty() || !digest(b.artifactSHA256) || b.runtimeIdentity.empty() ||
     b.toolchainIdentity.empty() || b.pathIdentity.empty() || b.observerIdentity.empty() || b.conditionIdentity.empty() ||
     b.completionBoundary!="submit-to-successful-wait-return")
    return fail(e,"missing checked graph/artifact/runtime/path/observer/condition or unsupported completion boundary");
  return true;
}
bool modelValid(const DeviceTransferCostModel &m,std::string &e) {
  if(m.modelId.empty() || !digest(m.parameterFileSHA256))return fail(e,"model label and exact-file digest required");
  for(size_t i=0;i<m.rows.size();++i){const auto &r=m.rows[i];const auto &d=r.domain;
    if(d.runtimeIdentity.empty() || d.toolchainIdentity.empty() || d.pathIdentity.empty() || d.observerIdentity.empty() ||
       d.conditionIdentity.empty() || d.architectureIdentity.empty() || d.implementation!="identity-copy-f32" ||
       d.completionBoundary!="submit-to-successful-wait-return" || d.minBytes<=0 || d.maxBytes<d.minBytes ||
       d.minSegments<=0 || d.maxSegments<d.minSegments || d.minTileElements<=0 || d.maxTileElements<d.minTileElements)
      return fail(e,"invalid composite model domain");
    unsigned n=bool(r.entryNs)+bool(r.segmentNs)+bool(r.byteNs);
    if(n!=0 && n!=3)return fail(e,"all three coefficients must be supplied together, or all unknown");
    if(n){if(r.provenance.kind.empty() || r.provenance.source.empty() || !digest(r.rawManifestSHA256))
        return fail(e,"priced composite requires explicit calibration provenance/raw digest");
      for(auto v:{*r.entryNs,*r.segmentNs,*r.byteNs})if(!std::isfinite(v) || v<0)return fail(e,"invalid coefficient");
      if(*r.entryNs==0 && *r.segmentNs==0 && *r.byteNs==0)return fail(e,"zero composite is not an unknown fallback");}
    for(size_t j=0;j<i;++j){const auto &x=m.rows[j].domain;
      if(std::tie(d.runtimeIdentity,d.toolchainIdentity,d.pathIdentity,d.observerIdentity,d.conditionIdentity,d.architectureIdentity)==
         std::tie(x.runtimeIdentity,x.toolchainIdentity,x.pathIdentity,x.observerIdentity,x.conditionIdentity,x.architectureIdentity) &&
         d.minBytes<=x.maxBytes && x.minBytes<=d.maxBytes && d.minSegments<=x.maxSegments && x.minSegments<=d.maxSegments &&
         d.minTileElements<=x.maxTileElements && x.minTileElements<=d.maxTileElements)
        return fail(e,"overlapping composite calibration domains");}
  }return true;
}
}

bool orbit::taskflow::collectDeviceTransferFacts(const GenericAnalysis &a,const GenericMappingPlan &p,
    const DeviceTransferBackendBinding &b,DeviceTransferFacts &out,std::string &error) {
  if(!checkGenericMapping(a,p,error) || !bindingIdentity(b,error))return false;
  const auto &g=p.communication;
  if(b.graphId!=programGraphIdentity(g) || b.nativePlanId!=p.planId)return fail(error,"stale graph/native plan binding");
  size_t input,output;int64_t elements;
  if(!copyBody(a,input,output,elements,error))return false;
  if(p.strategy!="serial-tiled" || p.hardwareStages.size()!=1 || p.executionGroups.size()!=1 ||
     p.phases.empty() || p.memoryBinding.architecture.resources.empty())
    return fail(error,"initial device interval requires one serial tiled hardware stage and one core");
  for(const auto &i:g.instances)if(i.implementation!="scalar" || i.resourceCount!=1)
    return fail(error,"copy interval requires scalar single-core instances");
  DeviceTransferFacts f;f.binding=b;f.elements=elements;f.segmentCount=p.phases.size();
  f.architectureIdentity=staticFingerprint(serializeMemoryArchitecture(p.memoryBinding.architecture));
  f.reservationResources={p.memoryBinding.dmaResource,p.memoryBinding.computeResource,p.memoryBinding.submissionResource};
  int64_t cursor=0;for(const auto &phase:p.phases){
    if(phase.task!=0 || phase.loops.size()!=2 || phase.loops[0].loop!=0 || phase.loops[0].lower!=0 || phase.loops[0].upper!=1 ||
       phase.loops[1].loop!=1 || phase.loops[1].lower!=cursor)return fail(error,"copy tile sequence is not contiguous source order");
    auto width=phase.loops[1].upper-phase.loops[1].lower;
    if(width<=0 || width%16 || (f.tileElements && f.tileElements!=width))return fail(error,"initial copy chunks must be uniform 64-byte aligned");
    f.tileElements=width;cursor=phase.loops[1].upper;
  }
  if(cursor!=elements)return fail(error,"copy tile sequence does not cover source extent");
  using Key=std::pair<size_t,std::string>;std::map<Key,std::vector<size_t>> expected;
  for(size_t event=0;event<g.events.size();++event){const auto &v=g.events[event];
    if(v.kind!="transfer"){f.coveredComputeEvents.push_back(event);continue;}
    if(!v.transferRegion || v.reads.empty() || v.writes.empty())return fail(error,"missing transfer endpoints/rectangle");
    DeviceTransferFact x;x.event=event;x.storage=v.transferRegion->storage;x.region=*v.transferRegion;
    x.sourceCopies=v.reads;x.targetCopies=v.writes;x.requestedBytes=v.transferBytes;x.resource=v.resource;x.link=v.link;
    x.sourceMemory=g.allocations[g.copies[v.reads.front()].allocation].memory;
    x.targetMemory=g.allocations[g.copies[v.writes.front()].allocation].memory;
    if(x.sourceMemory==p.memoryBinding.hostMemory && x.targetMemory==p.memoryBinding.localMemory && x.storage==input)x.direction="host_to_l1";
    else if(x.sourceMemory==p.memoryBinding.localMemory && x.targetMemory==p.memoryBinding.hostMemory && x.storage==output)x.direction="l1_to_host";
    else return fail(error,"copy transfer endpoint/storage differs from the actual implementation");
    for(auto copy:v.reads)x.versions.push_back(g.copies[copy].version);
    expected[{x.storage,x.direction}].push_back(event);
    if(!add(f.requestedBytes,v.transferBytes))return fail(error,"requested payload overflow");
    f.transfers.push_back(std::move(x));
  }
  if(expected.size()!=2 || b.streams.size()!=2)return fail(error,"initial path requires exactly two directional streams");
  std::set<Key> streams;std::set<std::string> descriptorsSeen;
  for(const auto &s:b.streams){Key key{s.storage,s.direction};
    if(s.stage!=0 || !streams.insert(key).second || !expected.count(key) || s.publicEvents!=expected.at(key) ||
       s.runtimeDescriptors.empty() || s.coreDescriptors.empty())return fail(error,"descriptor stream omits/duplicates or changes checked public events");
    int64_t requested=0;for(auto event:s.publicEvents)if(!add(requested,g.events[event].transferBytes))return fail(error,"stream bytes overflow");
    auto validateDescriptors=[&](const std::vector<DeviceTransferDescriptor> &list,bool core,int64_t &total){
      int64_t offset=0;
      for(const auto &d:list){
        if(d.id.empty() || !descriptorsSeen.insert(d.id).second || d.elementBytes!=4 || d.offsetBytes<0 || d.offsetBytes%64 ||
           d.elements<=0 || d.elements>INT64_MAX/4 || d.repetitions<=0 || d.elements*4>INT64_MAX/d.repetitions)
          return fail(error,"invalid/duplicate descriptor identity, alignment, width or length/repetition overflow");
        int64_t bytes=d.elements*4*d.repetitions;
        if(core){if(d.offsetBytes!=0 || d.elements!=f.tileElements || !add(f.coreTransactions,d.repetitions))
            return fail(error,"core cyclic descriptor differs from checked tile transactions");}
        else {if(d.offsetBytes!=offset || d.repetitions!=1 || !add(offset,bytes) || !add(f.runtimeDescriptorCount,1))
            return fail(error,"runtime descriptors must cover the contiguous tensor exactly once");}
        if(!add(total,bytes))return fail(error,"descriptor payload overflow");
      }return true;
    };
    int64_t runtime=0,core=0;
    if(!validateDescriptors(s.runtimeDescriptors,false,runtime) || !validateDescriptors(s.coreDescriptors,true,core))return false;
    if(runtime!=requested || core!=requested || !add(f.descriptorBytes,runtime))return fail(error,"runtime/core descriptor bytes differ from public payload");
  }
  f.publicTransferCount=f.transfers.size();
  if(f.publicTransferCount!=f.segmentCount*2 || f.coreTransactions!=f.publicTransferCount ||
     f.requestedBytes!=elements*8)return fail(error,"copy event/transaction/payload totals disagree");
  out=std::move(f);return true;
}
bool orbit::taskflow::checkDeviceTransferFacts(const GenericAnalysis &a,const GenericMappingPlan &p,
    const DeviceTransferFacts &facts,std::string &error) {
  DeviceTransferFacts expected;if(!collectDeviceTransferFacts(a,p,facts.binding,expected,error))return false;
  return text(serializeDeviceTransferFacts(expected))==text(serializeDeviceTransferFacts(facts))?true:fail(error,"facts differ from checked source/plan/backend binding");
}

bool orbit::taskflow::parseDeviceTransferCostModel(llvm::StringRef bytes,DeviceTransferCostModel &out,std::string &error) {
  auto parsed=llvm::json::parse(bytes);
  if(!parsed){llvm::consumeError(parsed.takeError());return fail(error,"invalid model JSON");}
  auto *top=parsed->getAsObject();
  if(!top || top->getString("schema")!="orbit.device-transfer.model.v15" || top->getString("units")!="ns")
    return fail(error,"unsupported model schema/units");
  auto id=top->getString("model_id");auto *rows=top->getArray("rows");
  if(!id || !rows)return fail(error,"missing model label/rows");
  DeviceTransferCostModel m;m.modelId=id->str();llvm::SHA256 hash;hash.update(bytes);m.parameterFileSHA256=llvm::toHex(hash.final(),true);
  for(const auto &v:*rows){auto *o=v.getAsObject();auto *d=o?o->getObject("domain"):nullptr;
    if(!d)return fail(error,"row missing domain");DeviceTransferCostRow r;
    const std::array<std::pair<const char *,std::string *>,8> texts={{{"runtime_identity",&r.domain.runtimeIdentity},
      {"toolchain_identity",&r.domain.toolchainIdentity},{"path_identity",&r.domain.pathIdentity},
      {"observer_identity",&r.domain.observerIdentity},{"condition_identity",&r.domain.conditionIdentity},
      {"architecture_identity",&r.domain.architectureIdentity},{"implementation",&r.domain.implementation},
      {"completion_boundary",&r.domain.completionBoundary}}};
    for(const auto &item:texts){auto x=d->getString(item.first);if(!x)return fail(error,"missing domain string "+std::string(item.first));*item.second=x->str();}
    const std::array<std::pair<const char *,int64_t *>,6> integers={{{"min_bytes",&r.domain.minBytes},{"max_bytes",&r.domain.maxBytes},
      {"min_segments",&r.domain.minSegments},{"max_segments",&r.domain.maxSegments},
      {"min_tile_elements",&r.domain.minTileElements},{"max_tile_elements",&r.domain.maxTileElements}}};
    for(const auto &item:integers){auto x=d->getInteger(item.first);if(!x)return fail(error,"missing domain integer "+std::string(item.first));*item.second=*x;}
    const std::array<std::pair<const char *,std::optional<double> *>,3> coefficients={{{"entry_ns",&r.entryNs},{"segment_ns",&r.segmentNs},{"byte_ns",&r.byteNs}}};
    for(const auto &item:coefficients)if(auto *value=o->get(item.first)) {
      if(value->getAsNull())continue;auto x=value->getAsNumber();if(!x)return fail(error,"coefficient must be numeric or null");*item.second=*x;
    }
    if(auto *provenance=o->getObject("provenance")) {
      r.provenance.kind=provenance->getString("kind").value_or("").str();
      r.provenance.source=provenance->getString("source").value_or("").str();
    }
    r.rawManifestSHA256=o->getString("raw_manifest_sha256").value_or("").str();m.rows.push_back(std::move(r));
  }
  if(!modelValid(m,error))return false;out=std::move(m);return true;
}

bool orbit::taskflow::evaluateDeviceTransferCost(const DeviceTransferFacts &f,const DeviceTransferCostModel &m,
    const DeviceTransferComposition &composition,DeviceTransferCostEvaluation &out,std::string &error) {
  out={};out.modelId=m.modelId;out.parameterFileSHA256=m.parameterFileSHA256;
  if(!modelValid(m,error) || !bindingIdentity(f.binding,error))return false;
  std::set<size_t> covered;
  for(const auto &t:f.transfers)if(!covered.insert(t.event).second)return fail(error,"duplicate transfer ownership");
  for(auto id:f.coveredComputeEvents)if(!covered.insert(id).second)return fail(error,"duplicate compute/transfer ownership");
  if(composition.submitAlreadyPriced || composition.completionAlreadyPriced)
    return fail(error,"submit or completion is already charged by another API/composite");
  for(auto id:composition.separatelyPricedEvents)if(covered.count(id))return fail(error,"device interval overlaps separately charged DMA/compute/wait-covered events");
  out.coveredEvents.assign(covered.begin(),covered.end());out.reservationResources=f.reservationResources;
  if(f.requestedBytes<=0 || f.descriptorBytes!=f.requestedBytes || f.segmentCount<=0 || f.tileElements<=0 ||
     f.coreTransactions!=f.publicTransferCount || f.reservationResources.size()!=3)
    return fail(error,"malformed checked interval facts");
  for(size_t i=0;i<m.rows.size();++i){const auto &r=m.rows[i];const auto &d=r.domain;const auto &b=f.binding;
    if(d.runtimeIdentity!=b.runtimeIdentity || d.toolchainIdentity!=b.toolchainIdentity || d.pathIdentity!=b.pathIdentity ||
       d.observerIdentity!=b.observerIdentity || d.conditionIdentity!=b.conditionIdentity ||
       d.completionBoundary!=b.completionBoundary || d.architectureIdentity!=f.architectureIdentity || d.implementation!=f.implementation ||
       f.requestedBytes<d.minBytes || f.requestedBytes>d.maxBytes || f.segmentCount<d.minSegments || f.segmentCount>d.maxSegments ||
       f.tileElements<d.minTileElements || f.tileElements>d.maxTileElements)continue;
    out.matchedRow=i;
    if(!r.entryNs){out.reason="matching interval domain has uncalibrated coefficients";return true;}
    double ns=*r.entryNs+*r.segmentNs*double(f.segmentCount)+*r.byteNs*double(f.requestedBytes);
    if(!std::isfinite(ns)){out.reason="finite interval estimate overflow";return true;}
    out.durationNs=ns;out.status="predicted";
    out.reason="calibrated submit-to-successful-wait-return composite; not a pure-DMA or physical-bandwidth estimate";return true;
  }
  out.reason="no model domain matches checked implementation, target/path/runtime/condition/observer and size/segmentation";return true;
}

llvm::json::Value orbit::taskflow::serializeDeviceTransferBackendBinding(const DeviceTransferBackendBinding &b) {
  A streams;for(const auto &s:b.streams)streams.push_back(O{{"stage",int64_t(s.stage)},{"storage",int64_t(s.storage)},
    {"direction",s.direction},{"public_events",ids(s.publicEvents)},
    {"runtime_descriptors",descriptors(s.runtimeDescriptors)},{"core_descriptors",descriptors(s.coreDescriptors)}});
  return O{{"graph_id",b.graphId},{"native_plan_id",b.nativePlanId},{"artifact_sha256",b.artifactSHA256},
    {"runtime_identity",b.runtimeIdentity},{"toolchain_identity",b.toolchainIdentity},{"path_identity",b.pathIdentity},
    {"observer_identity",b.observerIdentity},{"condition_identity",b.conditionIdentity},
    {"completion_boundary",b.completionBoundary},{"streams",std::move(streams)}};
}
llvm::json::Value orbit::taskflow::serializeDeviceTransferFacts(const DeviceTransferFacts &f) {
  A transfers;for(const auto &t:f.transfers){A offsets,sizes;for(auto x:t.region.offsets)offsets.push_back(x);for(auto x:t.region.sizes)sizes.push_back(x);
    transfers.push_back(O{{"event",int64_t(t.event)},{"stage",int64_t(t.stage)},{"storage",int64_t(t.storage)},
      {"direction",t.direction},{"source_memory",t.sourceMemory},{"target_memory",t.targetMemory},
      {"resource",t.resource},{"link",t.link},{"source_copies",ids(t.sourceCopies)},{"target_copies",ids(t.targetCopies)},
      {"versions",ids(t.versions)},{"offsets",std::move(offsets)},{"sizes",std::move(sizes)},{"requested_bytes",t.requestedBytes}});
  }
  return O{{"schema","orbit.device-transfer.facts.v15"},{"binding",serializeDeviceTransferBackendBinding(f.binding)},
    {"architecture_identity",f.architectureIdentity},{"implementation",f.implementation},
    {"transfers",std::move(transfers)},{"covered_compute_events",ids(f.coveredComputeEvents)},
    {"reservation_resources",strings(f.reservationResources)},{"public_transfer_count",f.publicTransferCount},
    {"runtime_descriptor_count",f.runtimeDescriptorCount},{"core_transactions",f.coreTransactions},
    {"requested_bytes",f.requestedBytes},{"descriptor_bytes",f.descriptorBytes},{"physical_ddr_bytes",nullptr},
    {"segment_count",f.segmentCount},{"tile_elements",f.tileElements},{"elements",f.elements},
    {"completion_scope","device DMA + local copy + locks + configuration + runtime/polling until successful wait return"},
    {"measurement_proof","typed backend witness required; Core validation does not itself observe hardware"}};
}
llvm::json::Value orbit::taskflow::serializeDeviceTransferCost(const DeviceTransferCostEvaluation &v) {
  A resources;for(const auto &id:v.reservationResources)resources.push_back(O{{"resource",id},{"units",1},
    {"reserved_ns",v.durationNs?J(*v.durationNs):J(nullptr)},
    {"scope","conservative whole-interval serial reservation; not measured utilization"}});
  return O{{"schema","orbit.device-transfer.cost.v15"},{"status",v.status},{"reason",v.reason},{"model_id",v.modelId},
    {"parameter_file_sha256",v.parameterFileSHA256},{"units","ns"},{"duration_ns",v.durationNs?J(*v.durationNs):J(nullptr)},
    {"matched_row",v.matchedRow?J(int64_t(*v.matchedRow)):J(nullptr)},{"covered_events",ids(v.coveredEvents)},
    {"resource_reservations",std::move(resources)},{"physical_ddr_bytes",nullptr},{"complete_program_cost",false},
    {"owner_scope","submit-to-successful-wait-return"},{"separately_charge_native_dma",false},
    {"excluded_costs",A{"BO-sync","host-input-reset","final-output-copy","outside-interval-dispatch"}}};
}

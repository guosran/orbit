#include "Taskflow/GenericProgramContrast.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
using namespace orbit::taskflow;
namespace {
using J=llvm::json::Value; using O=llvm::json::Object; using A=llvm::json::Array;
bool fail(std::string &error,const std::string &s){error="program contrast: "+s;return false;}
A strings(const std::vector<std::string> &v){A out;for(const auto &s:v)out.push_back(s);return out;}
O quantities(const std::map<std::string,int64_t> &v){O out;for(const auto &x:v)out[x.first]=x.second;return out;}
J optional(std::optional<double> v){return v?J(*v):J(nullptr);}
bool increment(int64_t &v,int64_t amount){if(amount<0 || v>INT64_MAX-amount)return false;v+=amount;return true;}
bool facts(const GenericAnalysis &a,const GenericMappingPlan &p,GenericContrastPlanFacts &out,std::string &error){
  if(!checkGenericMapping(a,p,error))return false;
  GenericCostFeatures f;if(!collectGenericCostFeatures(a,p,f,error))return false;
  out={};out.graphId=programGraphIdentity(p.communication);out.planId=p.planId;
  out.bodyFingerprint=a.bodyFingerprint;out.familyFingerprint=f.regionFamilyFingerprint;
  out.architectureFingerprint=f.architectureFingerprint;out.implementations=f.implementations;
  auto &q=out.quantities;
  for(const auto &x:f.values)q[x.first]=x.second;
  q["requested_dma_bytes"]=p.communication.transferBytes;
  q["source_task_count"]=a.tasks.size();q["abi_to_sync_calls"]=0;q["abi_from_sync_calls"]=0;
  q["abi_to_requested_bytes"]=0;q["abi_from_requested_bytes"]=0;
  q["abi_to_requested_64byte_chunks"]=0;q["abi_from_requested_64byte_chunks"]=0;
  for(const auto &s:a.storages){
    int64_t bytes=4;
    for(auto extent:s.shape)if(extent<=0 || bytes>INT64_MAX/extent)return fail(error,"ABI extent overflows");else bytes*=extent;
    // This current execution contract is scalar f32, caller-bound BOs, all ABI
    // buffers TO and only outputs FROM. These are API requests, not DDR traffic.
    if(s.dtype!="f32")return fail(error,"contrast ABI currently requires f32 storage");
    if(s.role!="temporary" || p.strategy=="serial-tiled"){
      if(!increment(q["abi_to_sync_calls"],1) || !increment(q["abi_to_requested_bytes"],bytes) ||
         !increment(q["abi_to_requested_64byte_chunks"],bytes/64+(bytes%64!=0)))return fail(error,"ABI quantity overflow");
    }
    if(s.role=="output"){
      if(!increment(q["abi_from_sync_calls"],1) || !increment(q["abi_from_requested_bytes"],bytes) ||
         !increment(q["abi_from_requested_64byte_chunks"],bytes/64+(bytes%64!=0)))return fail(error,"output ABI quantity overflow");
    }
  }
  q["input_delivery_count"]=0;q["input_delivery_bytes"]=0;q["intermediate_delivery_bytes"]=0;
  for(const auto &event:p.communication.events)if(event.kind=="transfer"){
    if(!event.transferRegion)return fail(error,"transfer region missing");
    auto storage=event.transferRegion->storage;
    if(storage>=a.storages.size())return fail(error,"transfer storage outside source");
    const auto &role=a.storages[storage].role;
    if(role=="input" && event.link==p.memoryBinding.inputLink){
      if(!increment(q["input_delivery_count"],1) || !increment(q["input_delivery_bytes"],event.transferBytes))return fail(error,"input delivery quantity overflow");
    }
    if(role=="temporary" && !increment(q["intermediate_delivery_bytes"],event.transferBytes))return fail(error,"intermediate delivery quantity overflow");
  }
  q["shared_local_copy_count"]=0;
  for(const auto &copy:p.communication.copies)
    if(copy.consumerEdges.size()>1 && p.communication.allocations[copy.allocation].memory==p.memoryBinding.localMemory)
      ++q["shared_local_copy_count"];
  // A structural coupled implementation unit, NOT a count of configuration
  // calls. The row owns all changed stage/ABI/buffering/loop behavior together.
  q["materialized_region_realizations"]=(p.strategy=="serial-tiled" && q["intermediate_delivery_bytes"]>0)?1:0;
  return true;
}
J plan(const GenericContrastPlanFacts &p){return O{{"graph_id",p.graphId},{"plan_id",p.planId},{"body_fingerprint",p.bodyFingerprint},{"family_fingerprint",p.familyFingerprint},{"architecture_fingerprint",p.architectureFingerprint},{"implementations",strings(p.implementations)},{"quantities",quantities(p.quantities)}};}
bool member(const A &v,llvm::StringRef key){return llvm::any_of(v,[&](const J &x){return x.getAsString()==key;});}
}
bool orbit::taskflow::collectGenericProgramContrastFacts(const GenericAnalysis &la,const GenericMappingPlan &lp,
    const GenericAnalysis &ra,const GenericMappingPlan &rp,GenericProgramContrastFacts &out,std::string &error){
  out={};if(!facts(la,lp,out.baseline,error) || !facts(ra,rp,out.candidate,error))return false;
  const auto &l=out.baseline;const auto &r=out.candidate;
  if(l.bodyFingerprint!=r.bodyFingerprint || l.familyFingerprint!=r.familyFingerprint ||
      l.architectureFingerprint!=r.architectureFingerprint || l.implementations!=r.implementations)
    return fail(error,"same source body, implementation and architecture required");
  if(l.quantities.at("command_count")!=1 || r.quantities.at("command_count")!=1)
    return fail(error,"contrast covers a single prepared invocation per plan");
  for(const auto &x:l.quantities){auto it=r.quantities.find(x.first);if(it==r.quantities.end())return fail(error,"native quantity keys differ");out.delta[x.first]=it->second-x.second;}
  out.commonConditions={"same authenticated computation body, source data, dtype and arithmetic implementation",
      "two prepared contexts, same runtime/observer/polling and bound-BO policy required by model target",
      "same host invocation completion boundary; command count is one on each side"};
  out.unknowns={"absolute invocation durations and shared baseline offset remain unknown",
      "same source arithmetic does not prove unchanged local-code duration",
      "pure DMA/core time, physical traffic and actual overlap remain unidentified"};
  bool retained=rp.choices.residency=="readonly-panel";
  if(retained && lp.readOnlyRetentions.empty() && !rp.readOnlyRetentions.empty() &&
      lp.strategy=="serial-tiled" && rp.strategy=="serial-tiled" && lp.tileSizes==rp.tileSizes &&
      l.quantities.at("hardware_stage_count")==1 && r.quantities.at("hardware_stage_count")==1 &&
      out.delta.at("abi_to_sync_calls")==0 && out.delta.at("abi_from_sync_calls")==0 &&
      out.delta.at("abi_to_requested_bytes")==0 && out.delta.at("abi_from_requested_bytes")==0 &&
      out.delta.at("intermediate_delivery_bytes")==0 && out.delta.at("dma_output_bytes")==0 &&
      out.delta.at("input_delivery_count")<0 && out.delta.at("requested_dma_bytes")<0){
    out.changeClass="readonly-copy-retention";
    out.coveredChange={"changed input delivery/locks/local-buffer reuse and generated loop implementation as one calibrated contrast",
        "requested input bytes and public delivery count, independently bound to final runtime/core implementation",
        "fixed ABI and stage counts are checked, not silently priced again"};
  } else if(lp.strategy=="serial-tiled" && rp.strategy=="single-herd-resident" &&
      rp.choices.residency=="whole-program" && l.quantities.at("intermediate_delivery_bytes")>0 &&
      r.quantities.at("intermediate_delivery_bytes")==0 && r.quantities.at("shared_local_copy_count")>0){
    out.changeClass="intermediate-region-residency";
    out.coveredChange={"whole materialized-to-resident implementation change: configurations, ABI sync, locks, local buffers and loops coupled",
        "intermediate writes/reloads removed and shared local copies retained",
        "materialized_region_realizations is a coupled implementation unit, not isolated launch latency"};
  } else {out.reason="plan contrast outside supported structural change classes; no common-cost cancellation assumed";return true;}
  out.supported=true;return true;
}
bool orbit::taskflow::evaluateGenericProgramContrast(const GenericProgramContrastFacts &f,const J &model,
    const J &target,GenericProgramContrastCost &out,std::string &error){
  out={};out.changeClass=f.changeClass;out.modelIdentity=staticFingerprint(model);
  auto unknown=[&](const std::string &why){out.reason=why;return true;};
  if(!f.supported)return unknown(f.reason);
  if(f.baseline.graphId.empty() || f.candidate.graphId.empty() ||
      f.baseline.bodyFingerprint!=f.candidate.bodyFingerprint ||
      f.baseline.architectureFingerprint!=f.candidate.architectureFingerprint ||
      f.baseline.familyFingerprint!=f.candidate.familyFingerprint ||
      f.baseline.implementations!=f.candidate.implementations ||
      f.baseline.quantities.size()!=f.candidate.quantities.size() ||
      f.baseline.quantities.size()!=f.delta.size())return fail(error,"contrast fact identities differ");
  for(const auto &q:f.baseline.quantities){auto r=f.candidate.quantities.find(q.first);auto d=f.delta.find(q.first);
    if(q.second<0 || r==f.candidate.quantities.end() || r->second<0 || d==f.delta.end() || d->second!=r->second-q.second)
      return fail(error,"contrast quantities/delta were modified");
  }
  auto object=model.getAsObject();auto actual=target.getAsObject();
  if(!object || object->getString("schema")!="orbit.direct-air.cost.v1" ||
      object->getString("kind")!="contrast-composite" || object->getString("units")!="ns")
    return fail(error,"existing direct-AIR schema, contrast-composite kind and ns units required");
  for(auto key:{"model_id","training_manifest_digest","boundary"})
    if(!object->getString(key) || object->getString(key)->empty())return fail(error,"model identity/provenance/boundary missing");
  auto digest=object->getString("training_manifest_digest");
  if(digest->size()!=64 || !llvm::all_of(*digest,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))return fail(error,"training provenance requires SHA256");
  if(object->getString("boundary")!="prepared-command-host-to-host")return fail(error,"unsupported contrast timing boundary");
  auto bound=object->getObject("target");if(!bound)return fail(error,"model target required");
  for(auto key:{"id","runtime","toolchain","condition"}){
    auto value=bound->getString(key);if(!value || value->empty())return fail(error,"target binding incomplete");
    if(!actual || actual->getString(key)!=value)return unknown("runtime/toolchain/observer/condition target mismatch");
  }
  auto rows=object->getArray("rows");if(!rows || rows->empty())return fail(error,"contrast implementation rows required");
  const O *selected=nullptr;std::set<std::string> classes;
  for(const auto &value:*rows){auto row=value.getAsObject();auto name=row?row->getString("change_class"):std::nullopt;
    if(!name || !classes.insert(name->str()).second)return fail(error,"duplicate or invalid contrast class row");
    if(*name==f.changeClass)selected=row;
  }
  if(!selected)return unknown("no calibrated implementation change row");
  auto domain=selected->getObject("domain");auto terms=selected->getArray("terms");
  if(!domain || !terms || terms->empty() || terms->size()>8)return fail(error,"finite domain and one to eight terms required");
  auto families=domain->getArray("family_fingerprints");auto arch=domain->getArray("architecture_fingerprints");
  auto implementations=domain->getArray("implementations");
  if(!families || families->empty() || !arch || arch->empty() || !implementations || implementations->empty())return fail(error,"source-family, architecture and implementation domains required");
  if(!member(*families,f.baseline.familyFingerprint) || !member(*arch,f.baseline.architectureFingerprint))return unknown("source family or architecture outside frozen domain");
  for(const auto &s:f.baseline.implementations)if(!member(*implementations,s))return unknown("arithmetic implementation outside frozen domain");
  for(const auto side:std::array<std::pair<const char *,const GenericContrastPlanFacts *>,2>{{{"baseline",&f.baseline},{"candidate",&f.candidate}}}){
    auto ranges=domain->getObject(side.first);if(!ranges || ranges->empty())return fail(error,"both plan feature ranges required");
    for(const auto &entry:*ranges){auto limits=entry.second.getAsArray();auto quantity=side.second->quantities.find(entry.first.str());
      if(!limits || limits->size()!=2 || !(*limits)[0].getAsInteger() || !(*limits)[1].getAsInteger() ||
          *(*limits)[0].getAsInteger()<0 || *(*limits)[1].getAsInteger()<*(*limits)[0].getAsInteger() || quantity==side.second->quantities.end())return fail(error,"invalid contrast feature domain");
      if(quantity->second<*(*limits)[0].getAsInteger() || quantity->second>*(*limits)[1].getAsInteger())return unknown("plan feature outside frozen contrast domain: "+entry.first.str());
    }
  }
  double prediction=0;std::set<std::string> seen;
  for(const auto &value:*terms){auto term=value.getAsObject();auto key=term?term->getString("feature"):std::nullopt;auto coefficient=term?term->getNumber("coefficient_ns"):std::nullopt;
    if(!key || !coefficient || !std::isfinite(*coefficient) || *coefficient<0 || !seen.insert(key->str()).second || !f.delta.count(key->str()))return fail(error,"unknown/duplicate term or invalid coefficient");
    if(!domain->getObject("baseline")->get(key->str()) || !domain->getObject("candidate")->get(key->str()))return fail(error,"priced feature requires both frozen plan ranges");
    double contribution=*coefficient*f.delta.at(key->str());prediction+=contribution;
    if(!std::isfinite(prediction))return fail(error,"contrast duration overflows");
    out.contributions.push_back(O{{"feature",key->str()},{"baseline",f.baseline.quantities.at(key->str())},{"candidate",f.candidate.quantities.at(key->str())},{"delta",f.delta.at(key->str())},{"coefficient_ns",*coefficient},{"contribution_delta_ns",contribution}});
  }
  auto boundNs=selected->getNumber("prediction_error_bound_ns");
  if(!boundNs || !std::isfinite(*boundNs) || *boundNs<0)return fail(error,"training-derived prediction error bound required");
  out.deltaNs=prediction;out.lowerDeltaNs=prediction-*boundNs;out.upperDeltaNs=prediction+*boundNs;
  out.status="conditional-implementation-contrast";
  out.reason="One exclusive changed-implementation composite; a shared unknown absolute offset is hypothesized only within the frozen contrast domain and must pass unseen-plan/session validation. No pure mechanism separation or full score.";
  return true;
}
J orbit::taskflow::serializeGenericProgramContrastFacts(const GenericProgramContrastFacts &f){
  return O{{"schema","orbit.generic-program-contrast.facts.v17"},{"supported",f.supported},{"change_class",f.changeClass},{"reason",f.reason},{"baseline",plan(f.baseline)},{"candidate",plan(f.candidate)},{"delta_quantities",quantities(f.delta)},{"covered_change",strings(f.coveredChange)},{"common_conditions",strings(f.commonConditions)},{"unknowns",strings(f.unknowns)},{"physical_ddr_bytes",nullptr},{"counts_scope","Native requested payload/delivery and installed bound-BO API requests; final backend binding must independently corroborate physical implementation counts."}};
}
J orbit::taskflow::serializeGenericProgramContrastCost(const GenericProgramContrastCost &c){
  return O{{"schema","orbit.generic-program-contrast.cost.v17"},{"status",c.status},{"reason",c.reason},{"change_class",c.changeClass},{"model_identity",c.modelIdentity},{"candidate_minus_baseline_ns",optional(c.deltaNs)},{"lower_delta_ns",optional(c.lowerDeltaNs)},{"upper_delta_ns",optional(c.upperDeltaNs)},{"contributions",A(c.contributions)},{"baseline_duration_ns",nullptr},{"candidate_duration_ns",nullptr},{"absolute_time_expression","shared_unknown + implementation-specific work; only the calibrated conditional difference is predicted"},{"critical_path_scope","one completed single-NPU command; child configurations serial; DMA/core internal critical intervals unknown"},{"resource_scope","existing ProgramGraph dependencies and npu compute/DMA/submission reservations; no measured overlap assumed"}};
}

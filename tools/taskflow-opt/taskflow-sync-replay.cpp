#include "Taskflow/SyncCost.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace cl=llvm::cl;
namespace {
cl::opt<std::string> input("input",cl::Required,cl::desc("Observed operation-stream JSONL"));
cl::opt<std::string> modelPath("sync-model",cl::desc("Optional calibrated API model JSON"));
cl::opt<std::string> output("output",cl::Required,cl::desc("New replay-results JSONL"));
cl::opt<uint64_t> maxRecords("max-records",cl::init(0),cl::desc("Maximum records; zero means all"));
using J=llvm::json::Value;using O=llvm::json::Object;using A=llvm::json::Array;
bool fail(std::string &error,const std::string &why){error="sync replay: "+why;return false;}
std::string text(const J &v){std::string s;llvm::raw_string_ostream out(s);out<<v;return out.str();}
J clone(const J &v) {
  auto p=llvm::json::parse(text(v));
  if(!p) {
    llvm::consumeError(p.takeError());
    llvm::report_fatal_error("internal JSON serialization did not round-trip");
  }
  return std::move(*p);
}
bool range(const O *o,PhysicalBORange &r,std::string &e) {
  if(!o)return fail(e,"missing range object");
  auto bo=o->getString("bo");auto generation=o->getInteger("generation");
  auto offset=o->getInteger("offset");auto bytes=o->getInteger("bytes");
  if(!bo || bo->empty() || !generation || *generation<0 || !offset || !bytes)
    return fail(e,"range requires BO, nonnegative generation, offset and bytes");
  r={bo->str(),uint64_t(*generation),*offset,*bytes};return true;
}
bool direction(const O &o,SyncDirection &d,std::string &e) {
  auto s=o.getString("direction");if(s=="TO_DEVICE")d=SyncDirection::ToDevice;
  else if(s=="FROM_DEVICE")d=SyncDirection::FromDevice;
  else return fail(e,"sync requires exact TO_DEVICE or FROM_DEVICE direction");return true;
}
bool event(const J &value,SyncObservation &v,std::string &e) {
  const auto *o=value.getAsObject();if(!o)return fail(e,"event must be an object");
  auto id=o->getInteger("id");auto k=o->getString("kind");
  if(!id || *id<=0 || !k)return fail(e,"event requires positive ordered id and kind");v.id=*id;
  static const std::map<std::string,SyncObservationKind> kinds={
    {"HostRead",SyncObservationKind::HostRead},{"HostWrite",SyncObservationKind::HostWrite},
    {"SyncBegin",SyncObservationKind::SyncBegin},{"SyncComplete",SyncObservationKind::SyncComplete},
    {"DeviceSubmit",SyncObservationKind::DeviceSubmit},{"WaitSuccess",SyncObservationKind::WaitSuccess},
    {"WaitFailure",SyncObservationKind::WaitFailure},{"UnknownMutation",SyncObservationKind::UnknownMutation}};
  auto found=kinds.find(k->str());if(found==kinds.end())return fail(e,"unknown typed event kind");v.kind=found->second;
  if(auto operation=o->getString("operation"))v.operation=operation->str();
  bool sync=v.kind==SyncObservationKind::SyncBegin || v.kind==SyncObservationKind::SyncComplete;
  bool access=v.kind==SyncObservationKind::HostRead || v.kind==SyncObservationKind::HostWrite ||
              v.kind==SyncObservationKind::UnknownMutation;
  if((sync || access) && !range(o->getObject("range"),v.range,e))return false;
  if(sync && !direction(*o,v.direction,e))return false;
  if(v.kind==SyncObservationKind::DeviceSubmit) {
    auto *list=o->getArray("accesses");if(!list)return fail(e,"device submit missing declared accesses");
    for(const auto &entry:*list) {
      auto *a=entry.getAsObject();if(!a)return fail(e,"declared access must be object");
      SyncDeclaredAccess x;if(!range(a->getObject("range"),x.range,e))return false;
      auto read=a->getBoolean("read"),write=a->getBoolean("write");
      if(!read || !write)return fail(e,"declared access requires read/write booleans");
      x.read=*read;x.write=*write;v.accesses.push_back(x);
    }
  }
  // No elapsed time, timestamps, descriptive labels or cell names enter state.
  return true;
}
bool context(const O *o,SyncQueryContext &c,std::string &e) {
  if(!o)return fail(e,"missing query context");
  auto f=o->getString("call_form"),b=o->getString("observation_boundary");
  if(f=="implicit-full-buffer")c.callForm=SyncCallForm::ImplicitFullBuffer;
  else if(f=="explicit-range")c.callForm=SyncCallForm::ExplicitRange;
  else return fail(e,"unsupported query invocation form");
  if(b=="minimal-api")c.observerBoundary=SyncObserverBoundary::MinimalAPI;
  else if(b=="detailed-compatibility")c.observerBoundary=SyncObserverBoundary::DetailedCompatibility;
  else if(b=="batch-outer")c.observerBoundary=SyncObserverBoundary::BatchOuter;
  else return fail(e,"unsupported query observation boundary");
  auto runtime=o->getString("runtime_identity"),observer=o->getString("observer_identity");
  if(!runtime || runtime->empty() || !observer || observer->empty())
    return fail(e,"query requires explicit runtime and observer identities");
  c.runtimeIdentity=runtime->str();c.observerIdentity=observer->str();return true;
}
bool replay(const J &record,const SyncAPICostModel *model,J &result,std::string &error) {
  const auto *o=record.getAsObject();if(!o || o->getString("schema")!="orbit.sync-observation.stream.v14")
    return fail(error,"unsupported observed stream schema");
  auto sample=o->getString("sample_key");auto *events=o->getArray("events"),*queries=o->getArray("queries");
  if(!sample || sample->empty() || !events || !queries)return fail(error,"record requires sample_key/events/queries");
  SyncState state;std::map<std::pair<std::string,uint64_t>,int64_t> extents;
  auto registration=[&](const O *r) {
    if(!r)return fail(error,"missing registration object");
    auto bo=r->getString("bo");auto gen=r->getInteger("generation"),bytes=r->getInteger("total_bytes");
    auto *copies=r->getArray("copies");std::vector<size_t> aliases;
    if(!bo || !gen || *gen<0 || !bytes || !copies)return fail(error,"invalid BO registration fields");
    for(const auto &v:*copies){auto i=v.getAsInteger();if(!i || *i<0)return fail(error,"invalid copy alias");aliases.push_back(*i);}
    if(!state.registerPhysicalBO(bo->str(),*gen,*bytes,aliases,error))return false;
    extents[{bo->str(),uint64_t(*gen)}]=*bytes;return true;
  };
  if(auto *list=o->getArray("registrations")) {
    if(o->get("registration"))return fail(error,"ambiguous singular/plural registration");
    for(const auto &v:*list)if(!registration(v.getAsObject()))return false;
  } else if(!registration(o->getObject("registration")))return false;
  std::vector<SyncObservation> ordered;
  for(const auto &v:*events){SyncObservation x;if(!event(v,x,error))return false;ordered.push_back(std::move(x));}
  std::map<uint64_t,std::vector<size_t>> before;
  for(size_t i=0;i<queries->size();++i) {
    auto *q=(*queries)[i].getAsObject();auto id=q?q->getInteger("before_event"):std::nullopt;
    if(!id || *id<=0)return fail(error,"query requires positive before_event");before[*id].push_back(i);
  }
  A answers;size_t answered=0;
  for(const auto &ev:ordered) {
    for(auto qi:before[ev.id]) {
      const auto &source=(*queries)[qi];const auto &q=*source.getAsObject();
      SyncQueryContext c;PhysicalBORange r;
      if(!range(q.getObject("range"),r,error) || !context(q.getObject("context"),c,error))return false;
      SyncQueryResult observed;if(!state.query(r,c,observed,error))return false;
      O answer{{"query_index",int64_t(qi)},{"before_event",int64_t(ev.id)},
               {"source_query",clone(source)},{"query",serializeSyncQuery(observed)}};
      bool complete=true;
      for(auto field:{"site","argument_ordinal","count","observed_extent_bytes","condition_identity"})
        complete &= q.get(field)!=nullptr;
      if(model && complete && ev.kind==SyncObservationKind::SyncBegin) {
        SyncCostRequest request;request.query=observed;request.direction=ev.direction;
        auto site=q.getString("site"),condition=q.getString("condition_identity");
        auto ordinal=q.getInteger("argument_ordinal"),count=q.getInteger("count"),extent=q.getInteger("observed_extent_bytes");
        if(!site || !condition || !ordinal || !count || !extent)return fail(error,"malformed performance context");
        request.site=site->str();request.conditionIdentity=condition->str();request.argumentOrdinal=*ordinal;
        request.count=*count;request.observedExtentBytes=*extent;
        if(r.bo!=ev.range.bo || r.generation!=ev.range.generation || r.offset!=ev.range.offset || r.bytes!=ev.range.bytes ||
           extents.at({r.bo,r.generation})!=*extent)
          return fail(error,"cost query is not the actual sync invocation range/registered BO extent");
        SyncAPICostEvaluation priced;if(!evaluateSyncAPICost(*model,request,priced,error))return false;
        answer["cost"]=serializeSyncAPICostEvaluation(priced);
      } else {
        answer["cost"]=nullptr;answer["cost_status"]="state-only";
        answer["cost_reason"]=!model?"no model supplied":!complete?"performance context incomplete; no site/condition inferred":"query does not precede a sync invocation";
      }
      answers.push_back(std::move(answer));++answered;
    }
    if(!state.observe(ev,error))return false;
  }
  if(answered!=queries->size())return fail(error,"query before_event does not identify a replayed event");
  result=O{{"schema","orbit.sync-observation.replay.v14"},{"sample_key",sample->str()},
    {"origin","observed-operation-stream-replay"},{"timing_used_as_state",false},
    {"event_count",int64_t(ordered.size())},{"query_count",int64_t(answered)},
    {"queries",std::move(answers)}};
  if(auto *scope=o->get("scope"))(*result.getAsObject())["source_scope"]=clone(*scope);
  if(auto *sampleEvidence=o->get("raw_sample"))(*result.getAsObject())["raw_sample"]=clone(*sampleEvidence);
  return true;
}
}
int main(int argc,char **argv) {
  cl::ParseCommandLineOptions(argc,argv,"Observed BO operation replay; no timing-based state\n");
  std::ifstream stream(input);if(!stream){llvm::errs()<<"cannot open input: "<<input<<"\n";return 1;}
  if(llvm::sys::fs::exists(output)){llvm::errs()<<"output must be new: "<<output<<"\n";return 1;}
  SyncAPICostModel model;std::string error;
  if(!modelPath.empty()) {
    auto bytes=llvm::MemoryBuffer::getFile(modelPath);
    if(!bytes || !parseSyncAPICostModel((*bytes)->getBuffer(),model,error)){llvm::errs()<<"model: "<<error<<"\n";return 1;}
  }
  std::error_code ec;llvm::raw_fd_ostream out(output,ec,llvm::sys::fs::CD_CreateNew);
  if(ec){llvm::errs()<<ec.message()<<"\n";return 1;}
  std::string line;uint64_t records=0,lineNumber=0;
  while((!maxRecords || records<maxRecords) && std::getline(stream,line)) {
    ++lineNumber;if(line.empty())continue;
    auto value=llvm::json::parse(line);J result=nullptr;
    if(!value){llvm::consumeError(value.takeError());error="invalid JSON";}
    else if(replay(*value,modelPath.empty()?nullptr:&model,result,error)) {
      llvm::SHA256 sha;sha.update(line);
      (*result.getAsObject())["source_record_sha256"]=llvm::toHex(sha.final(),true);
      if(!modelPath.empty())(*result.getAsObject())["parameter_file_sha256"]=model.parameterFileSHA256;
      out<<result<<"\n";++records;continue;
    }
    llvm::errs()<<"record line "<<lineNumber<<": "<<error<<"\n";return 1;
  }
  if(stream.bad()){llvm::errs()<<"input read failed\n";return 1;}
  out.flush();if(out.has_error()){llvm::errs()<<"output write failed\n";return 1;}
  llvm::outs()<<"replayed "<<records<<" records\n";return 0;
}

#include "Taskflow/SyncState.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
#include <functional>
using namespace orbit::taskflow;
int main(){
 SyncState state;std::string error;unsigned checks=0;bool ok=true;
 auto require=[&](bool value,const char *label){if(value)++checks;else{ok=false;llvm::errs()<<label<<": "<<error<<"\n";}};
 PhysicalBORange full{"physical-bo",0,0,64},left{"physical-bo",0,0,32},right{"physical-bo",0,32,32};
 SyncQueryContext context;context.callForm=SyncCallForm::ExplicitRange;context.observerBoundary=SyncObserverBoundary::MinimalAPI;
 context.runtimeIdentity="CPU-state-test";context.observerIdentity="synthetic ordered observations, no measurements";
 SyncQueryResult result;
 require(state.registerPhysicalBO(full.bo,0,64,{2,7},error),"register one physical allocation with two logical-copy aliases");
 auto query=[&](const PhysicalBORange &range,const char *key){bool queried=state.query(range,context,result,error);require(queried,"bounded query");if(queried)require(result.state==key,"observable state key");};
 query(full,"unknown");require(!result.known && result.copies==std::vector<size_t>({2,7}),"new allocation unknown; aliases do not fabricate versions");
 auto event=[&](uint64_t id,SyncObservationKind kind,const PhysicalBORange &range){SyncObservation e;e.id=id;e.kind=kind;e.range=range;return e;};
 auto apply=[&](SyncObservation e){require(state.observe(e,error),"ordered event accepted");};
 apply(event(1,SyncObservationKind::HostWrite,full));query(full,"host-write-observed");
 auto to=event(2,SyncObservationKind::SyncBegin,full);to.operation="to-1";apply(to);
 query(full,"unknown");require(result.pending && !result.known,"begin alone never proves sync completion");
 auto conflict=event(3,SyncObservationKind::HostWrite,left);
 require(!state.observe(conflict,error),"host write cannot race outstanding sync");
 to.id=3;to.kind=SyncObservationKind::SyncComplete;apply(to);query(full,"post-TO-no-access");
 require(result.known && result.segments.front().deviceVisible,"completed TO establishes declared visibility");
 auto fullForm=context;fullForm.callForm=SyncCallForm::ImplicitFullBuffer;
 require(state.query(full,fullForm,result,error) && result.state=="post-TO-no-access","invocation form does not mutate state");
 require(!state.query(left,fullForm,result,error),"implicit full-buffer call cannot pretend to be a partial range");
 auto detailed=context;detailed.observerBoundary=SyncObserverBoundary::DetailedCompatibility;
 require(state.query(full,detailed,result,error) && result.state=="post-TO-no-access" &&
         result.context.observerBoundary==SyncObserverBoundary::DetailedCompatibility,
         "observer boundary is explicit metadata rather than timing-based state");
 auto missing=context;missing.callForm.reset();require(!state.query(full,missing,result,error),"query requires invocation form");
 missing=context;missing.observerBoundary.reset();require(!state.query(full,missing,result,error),"query requires observation boundary");
 apply(event(4,SyncObservationKind::HostRead,left));query(left,"host-read-since-sync");query(right,"post-TO-no-access");
 query(full,"unknown");require(!result.homogeneous && result.segments.size()==2,"partially accessed BO cannot inherit one homogeneous state");
 apply(event(5,SyncObservationKind::HostWrite,right));query(right,"host-write-since-sync");
 SyncObservation command;command.id=6;command.kind=SyncObservationKind::DeviceSubmit;command.operation="read-left";
 command.accesses={{left,true,false}};apply(command);query(left,"unknown");
 require(result.pending,"submitted read is still pending until observed wait");
 auto writeConflict=event(7,SyncObservationKind::HostWrite,left);require(!state.observe(writeConflict,error),"host write cannot race device read");
 SyncObservation waited;waited.id=7;waited.kind=SyncObservationKind::WaitSuccess;waited.operation="read-left";apply(waited);
 query(left,"after-device-read-complete");
 command.id=8;command.operation="write-right";command.accesses={{right,false,true}};apply(command);
 waited.id=9;waited.operation="write-right";apply(waited);query(right,"after-device-write-complete");
 require(result.known && !result.segments.front().hostVisible && result.segments.front().deviceVisible,
         "completed declared device write is not host-visible before FROM");
 auto from=event(10,SyncObservationKind::SyncBegin,right);from.operation="from-right";from.direction=SyncDirection::FromDevice;apply(from);
 auto wrong=from;wrong.id=11;wrong.kind=SyncObservationKind::SyncComplete;wrong.direction=SyncDirection::ToDevice;
 require(!state.observe(wrong,error),"completion must match actual sync direction");
 from.id=11;from.kind=SyncObservationKind::SyncComplete;apply(from);query(right,"post-FROM-no-access");
 require(result.segments.front().hostVisible,"successful FROM makes completed device version host-visible");
 command.id=12;command.operation="failed-command";command.accesses={{left,true,true}};apply(command);
 waited.id=13;waited.operation="failed-command";waited.kind=SyncObservationKind::WaitFailure;apply(waited);
 query(left,"unknown");query(right,"post-FROM-no-access");
 require(result.known,"failed partial command invalidates only declared overlapping ranges");
 apply(event(14,SyncObservationKind::HostWrite,left));query(left,"host-write-since-sync");
 apply(event(15,SyncObservationKind::UnknownMutation,right));query(right,"unknown");
 require(!state.observe(event(15,SyncObservationKind::HostWrite,left),error),"observation IDs cannot repeat");
 auto orphan=event(16,SyncObservationKind::SyncComplete,left);orphan.operation="never-began";
 require(!state.observe(orphan,error),"orphan sync completion rejected");
 SyncObservation orphanWait;orphanWait.id=16;orphanWait.kind=SyncObservationKind::WaitSuccess;orphanWait.operation="never-submitted";
 require(!state.observe(orphanWait,error),"orphan wait completion rejected");
 require(!state.query({full.bo,0,63,2},context,result,error),"range beyond extent rejected");
 require(!state.query({full.bo,0,1,INT64_MAX},context,result,error),"range arithmetic overflow rejected without addition");
 require(!state.query({full.bo,0,0,0},context,result,error),"empty range rejected");
 require(!state.registerPhysicalBO(full.bo,0,64,{},error),"generation cannot silently reuse prior state");
 require(state.registerPhysicalBO(full.bo,1,64,{9},error),"new allocation generation starts a new state");
 require(!state.query(full,context,result,error),"retired generation is not a live physical BO");
 PhysicalBORange fresh{full.bo,1,0,64};query(fresh,"unknown");require(result.copies==std::vector<size_t>({9}),"new generation does not retain prior logical aliases");
 require(!state.registerPhysicalBO("duplicate-alias",0,64,{1,1},error),"duplicate copy aliases rejected");
 require(!state.registerPhysicalBO("empty",0,0,{},error),"zero physical extent rejected");
 // Unknown origin is not repaired merely by observing an API completion.
 SyncState unknown;require(unknown.registerPhysicalBO("uninitialized",0,32,{},error),"independent unknown BO");
 PhysicalBORange raw{"uninitialized",0,0,32};
 auto rawFrom=event(1,SyncObservationKind::SyncBegin,raw);rawFrom.operation="raw-from";rawFrom.direction=SyncDirection::FromDevice;
 require(unknown.observe(rawFrom,error),"raw FROM begin observed without invented device producer");
 rawFrom.id=2;rawFrom.kind=SyncObservationKind::SyncComplete;require(unknown.observe(rawFrom,error),"raw FROM API completed");
 require(unknown.query(raw,context,result,error) && !result.known,"raw FROM completion does not invent known device contents");
 auto exported=serializeSyncQuery(result);require(exported.getAsObject()->getString("call_form")=="explicit-range" &&
  exported.getAsObject()->getString("observation_boundary")=="minimal-api","query JSON preserves cost-domain boundary and invocation form");
 llvm::outs()<<"{\"status\":\""<<(ok?"passed":"failed")<<"\",\"checks\":"<<checks
             <<",\"scope\":\"synthetic ordered BO observations only; no timing, cache or device execution proof\"}\n";return ok?0:1;
}

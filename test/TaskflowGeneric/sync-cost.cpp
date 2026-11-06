#include "Taskflow/SyncCost.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <functional>
#include <limits>
using namespace orbit::taskflow;
namespace {
std::string text(const llvm::json::Value &v) {
  std::string s;llvm::raw_string_ostream out(s);out<<v;return out.str();
}
}
int main() {
  std::string error;unsigned checks=0;bool ok=true;
  auto require=[&](bool value,const char *label){if(value)++checks;else {ok=false;llvm::errs()<<label<<": "<<error<<"\n";}};
  PhysicalBORange range{"bo",1,0,1024};SyncState engine;
  require(engine.registerPhysicalBO("bo",1,1024,{3,7},error),"register explicit BO extent");
  SyncObservation event;event.id=1;event.kind=SyncObservationKind::HostWrite;event.range=range;
  require(engine.observe(event,error),"real host write history");
  event.id=2;event.kind=SyncObservationKind::SyncBegin;event.operation="first-to";
  require(engine.observe(event,error),"begin first TO");
  event.id=3;event.kind=SyncObservationKind::SyncComplete;
  require(engine.observe(event,error),"complete first TO");
  SyncCostRequest request;request.site="prebound-contiguous-sync-loop";
  request.conditionIdentity="contexts=2;allocation=fresh-standalone";
  request.count=10;request.observedExtentBytes=1024;
  SyncQueryContext context;context.callForm=SyncCallForm::ImplicitFullBuffer;
  context.observerBoundary=SyncObserverBoundary::BatchOuter;
  context.runtimeIdentity="unit-runtime";context.observerIdentity="unit-batch-observer";
  require(engine.query(range,context,request.query,error) && request.query.known &&
          request.query.state=="post-TO-no-access","query is real known post-TO state");
  SyncAPICostModel seed;seed.modelId="synthetic-unit-not-measured";
  seed.parameterFileSHA256=std::string(64,'a');SyncCostRow row;
  row.domain.site=request.site;row.domain.conditionIdentity=request.conditionIdentity;
  row.domain.callForm="whole-implicit";row.domain.observerBoundary="batch-outer";
  row.domain.direction="TO_DEVICE";row.domain.runtimeIdentity=context.runtimeIdentity;
  row.domain.observerIdentity=context.observerIdentity;row.domain.allowedStates={"post-TO-no-access"};
  row.domain.minBytes=64;row.domain.maxBytes=4096;row.domain.minCount=1;row.domain.maxCount=100;
  row.domain.alignmentBytes=64;row.formula="batch-entry-plus-cachelines";
  row.entryNs=100;row.callNs=20;row.cachelineNs=2;
  row.provenance="synthetic-unit-test-coefficients-not-calibration";row.rawManifestSHA256=std::string(64,'b');
  seed.rows={row};auto json=text(serializeSyncAPICostModel(seed));SyncAPICostModel model;
  require(parseSyncAPICostModel(json,model,error),"parse exact-byte-bound model");
  require(model.parameterFileSHA256.size()==64 && model.parameterFileSHA256!=seed.parameterFileSHA256,
          "identity derives from bytes not declared model label");
  SyncAPICostEvaluation evaluation;
  require(evaluateSyncAPICost(model,request,evaluation,error) && evaluation.durationNs &&
          *evaluation.durationNs==620 && evaluation.status=="predicted","entry + count*(call + 16 API-size lines)");
  auto unknown=[&](SyncCostRequest r,const char *label) {
    SyncAPICostEvaluation result;require(evaluateSyncAPICost(model,r,result,error) &&
      !result.durationNs && result.status=="unknown" && !result.reason.empty(),label);
  };
  auto malformed=[&](SyncCostRequest r,const char *label) {
    SyncAPICostEvaluation result;require(!evaluateSyncAPICost(model,r,result,error),label);
  };
  auto r=request;r.query.known=false;unknown(r,"unknown state is not zero");
  r=request;r.query.homogeneous=false;unknown(r,"mixed state is unpriced");
  r=request;r.query.pending=true;unknown(r,"pending operation is unpriced");
  r=request;r.query.state="host-write-observed";unknown(r,"uncalibrated known state is unpriced");
  r=request;r.conditionIdentity.clear();unknown(r,"undeclared execution conditions are unpriced");
  r=request;r.conditionIdentity="contexts=2;allocation=reused-prepared-bo";unknown(r,"fresh/reused condition mismatch is unpriced");
  r=request;r.query.context.runtimeIdentity="different-runtime";unknown(r,"runtime mismatch is unpriced");
  r=request;r.query.context.observerIdentity="different-observer";unknown(r,"observer implementation mismatch is unpriced");
  r=request;r.query.context.observerBoundary=SyncObserverBoundary::MinimalAPI;unknown(r,"minimal versus batch boundary never silently equated");
  r=request;r.site="some-other-site";unknown(r,"site mismatch is unpriced");
  r=request;r.argumentOrdinal=0;unknown(r,"argument ordinal mismatch is unpriced");
  r=request;r.count=101;unknown(r,"count outside frozen range is unpriced");
  r=request;r.direction=SyncDirection::FromDevice;unknown(r,"direction mismatch is unpriced");
  r=request;r.observedExtentBytes=4096;r.query.range.bytes=4096;r.query.segments.front().range.bytes=4096;
  require(evaluateSyncAPICost(model,r,evaluation,error) && evaluation.durationNs && *evaluation.durationNs==1580,
          "size dependence is API formula not whole-program work");
  r.observedExtentBytes=8192;r.query.range.bytes=8192;r.query.segments.front().range.bytes=8192;
  unknown(r,"bytes outside calibrated domain are unpriced");
  r=request;r.observedExtentBytes=0;malformed(r,"observed extent mandatory");
  r=request;r.observedExtentBytes=1023;malformed(r,"query cannot exceed actual extent");
  r=request;r.count=0;malformed(r,"nonpositive count invalid");
  r=request;r.query.context.callForm.reset();malformed(r,"call form mandatory");
  r=request;r.query.context.observerBoundary.reset();malformed(r,"boundary mandatory");
  r=request;r.query.segments.clear();unknown(r,"known flag without segment evidence is unpriced");
  r=request;r.query.segments.front().range.bytes=512;unknown(r,"incomplete range evidence is unpriced");
  r=request;r.query.segments.front().known=false;unknown(r,"unknown evidence cannot be priced by outer known flag");
  r=request;r.query.context.callForm=SyncCallForm::ExplicitRange;
  require(evaluateSyncAPICost(model,r,evaluation,error) && !evaluation.durationNs &&
          evaluation.callForm=="whole-explicit","whole explicit separated from implicit");
  r.query.range.bytes=512;r.query.segments.front().range.bytes=512;
  require(evaluateSyncAPICost(model,r,evaluation,error) && !evaluation.durationNs &&
          evaluation.callForm=="region-explicit","region explicit separated from whole explicit");
  r.query.context.callForm=SyncCallForm::ImplicitFullBuffer;malformed(r,"implicit full cannot masquerade as a region");
  auto reject=[&](SyncAPICostModel m,const char *label) {
    SyncAPICostModel ignored;require(!parseSyncAPICostModel(text(serializeSyncAPICostModel(m)),ignored,error),label);
  };
  auto bad=seed;bad.rows.front().entryNs=-1;reject(bad,"negative calibration rejected");
  bad=seed;bad.rows.front().cachelineNs=std::numeric_limits<double>::infinity();reject(bad,"nonfinite coefficient rejected");
  bad=seed;bad.rows.front().rawManifestSHA256="wrong";reject(bad,"raw provenance digest required");
  bad=seed;bad.rows.front().domain.conditionIdentity.clear();reject(bad,"calibration conditions mandatory");
  bad=seed;bad.rows.front().domain.maxBytes=0;reject(bad,"reversed byte range rejected");
  bad=seed;bad.rows.front().domain.alignmentBytes=0;reject(bad,"alignment positive");
  bad=seed;bad.rows.front().domain.alignmentBytes=1;reject(bad,"batch line formula cannot assume aligned start from byte alignment one");
  bad=seed;bad.rows.front().domain.allowedStates={"unknown"};reject(bad,"unknown cannot become calibrated state");
  bad=seed;bad.rows.front().domain.allowedStates={"post-TO-no-access","post-TO-no-access"};reject(bad,"duplicate state rejected");
  bad=seed;bad.rows.push_back(bad.rows.front());reject(bad,"overlapping duplicate rows rejected");
  bad=seed;bad.rows.front().formula="compute-and-DMA";reject(bad,"opaque cost cannot enter sync formula");
  bad=seed;bad.rows.front().domain.site="unregistered-batch";reject(bad,"batch site bound");
  bad=seed;bad.rows.front().domain.allowedStates={"host-write-observed"};reject(bad,"initial batch fit cannot generalize to first-TO state");
  SyncAPICostModel whitespace;
  require(parseSyncAPICostModel(json+"\n",whitespace,error) && whitespace.parameterFileSHA256!=model.parameterFileSHA256,
          "exact parameter file bytes including whitespace bound separately");
  auto fixed=row;fixed.formula="fixed-api-site";fixed.entryNs=0;fixed.cachelineNs=0;fixed.callNs=500;
  fixed.domain.site="installed-bound-bo-input-loop";fixed.domain.argumentOrdinal=2;
  fixed.domain.observerBoundary="detailed-compatibility";fixed.domain.minCount=1;fixed.domain.maxCount=1;
  fixed.domain.allowedStates={"post-TO-no-access"};seed.rows={fixed};
  require(parseSyncAPICostModel(text(serializeSyncAPICostModel(seed)),model,error),"fixed installed API site model accepted");
  r=request;r.site=fixed.domain.site;r.argumentOrdinal=2;r.count=1;
  r.query.context.observerBoundary=SyncObserverBoundary::DetailedCompatibility;
  require(evaluateSyncAPICost(model,r,evaluation,error) && evaluation.durationNs && *evaluation.durationNs==500,
          "fixed API row has no forced byte slope");
  auto larger=r;larger.query.range.bytes=2048;larger.observedExtentBytes=2048;larger.query.segments.front().range.bytes=2048;
  require(evaluateSyncAPICost(model,larger,evaluation,error) && evaluation.durationNs && *evaluation.durationNs==500,
          "fixed site size invariant inside its explicitly calibrated domain");
  bad=seed;bad.rows.front().cachelineNs=1;reject(bad,"fixed site cannot hide byte charges");
  bad=seed;bad.rows.front().domain.maxCount=2;reject(bad,"fixed observed site is a single call");
  bad=seed;bad.rows.front().domain.site="installed-bound-bo-readback-loop";reject(bad,"readback cannot be TO");
  auto from=fixed;from.domain.site="installed-bound-bo-readback-loop";from.domain.direction="FROM_DEVICE";
  from.domain.allowedStates={"after-device-write-complete"};seed.rows={from};
  require(parseSyncAPICostModel(text(serializeSyncAPICostModel(seed)),model,error),"explicit device-produced FROM model accepted");
  event={};event.id=4;event.kind=SyncObservationKind::DeviceSubmit;event.operation="device-writer";
  event.accesses={{range,false,true}};require(engine.observe(event,error),"source-declared successful writer submitted");
  event.id=5;event.kind=SyncObservationKind::WaitSuccess;require(engine.observe(event,error),"writer wait completes");
  context.observerBoundary=SyncObserverBoundary::DetailedCompatibility;
  r=request;r.site=from.domain.site;r.argumentOrdinal=2;r.count=1;r.direction=SyncDirection::FromDevice;
  require(engine.query(range,context,r.query,error),"actual state engine provides producer evidence");
  require(evaluateSyncAPICost(model,r,evaluation,error) && evaluation.durationNs && *evaluation.durationNs==500,
          "FROM calibrated only with successful device producer");
  r.query.segments.front().deviceWriteEpoch=0;unknown(r,"FROM with removed producer evidence stays unknown");
  // A historical device producer is insufficient after a new host write. The
  // state engine deliberately retains history; pricing must test current origin.
  event={};event.id=6;event.kind=SyncObservationKind::SyncBegin;event.range=range;
  event.direction=SyncDirection::FromDevice;event.operation="first-from";
  require(engine.observe(event,error),"begin FROM current device producer");
  event.id=7;event.kind=SyncObservationKind::SyncComplete;
  require(engine.observe(event,error),"complete FROM current device producer");
  auto batchFrom=row;batchFrom.domain.direction="FROM_DEVICE";
  batchFrom.domain.allowedStates={"post-FROM-no-access"};seed.rows={batchFrom};
  require(parseSyncAPICostModel(text(serializeSyncAPICostModel(seed)),model,error),"post-FROM batch calibration has explicit current-producer guard");
  r=request;r.direction=SyncDirection::FromDevice;
  require(engine.query(range,request.query.context,r.query,error),"query current device-origin post-FROM history");
  require(evaluateSyncAPICost(model,r,evaluation,error) && evaluation.durationNs && *evaluation.durationNs==620,
          "post-FROM with current successful device producer remains supported");
  event={};event.id=8;event.kind=SyncObservationKind::HostWrite;event.range=range;
  require(engine.observe(event,error),"host overwrites old device-produced contents");
  event.id=9;event.kind=SyncObservationKind::SyncBegin;event.operation="host-new-to";
  require(engine.observe(event,error),"begin TO new host contents");
  event.id=10;event.kind=SyncObservationKind::SyncComplete;
  require(engine.observe(event,error),"complete TO new host contents");
  event.id=11;event.kind=SyncObservationKind::SyncBegin;event.operation="host-new-from";event.direction=SyncDirection::FromDevice;
  require(engine.observe(event,error),"begin FROM new host contents");
  event.id=12;event.kind=SyncObservationKind::SyncComplete;
  require(engine.observe(event,error),"complete FROM new host contents");
  require(engine.query(range,request.query.context,r.query,error) && r.query.known && r.query.state=="post-FROM-no-access" &&
          r.query.segments.front().deviceWriteEpoch==5 && r.query.segments.front().contentEpoch==8,
          "real history retains stale device epoch but current content is host-produced");
  unknown(r,"stale historical producer cannot price host-produced post-FROM contents");
  const auto serialized=serializeSyncAPICostEvaluation(evaluation);const auto *object=serialized.getAsObject();
  require(object && object->get("physical_bytes") && object->get("physical_bytes")->getAsNull() &&
      object->getBoolean("complete_program_cost")==false,"scope excludes physical traffic and full program cost");
  llvm::outs()<<"{\"passed\":"<<(ok?"true":"false")<<",\"checks\":"<<checks<<"}\n";
  return ok?0:1;
}

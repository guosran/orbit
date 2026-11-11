#include "Taskflow/ConstraintSearch.h"
#include "Taskflow/NPUConstraintSchedule.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <stdexcept>
using namespace orbit::taskflow;
namespace j=llvm::json;
namespace {
int64_t integer(const j::Object &o,llvm::StringRef key,int64_t fallback=0){return o.getInteger(key).value_or(fallback);}
std::vector<int64_t> integers(const j::Array *a){std::vector<int64_t> result;if(a)for(auto &v:*a){auto i=v.getAsInteger();if(!i)throw std::runtime_error("integer array required");result.push_back(*i);}return result;}
const j::Object &object(const j::Value &v){auto *o=v.getAsObject();if(!o)throw std::runtime_error("object required");return *o;}
j::Object telemetry(const ConstraintSearchTelemetry &t){return j::Object{{"model_build_ms",t.modelBuildMs},{"solve_ms",t.solveMs},{"total_ms",t.totalMs},{"variables",t.variables},{"constraints",t.constraints},{"conflicts",t.conflicts},{"branches",t.branches},{"peak_rss_kib",t.peakRssKiB},{"objective_ns",t.hasObjective?j::Value(t.objectiveNs):j::Value(nullptr)},{"best_bound_ns",t.hasObjective?j::Value(t.bestBoundNs):j::Value(nullptr)},{"relative_gap",t.hasObjective?j::Value(t.relativeGap):j::Value(nullptr)},{"budget_exhausted",t.budgetExhausted},{"objective_exact_for_backend",false}};}
}
int main(){
 auto bytes=llvm::MemoryBuffer::getSTDIN();if(!bytes){llvm::errs()<<"Unable to read stdin\n";return 1;}
 auto parsed=j::parse((*bytes)->getBuffer());if(!parsed){llvm::errs()<<llvm::toString(parsed.takeError())<<"\n";return 1;}
 try {
 const auto &o=object(*parsed);ConstraintSearchOptions options;
 if(auto *s=o.getObject("options")){options.wallSeconds=s->getNumber("time_limit_seconds").value_or(1);options.workers=integer(*s,"workers",1);options.randomSeed=integer(*s,"random_seed",0);options.maxVariables=integer(*s,"max_variables",100000);options.maxConstraints=integer(*s,"max_constraints",500000);options.maxPatterns=integer(*s,"max_patterns",20000);options.maxTasks=integer(*s,"max_tasks",512);options.maxTiles=integer(*s,"max_tiles",16);options.maxInputEdges=integer(*s,"max_input_edges",65536);options.maxEdges=integer(*s,"max_edges",8192);options.maxMemberships=integer(*s,"max_memberships",65536);options.maxIntersectionWork=integer(*s,"max_intersection_work",2000000);options.maxPatternBytes=integer(*s,"max_pattern_bytes",INT64_MAX);options.maxPatternInputs=integer(*s,"max_pattern_inputs",INT64_MAX);options.maxPatternOutputs=integer(*s,"max_pattern_outputs",INT64_MAX);options.hintPatternIndices=integers(s->getArray("hint_pattern_indices"));}
 j::Object output;
 if(o.getString("kind").value_or("")=="npu-schedule"){
  NPUScheduleProblem p;p.exclusiveTiles=o.getBoolean("exclusive_tiles").value_or(true);
  if(auto *a=o.getArray("dma_resources"))for(auto &v:*a){const auto &x=object(v);p.dmaResources.push_back({integer(x,"id",p.dmaResources.size()),integer(x,"max_bds")});}
  if(auto *a=o.getArray("tiles"))for(auto &v:*a){const auto &x=object(v);NPUPhysicalTile t;t.id=integer(x,"id",p.tiles.size());t.column=integer(x,"column");t.row=integer(x,"row",2);t.staticBytesAvailable=integer(x,"static_bytes");t.inputDmaResource=integer(x,"input_resource",-1);t.outputDmaResource=integer(x,"output_resource",-1);t.inputChannel=integer(x,"input_channel");t.outputChannel=integer(x,"output_channel",1);p.tiles.push_back(t);}
  if(auto *a=o.getArray("shards"))for(auto &v:*a){const auto &x=object(v);NPUIndependentShard s;s.id=integer(x,"id",p.shards.size());s.inputNs=integer(x,"input_ns");s.computeNs=integer(x,"compute_ns");s.outputNs=integer(x,"output_ns");s.staticBytes=integer(x,"static_bytes");s.inputBdCount=integer(x,"input_bds");s.outputBdCount=integer(x,"output_bds");s.candidateTileIndices=integers(x.getArray("candidate_tiles"));p.shards.push_back(s);}
  auto r=solveNPUConstraintSchedule(p,options);output=telemetry(r.telemetry);output["status"]=constraintSearchStatusName(r.status);output["message"]=r.message;output["makespan_ns"]=r.telemetry.hasObjective?j::Value(r.makespanNs):j::Value(nullptr);j::Array schedule;for(auto s:r.shards)schedule.push_back(j::Object{{"shard",s.shardIndex},{"tile",s.tileIndex},{"column",s.column},{"row",s.row},{"input_channel",s.inputChannel},{"output_channel",s.outputChannel},{"input_start_ns",s.inputStartNs},{"input_end_ns",s.inputEndNs},{"compute_start_ns",s.computeStartNs},{"compute_end_ns",s.computeEndNs},{"output_start_ns",s.outputStartNs},{"output_end_ns",s.outputEndNs}});output["schedule"]=std::move(schedule);
 }else{
  ConstraintSearchProblem p;p.taskCount=integer(o,"tasks");p.tileCount=integer(o,"tiles");p.tileBaseCostNs=integers(o.getArray("tile_base_cost_ns"));
  if(auto *a=o.getArray("patterns"))for(auto &v:*a){const auto &x=object(v);ConstraintPattern pattern;pattern.id=p.patterns.size();pattern.tasks=integers(x.getArray("tasks"));pattern.exclusiveStorageIds=integers(x.getArray("exclusive_storages"));pattern.tileIndex=integer(x,"tile");pattern.costNs=integer(x,"cost_ns");pattern.bytes=integer(x,"bytes");pattern.inputCount=integer(x,"inputs");pattern.outputCount=integer(x,"outputs");p.patterns.push_back(std::move(pattern));}
  if(auto *a=o.getArray("edges"))for(auto &v:*a){auto e=integers(v.getAsArray());if(e.size()!=2)throw std::runtime_error("edge must have two integers");p.edges.push_back({e[0],e[1]});}
  auto r=solveConstraintSearch(p,options);output=telemetry(r.telemetry);output["status"]=constraintSearchStatusName(r.status);output["message"]=r.message;output["selected_tile"]=r.selectedTileIndex;j::Array selected;for(auto i:r.selectedPatternIndices)selected.push_back(i);output["selected_pattern_indices"]=std::move(selected);
 }
 output["available"]=constraintSearchAvailable();llvm::outs()<<llvm::formatv("{0:2}\n",j::Value(std::move(output)));return 0;
 }catch(const std::exception &e){llvm::outs()<<llvm::formatv("{0:2}\n",j::Value(j::Object{{"status","INVALID_INPUT"},{"message",e.what()}}));return 0;}
}

#include "Taskflow/NPUConstraintSchedule.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <sys/resource.h>
#ifdef ORBIT_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"
#endif
namespace orbit::taskflow {
NPUScheduleResult solveNPUConstraintSchedule(const NPUScheduleProblem &p,const ConstraintSearchOptions &o){
 using Clock=std::chrono::steady_clock;auto start=Clock::now();NPUScheduleResult r;
 auto ms=[&](auto t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
 auto finish=[&](ConstraintSearchStatus s,std::string message){r.status=s;r.message=std::move(message);r.telemetry.totalMs=ms(start);rusage u{};if(!getrusage(RUSAGE_SELF,&u))r.telemetry.peakRssKiB=u.ru_maxrss;return r;};
#ifndef ORBIT_HAS_ORTOOLS
 (void)p;(void)o;return finish(ConstraintSearchStatus::Unavailable,"Built without ORBIT_ENABLE_ORTOOLS");
#else
 using namespace operations_research;using namespace operations_research::sat;
 if(p.shards.empty()||p.tiles.empty()||p.dmaResources.empty()||!std::isfinite(o.wallSeconds)||o.wallSeconds<=0||o.workers<1||o.maxVariables<1||o.maxConstraints<1||o.maxPatterns<1||o.maxTasks<1||o.maxTiles<1)return finish(ConstraintSearchStatus::InvalidInput,"Invalid scheduling dimensions or budget");
 if(p.shards.size()>size_t(o.maxTasks)||p.tiles.size()>size_t(o.maxTiles)||p.dmaResources.size()>size_t(o.maxTiles)*2)return finish(ConstraintSearchStatus::ModelLimit,"Scheduling input dimension cap exceeded");
 constexpr int64_t limit=INT64_MAX/16;
 auto valid=[&](int64_t x){return x>=0&&x<=limit;};
 auto timeout=[&](){r.telemetry.budgetExhausted=true;r.telemetry.modelBuildMs=ms(start);return finish(ConstraintSearchStatus::Unknown,"Construction deadline");};
 auto expired=[&](){return ms(start)>=o.wallSeconds*1000;};
 std::set<int64_t> ids;for(auto x:p.dmaResources)if(!valid(x.maxStaticBds)||!ids.insert(x.id).second)return finish(ConstraintSearchStatus::InvalidInput,"Invalid DMA capacity/id");
 ids.clear();std::set<std::pair<int64_t,int64_t>> coords;
 for(auto t:p.tiles)if(t.column<0||t.row<0||!valid(t.staticBytesAvailable)||t.inputChannel<0||t.outputChannel<0||t.inputDmaResource<0||t.outputDmaResource<0||size_t(t.inputDmaResource)>=p.dmaResources.size()||size_t(t.outputDmaResource)>=p.dmaResources.size()||!ids.insert(t.id).second||!coords.emplace(t.column,t.row).second)return finish(ConstraintSearchStatus::InvalidInput,"Invalid physical tile/resource/coordinate");
 int64_t horizon=0,placements=0,sumBytes=0,sumBds=0;ids.clear();
 for(const auto &s:p.shards){
  if(!ids.insert(s.id).second||!valid(s.inputNs)||!valid(s.computeNs)||!valid(s.outputNs)||!valid(s.staticBytes)||!valid(s.inputBdCount)||!valid(s.outputBdCount))return finish(ConstraintSearchStatus::InvalidInput,"Invalid shard quantities/id");
  for(auto d:{s.inputNs,s.computeNs,s.outputNs}){if(d>limit-horizon)return finish(ConstraintSearchStatus::InvalidInput,"Unsafe horizon");horizon+=d;}
  if(s.staticBytes>limit-sumBytes||s.inputBdCount>limit-sumBds)return finish(ConstraintSearchStatus::InvalidInput,"Unsafe resource coefficient sum");sumBytes+=s.staticBytes;sumBds+=s.inputBdCount;if(s.outputBdCount>limit-sumBds)return finish(ConstraintSearchStatus::InvalidInput,"Unsafe BD sum");sumBds+=s.outputBdCount;
  if(s.candidateTileIndices.empty())return finish(ConstraintSearchStatus::Infeasible,"A shard has no supported tile");
  std::set<int64_t> candidates;for(auto t:s.candidateTileIndices)if(t<0||size_t(t)>=p.tiles.size()||!candidates.insert(t).second)return finish(ConstraintSearchStatus::InvalidInput,"Invalid candidate tile");
  if(s.candidateTileIndices.size()>size_t(o.maxPatterns)||placements>o.maxPatterns-int64_t(s.candidateTileIndices.size()))return finish(ConstraintSearchStatus::ModelLimit,"Placement cap exceeded");placements+=s.candidateTileIndices.size();if(expired())return timeout();
 }
 if(6.L*p.shards.size()+placements+1>o.maxVariables||7.L*p.shards.size()+4.L*placements+3.L*p.tiles.size()+2.L*p.dmaResources.size()+1>o.maxConstraints)return finish(ConstraintSearchStatus::ModelLimit,"Scheduling model preflight limit");
 CpModelBuilder b;
 struct Vars {IntVar is,ie,cs,ce,os,oe;std::vector<BoolVar> choices;Vars(CpModelBuilder &b,int64_t h):is(b.NewIntVar(Domain(0,h))),ie(b.NewIntVar(Domain(0,h))),cs(b.NewIntVar(Domain(0,h))),ce(b.NewIntVar(Domain(0,h))),os(b.NewIntVar(Domain(0,h))),oe(b.NewIntVar(Domain(0,h))){} };
 std::vector<Vars> vars;vars.reserve(p.shards.size());
 std::vector<std::vector<IntervalVar>> cores(p.tiles.size()),dmas(p.dmaResources.size());std::vector<std::vector<BoolVar>> occupants(p.tiles.size());
 std::vector<LinearExpr> bytes(p.tiles.size()),bds(p.dmaResources.size());std::vector<IntVar> ends;
 for(size_t i=0;i<p.shards.size();++i){const auto &s=p.shards[i];vars.emplace_back(b,horizon);auto &v=vars.back();
  b.AddEquality(v.ie,LinearExpr(v.is)+s.inputNs);b.AddEquality(v.ce,LinearExpr(v.cs)+s.computeNs);b.AddEquality(v.oe,LinearExpr(v.os)+s.outputNs);b.AddGreaterOrEqual(v.cs,v.ie);b.AddGreaterOrEqual(v.os,v.ce);ends.push_back(v.oe);
  for(auto ti:s.candidateTileIndices){const auto &t=p.tiles[ti];auto chosen=b.NewBoolVar();v.choices.push_back(chosen);occupants[ti].push_back(chosen);
   if(s.computeNs)cores[ti].push_back(b.NewOptionalIntervalVar(v.cs,s.computeNs,v.ce,chosen));if(s.inputNs)dmas[t.inputDmaResource].push_back(b.NewOptionalIntervalVar(v.is,s.inputNs,v.ie,chosen));if(s.outputNs)dmas[t.outputDmaResource].push_back(b.NewOptionalIntervalVar(v.os,s.outputNs,v.oe,chosen));
   bytes[ti]+=LinearExpr(chosen)*s.staticBytes;bds[t.inputDmaResource]+=LinearExpr(chosen)*s.inputBdCount;bds[t.outputDmaResource]+=LinearExpr(chosen)*s.outputBdCount;
  }b.AddExactlyOne(v.choices);if(expired())return timeout();
 }
 for(size_t t=0;t<p.tiles.size();++t){if(!cores[t].empty())b.AddNoOverlap(cores[t]);if(p.exclusiveTiles&&!occupants[t].empty())b.AddAtMostOne(occupants[t]);b.AddLessOrEqual(bytes[t],p.tiles[t].staticBytesAvailable);}
 for(size_t d=0;d<p.dmaResources.size();++d){if(!dmas[d].empty())b.AddNoOverlap(dmas[d]);b.AddLessOrEqual(bds[d],p.dmaResources[d].maxStaticBds);}
 auto makespan=b.NewIntVar(Domain(0,horizon));b.AddMaxEquality(makespan,ends);b.Minimize(makespan);const CpModelProto proto=b.Build();r.telemetry.variables=proto.variables_size();r.telemetry.constraints=proto.constraints_size();r.telemetry.modelBuildMs=ms(start);
 if(r.telemetry.variables>o.maxVariables||r.telemetry.constraints>o.maxConstraints)return finish(ConstraintSearchStatus::ModelLimit,"Constructed scheduling model limit");if(expired())return timeout();
 SatParameters params;params.set_max_time_in_seconds(o.wallSeconds-ms(start)/1000);params.set_num_search_workers(o.workers);params.set_random_seed(o.randomSeed);params.set_log_search_progress(false);Model model;model.Add(NewSatParameters(params));auto solveStart=Clock::now();auto response=SolveCpModel(proto,&model);r.telemetry.solveMs=ms(solveStart);r.telemetry.branches=response.num_branches();r.telemetry.conflicts=response.num_conflicts();
 if(response.status()==CpSolverStatus::MODEL_INVALID)return finish(ConstraintSearchStatus::InvalidInput,response.solution_info());if(response.status()==CpSolverStatus::INFEASIBLE)return finish(ConstraintSearchStatus::Infeasible,"No physical placement/schedule meets resource capacities");if(response.status()!=CpSolverStatus::FEASIBLE&&response.status()!=CpSolverStatus::OPTIMAL){r.telemetry.budgetExhausted=expired();return finish(ConstraintSearchStatus::Unknown,"No schedule incumbent");}
 r.makespanNs=SolutionIntegerValue(response,makespan);r.telemetry.hasObjective=true;r.telemetry.objectiveNs=r.makespanNs;r.telemetry.bestBoundNs=response.status()==CpSolverStatus::OPTIMAL?double(r.makespanNs):response.best_objective_bound();r.telemetry.relativeGap=std::max(0.,double(r.telemetry.objectiveNs)-r.telemetry.bestBoundNs)/std::max(1.,std::abs(double(r.telemetry.objectiveNs)));r.telemetry.budgetExhausted=response.status()==CpSolverStatus::FEASIBLE&&expired();
 for(size_t i=0;i<vars.size();++i){auto &v=vars[i];for(size_t j=0;j<v.choices.size();++j)if(SolutionBooleanValue(response,v.choices[j])){auto ti=p.shards[i].candidateTileIndices[j];auto &t=p.tiles[ti];NPUShardSchedule s;s.shardIndex=i;s.tileIndex=ti;s.column=t.column;s.row=t.row;s.inputChannel=t.inputChannel;s.outputChannel=t.outputChannel;s.inputStartNs=SolutionIntegerValue(response,v.is);s.inputEndNs=SolutionIntegerValue(response,v.ie);s.computeStartNs=SolutionIntegerValue(response,v.cs);s.computeEndNs=SolutionIntegerValue(response,v.ce);s.outputStartNs=SolutionIntegerValue(response,v.os);s.outputEndNs=SolutionIntegerValue(response,v.oe);r.shards.push_back(s);}}
 return finish(response.status()==CpSolverStatus::OPTIMAL?ConstraintSearchStatus::Optimal:ConstraintSearchStatus::Feasible,"Model schedule requires physical lowering and resource verification");
#endif
}
}

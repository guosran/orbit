#include "Taskflow/ProgramJointSearch.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <sys/resource.h>
#ifdef ORBIT_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#endif
using namespace orbit::taskflow;
ProgramJointSolution orbit::taskflow::solveProgramJointDomain(const ProgramJointDomain &d,const ConstraintSearchOptions &o,const ProgramJointSolution *prepared){
 using Clock=std::chrono::steady_clock;auto start=Clock::now();auto ms=[&](){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
 auto baseline=prepared?*prepared:baselineProgramJointDomain(d);ConstraintSearchTelemetry t;
 if(prepared){ProgramGraph graph;std::string error;if(!materializeProgramJointSolution(d,baseline,graph,error)||!checkProgramJointSolution(d,baseline,graph,error)){baseline.hasPlan=false;baseline.message="supplied baseline rejected: "+error;}}
 auto finish=[&](ProgramJointSolution s){t.totalMs=ms();struct rusage r;if(!getrusage(RUSAGE_SELF,&r))t.peakRssKiB=r.ru_maxrss;s.telemetry=t;return s;};
 auto fallback=[&](ConstraintSearchStatus status,const std::string &message){auto s=baseline;s.status=status;s.message=message;s.selection=s.hasPlan?"baseline":"none";return finish(s);};
#ifndef ORBIT_HAS_ORTOOLS
 (void)o;return fallback(ConstraintSearchStatus::Unavailable,"built without ORBIT_ENABLE_ORTOOLS");
#else
 using namespace operations_research;using namespace operations_research::sat;
 if(!baseline.hasPlan)return fallback(ConstraintSearchStatus::InvalidInput,baseline.message);
 if(!std::isfinite(o.wallSeconds)||o.wallSeconds<=0||o.workers<1||o.workers>16||o.maxVariables<1||o.maxConstraints<1||o.maxPatterns<1||d.maxConfigurations<1)return fallback(ConstraintSearchStatus::InvalidInput,"invalid solver budget/options");
 struct Stop{ConstraintSearchStatus status;std::string error;};
 try{
 CpModelBuilder b;auto checkpoint=[&](){t.variables=b.Proto().variables_size();t.constraints=b.Proto().constraints_size();if(ms()>=o.wallSeconds*1000){t.budgetExhausted=true;throw Stop{ConstraintSearchStatus::Unknown,"deadline exhausted during model construction; checked baseline retained"};}if(b.Proto().variables_size()>o.maxVariables||b.Proto().constraints_size()>o.maxConstraints)throw Stop{ConstraintSearchStatus::ModelLimit,"joint model size limit; checked baseline retained"};};
 const int64_t H=baseline.makespanNs;if(H<1||H>INT64_MAX/128)throw Stop{ConstraintSearchStatus::InvalidInput,"unsafe checked-baseline horizon"};
 const size_t V=d.variants.size(),G=d.groups.size(),P=d.places.size(),K=d.maxConfigurations;
 if(G>size_t(o.maxPatterns)||P>16||K>128||V>8)throw Stop{ConstraintSearchStatus::ModelLimit,"joint domain dimension limit"};
 auto boolean=[&](){checkpoint();return b.NewBoolVar();};auto time=[&](){checkpoint();return b.NewIntVar(Domain(0,H));};
 struct Usage{std::vector<IntervalVar> intervals;std::vector<int64_t> demands;};std::map<std::string,Usage> resources,memories;
 auto reserve=[&](const std::string &id,IntervalVar interval){std::map<std::string,int64_t> demands;std::string error;if(!programResourceDemands(d.target,{{id,1}},demands,error))throw Stop{ConstraintSearchStatus::InvalidInput,error};for(auto &entry:demands){resources[entry.first].intervals.push_back(interval);resources[entry.first].demands.push_back(entry.second);}};
 std::vector<BoolVar> variant,selected,used;for(size_t v=0;v<V;++v){variant.push_back(boolean());b.AddHint(variant.back(),baseline.variant==int64_t(v));}b.AddExactlyOne(variant);
 for(size_t g=0;g<G;++g){selected.push_back(boolean());bool hint=false;for(auto &a:baseline.groups)hint|=a.group==g;b.AddHint(selected.back(),hint);}
 std::vector<IntVar> begin,end,length;std::vector<std::vector<LinearExpr>> completions(K);
 for(size_t k=0;k<K;++k){used.push_back(boolean());begin.push_back(time());end.push_back(time());length.push_back(time());b.AddEquality(length[k],LinearExpr(end[k])-begin[k]);b.AddEquality(begin[k],k?LinearExpr(end[k-1]):LinearExpr(d.submitNs));if(k)b.AddImplication(used[k],used[k-1]);b.AddEquality(end[k],begin[k]).OnlyEnforceIf(used[k].Not());completions[k].push_back(LinearExpr(begin[k])+LinearExpr(used[k])*d.configurationNs);b.AddHint(used[k],k<baseline.configurationEndNs.size());b.AddHint(begin[k],k<baseline.configurationStartNs.size()?baseline.configurationStartNs[k]:baseline.configurationEndNs.back());b.AddHint(end[k],k<baseline.configurationEndNs.size()?baseline.configurationEndNs[k]:baseline.configurationEndNs.back());}
 std::vector<std::vector<std::vector<size_t>>> covers(V),exporters(V);std::vector<std::vector<IntVar>> phases(V),ready(V);std::vector<std::vector<LinearExpr>> phaseExpr(V);
 for(size_t v=0;v<V;++v){auto &x=d.variants[v];size_t N=x.graph.instances.size(),M=x.producers.size();if(N>size_t(o.maxTasks)||x.computeNs.size()!=N||x.readers.size()!=M||x.valueBytes.size()!=M)throw Stop{ConstraintSearchStatus::ModelLimit,"instance/version domain limit"};covers[v].resize(N);exporters[v].resize(M);phaseExpr[v].resize(N);for(size_t i=0;i<N;++i)phases[v].push_back(b.NewIntVar(Domain(0,K-1)));for(size_t value=0;value<M;++value){ready[v].push_back(time());if(x.producers[value]<0)b.AddEquality(ready[v].back(),d.submitNs);}}
 for(size_t g=0;g<G;++g){auto &x=d.groups[g];if(x.variant>=V||x.instances.empty())throw Stop{ConstraintSearchStatus::InvalidInput,"invalid local mode"};for(auto i:x.instances){if(i>=covers[x.variant].size())throw Stop{ConstraintSearchStatus::InvalidInput,"invalid instance reference"};covers[x.variant][i].push_back(g);}for(auto value:x.outputs)exporters[x.variant][value].push_back(g);b.AddImplication(selected[g],variant[x.variant]);}
 struct Route{size_t version,link;BoolVar chosen;IntVar begin,end;};struct Assignment{size_t group,place,configuration;BoolVar chosen;std::vector<IntVar> compute;std::vector<Route> inputs,outputs;};std::vector<Assignment> assignments;
 std::vector<std::vector<BoolVar>> groupChoices(G),configChoices(K);std::vector<LinearExpr> inputCounts(K),outputCounts(K);std::map<std::pair<std::string,size_t>,std::vector<BoolVar>> coreChoices;
 for(size_t g=0;g<G;++g){auto &mode=d.groups[g];auto v=mode.variant;auto &domain=d.variants[v];
  for(auto value:mode.inputs)if(domain.producers[value]>=0){LinearExpr exported;for(auto p:exporters[v][value])exported+=selected[p];b.AddLessOrEqual(selected[g],exported);}
  for(size_t p=0;p<P;++p){int64_t bytes;std::string error;if(!programJointGroupFits(d,g,p,bytes,error))continue;
   for(size_t k=0;k<K;++k){checkpoint();if(assignments.size()>=size_t(o.maxPatterns))throw Stop{ConstraintSearchStatus::ModelLimit,"group/place/configuration assignment limit"};auto x=boolean();Assignment a{g,p,k,x,{},{},{}};groupChoices[g].push_back(x);configChoices[k].push_back(x);coreChoices[{d.places[p].binding.resource,k}].push_back(x);b.AddImplication(x,used[k]);inputCounts[k]+=LinearExpr(x)*int64_t(mode.inputs.size());outputCounts[k]+=LinearExpr(x)*int64_t(mode.outputs.size());
    const ProgramJointSelection *hint=nullptr;for(auto &h:baseline.groups)if(h.group==g&&h.place==p&&h.configuration==k)hint=&h;b.AddHint(x,hint?1:0);
    auto mi=b.NewOptionalIntervalVar(begin[k],length[k],end[k],x);memories[d.places[p].binding.memory].intervals.push_back(mi);memories[d.places[p].binding.memory].demands.push_back(bytes);
    auto routes=[&](bool input){for(auto value:(input?mode.inputs:mode.outputs)){LinearExpr sum;for(auto li:(input?d.places[p].inputLinks:d.places[p].outputLinks)){int64_t duration;std::string ignored;if(!programJointTransferNs(d,v,value,li,duration,ignored))continue;auto r=boolean();auto rb=time(),re=time();sum+=r;auto interval=b.NewOptionalIntervalVar(rb,duration,re,r);b.AddEquality(re,0).OnlyEnforceIf(r.Not());completions[k].push_back(re);b.AddGreaterOrEqual(rb,LinearExpr(begin[k])+d.configurationNs).OnlyEnforceIf(r);b.AddLessOrEqual(re,end[k]).OnlyEnforceIf(r);if(input)b.AddGreaterOrEqual(rb,ready[v][value]).OnlyEnforceIf(r);else b.AddEquality(ready[v][value],re).OnlyEnforceIf(r);reserve(d.target.links[li].resource,interval);(input?a.inputs:a.outputs).push_back({value,li,r,rb,re});const ProgramJointTransfer *ht=nullptr;if(hint)for(auto &h:(input?hint->inputs:hint->outputs))if(h.version==value&&h.link==li)ht=&h;b.AddHint(r,ht?1:0);if(ht){b.AddHint(rb,ht->startNs);b.AddHint(re,ht->endNs);}}b.AddEquality(sum,x);}};routes(true);routes(false);
    std::vector<IntVar> ce;for(size_t i=0;i<mode.instances.size();++i){auto instance=mode.instances[i];auto cb=time(),finish=time();auto interval=b.NewOptionalIntervalVar(cb,domain.computeNs[instance],finish,x);a.compute.push_back(cb);ce.push_back(finish);b.AddEquality(finish,0).OnlyEnforceIf(x.Not());completions[k].push_back(finish);b.AddGreaterOrEqual(cb,LinearExpr(begin[k])+d.configurationNs).OnlyEnforceIf(x);b.AddLessOrEqual(finish,end[k]).OnlyEnforceIf(x);if(i)b.AddGreaterOrEqual(cb,ce[i-1]).OnlyEnforceIf(x);reserve(d.places[p].binding.resource,interval);phaseExpr[v][instance]+=LinearExpr(x)*int64_t(k);if(hint){b.AddHint(cb,hint->computeStartNs[i]);b.AddHint(finish,hint->computeStartNs[i]+domain.computeNs[instance]);}}
    for(auto &r:a.inputs)b.AddGreaterOrEqual(a.compute.front(),r.end).OnlyEnforceIf(r.chosen);for(auto &r:a.outputs)b.AddGreaterOrEqual(r.begin,ce.back()).OnlyEnforceIf(r.chosen);assignments.push_back(std::move(a));
   }
  }
  LinearExpr sum;for(auto x:groupChoices[g])sum+=x;b.AddEquality(sum,selected[g]);
 }
 for(size_t v=0;v<V;++v){for(size_t i=0;i<covers[v].size();++i){LinearExpr sum;for(auto g:covers[v][i])sum+=selected[g];b.AddEquality(sum,variant[v]);b.AddEquality(phases[v][i],phaseExpr[v][i]);}
  std::set<std::pair<size_t,size_t>> raw;for(size_t value=0;value<d.variants[v].producers.size();++value)if(d.variants[v].producers[value]>=0)for(auto reader:d.variants[v].readers[value])raw.emplace(size_t(d.variants[v].producers[value]),reader);
  int64_t intersectionWork=0;for(auto edge:raw){if(edge.first==edge.second)continue;auto &l=covers[v][edge.first],&r=covers[v][edge.second];intersectionWork+=l.size()+r.size();if(intersectionWork>o.maxIntersectionWork)throw Stop{ConstraintSearchStatus::ModelLimit,"RAW membership intersection limit"};std::vector<size_t> common;std::set_intersection(l.begin(),l.end(),r.begin(),r.end(),std::back_inserter(common));LinearExpr together;for(auto g:common)together+=selected[g];b.AddGreaterOrEqual(phases[v][edge.second],LinearExpr(phases[v][edge.first])+variant[v]-together);}
  for(auto edge:d.variants[v].graph.orderEdges){auto pt=d.variants[v].graph.events[edge.producer].task,ct=d.variants[v].graph.events[edge.consumer].task;for(size_t p=0;p<covers[v].size();++p)if(d.variants[v].graph.instances[p].task==pt)for(size_t c=0;c<covers[v].size();++c)if(d.variants[v].graph.instances[c].task==ct)b.AddGreaterOrEqual(phases[v][c],LinearExpr(phases[v][p])+variant[v]);}
 }
 for(auto &entry:coreChoices)b.AddAtMostOne(entry.second);
 for(size_t k=0;k<K;++k){LinearExpr sum;for(auto x:configChoices[k])sum+=x;b.AddGreaterOrEqual(sum,used[k]);b.AddLessOrEqual(inputCounts[k],d.maxConfigurationInputs);b.AddLessOrEqual(outputCounts[k],d.maxConfigurationOutputs);b.AddMaxEquality(end[k],completions[k]);}
 for(auto &entry:resources){auto r=programResource(d.target,entry.first);if(!r)throw Stop{ConstraintSearchStatus::InvalidInput,"unknown resource"};auto c=b.AddCumulative(r->capacity);for(size_t i=0;i<entry.second.intervals.size();++i)c.AddDemand(entry.second.intervals[i],entry.second.demands[i]);}
 for(auto &entry:memories){auto c=b.AddCumulative(programMemory(d.target,entry.first)->capacityBytes);for(size_t i=0;i<entry.second.intervals.size();++i)c.AddDemand(entry.second.intervals[i],entry.second.demands[i]);}
 auto makespan=time();b.AddEquality(makespan,LinearExpr(end.back())+d.completeNs);b.AddHint(makespan,H);b.Minimize(makespan);checkpoint();auto proto=b.Build();t.variables=proto.variables_size();t.constraints=proto.constraints_size();t.modelBuildMs=ms();
 SatParameters parameters;parameters.set_max_time_in_seconds(std::max(0.000001,(o.wallSeconds-ms()/1000)*.9));parameters.set_num_search_workers(o.workers);parameters.set_random_seed(o.randomSeed);Model model;model.Add(NewSatParameters(parameters));auto solveStart=Clock::now();auto response=SolveCpModel(proto,&model);t.solveMs=std::chrono::duration<double,std::milli>(Clock::now()-solveStart).count();t.branches=response.num_branches();t.conflicts=response.num_conflicts();
 if(response.status()==CpSolverStatus::MODEL_INVALID)return fallback(ConstraintSearchStatus::InvalidInput,response.solution_info());
 if(response.status()==CpSolverStatus::INFEASIBLE)return fallback(ConstraintSearchStatus::InvalidInput,"encoding mismatch: finite model rejects its independently checked baseline");
 if(response.status()!=CpSolverStatus::FEASIBLE&&response.status()!=CpSolverStatus::OPTIMAL){t.budgetExhausted=true;return fallback(ConstraintSearchStatus::Unknown,"no joint incumbent within budget; checked baseline retained");}
 ProgramJointSolution s;s.hasPlan=true;s.selection="joint";s.status=response.status()==CpSolverStatus::OPTIMAL?ConstraintSearchStatus::Optimal:ConstraintSearchStatus::Feasible;s.message="source-bound joint grouping, placement, copies, routes and configuration schedule";s.makespanNs=SolutionIntegerValue(response,makespan);
 for(size_t v=0;v<V;++v)if(SolutionBooleanValue(response,variant[v]))s.variant=v;
 for(size_t k=0;k<K;++k)if(SolutionBooleanValue(response,used[k])){s.configurationStartNs.push_back(SolutionIntegerValue(response,begin[k]));s.configurationEndNs.push_back(SolutionIntegerValue(response,end[k]));}
 for(auto &a:assignments)if(SolutionBooleanValue(response,a.chosen)){ProgramJointSelection x;x.group=a.group;x.place=a.place;x.configuration=a.configuration;for(auto cb:a.compute)x.computeStartNs.push_back(SolutionIntegerValue(response,cb));for(bool input:{true,false})for(auto &r:(input?a.inputs:a.outputs))if(SolutionBooleanValue(response,r.chosen))(input?x.inputs:x.outputs).push_back({r.version,r.link,SolutionIntegerValue(response,r.begin),SolutionIntegerValue(response,r.end)});s.groups.push_back(x);}
 t.hasObjective=true;t.objectiveNs=s.makespanNs;t.bestBoundNs=response.best_objective_bound();t.relativeGap=std::max(0.0,double(t.objectiveNs)-t.bestBoundNs)/std::max(1.0,std::abs(double(t.objectiveNs)));t.budgetExhausted=s.status==ConstraintSearchStatus::Feasible;
 ProgramGraph graph;std::string error;if(!materializeProgramJointSolution(d,s,graph,error)||!checkProgramJointSolution(d,s,graph,error))return fallback(ConstraintSearchStatus::InvalidInput,"joint incumbent rejected: "+error);
 return finish(s);
 }catch(const Stop &stop){t.modelBuildMs=ms();return fallback(stop.status,stop.error);}
#endif
}

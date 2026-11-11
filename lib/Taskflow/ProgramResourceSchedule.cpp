#include "Taskflow/ProgramMapping.h"
#include "Taskflow/ProgramTargetBinding.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

using namespace orbit::taskflow;
namespace {
bool fail(std::string &e, const std::string &s) {
  e = "public resource schedule: " + s;
  return false;
}
bool validTime(double x) { return std::isfinite(x) && x >= 0; }
bool near(double x, double y) { return std::abs(x-y) <= 1e-9; }
}

bool orbit::taskflow::checkProgramResourceSchedule(const ProgramGraph &g,
                                                  std::string &error) {
  const auto &s = g.schedule;
  if (s.policy != "constraint-resource-schedule" || s.modelIdentity.empty() ||
      s.timingBoundary.empty() || s.events.size() != g.events.size() ||
      s.eventDurationsMs.size() != g.events.size() || !s.estimatedMakespanMs ||
      !validTime(*s.estimatedMakespanMs) || !s.uncalibrated.empty() ||
      s.configurations.empty())
    return fail(error,"missing explicit event costs, model, configurations or times");
  StaticConfig target;
  target.domains=g.domains; target.resources=g.resources;
  target.memories=g.memories; target.links=g.links;
  std::vector<size_t> eventGroups(g.events.size(),SIZE_MAX);
  std::vector<size_t> groupConfigurations(g.executionGroups.size(),SIZE_MAX);
  double last=0,maximum=0;
  for(size_t k=0;k<s.configurations.size();++k) {
    const auto &c=s.configurations[k];
    if(c.groups.empty() || !validTime(c.startMs) || !validTime(c.endMs) ||
       !validTime(c.setupMs) || c.endMs+1e-9<c.startMs+c.setupMs ||
       c.startMs+1e-9<last || !std::is_sorted(c.groups.begin(),c.groups.end()))
      return fail(error,"invalid configuration barrier or lifecycle");
    std::set<std::string> cores;
    double end=c.startMs+c.setupMs;
    for(auto gi:c.groups) {
      if(gi>=g.executionGroups.size() || groupConfigurations[gi]!=SIZE_MAX)
        return fail(error,"configuration group coverage is not unique");
      groupConfigurations[gi]=k;
      const auto &group=g.executionGroups[gi];
      if(!cores.insert(group.resource).second)
        return fail(error,"configuration assigns two static groups to one core");
      for(auto event:group.memberEvents) {
        if(event>=s.events.size() || eventGroups[event]!=SIZE_MAX)
          return fail(error,"configuration event ownership overlaps");
        eventGroups[event]=gi;
        const auto &t=s.events[event];
        if(!t.startMs || !t.endMs || *t.startMs+1e-9<c.startMs+c.setupMs ||
           *t.endMs>c.endMs+1e-9)
          return fail(error,"event escapes its configuration");
        end=std::max(end,*t.endMs);
      }
    }
    if(!near(c.endMs,end))return fail(error,"configuration completion is not its last activity");
    last=c.endMs; maximum=std::max(maximum,end);
  }
  if(std::find(eventGroups.begin(),eventGroups.end(),SIZE_MAX)!=eventGroups.end() ||
     std::find(groupConfigurations.begin(),groupConfigurations.end(),SIZE_MAX)!=groupConfigurations.end() ||
     !near(maximum,*s.estimatedMakespanMs))
    return fail(error,"schedule/configuration completion or coverage mismatch");
  std::map<std::string,std::map<double,int64_t>> calendars;
  for(size_t i=0;i<g.events.size();++i) {
    const auto &event=g.events[i]; const auto &time=s.events[i];
    if(time.event!=i || !time.startMs || !time.endMs || !validTime(*time.startMs) ||
       !validTime(*time.endMs) || !validTime(s.eventDurationsMs[i]) ||
       !near(*time.endMs-*time.startMs,s.eventDurationsMs[i]) ||
       time.predecessors!=event.dependencies)
      return fail(error,"event times do not match their explicit cost/dependencies");
    for(auto parent:event.dependencies) {
      if(parent>=i || !s.events[parent].endMs ||
         *s.events[parent].endMs>*time.startMs+1e-9)
        return fail(error,"scheduled consumer precedes producer completion");
      if(eventGroups[parent]!=eventGroups[i] &&
         groupConfigurations[eventGroups[parent]]>=groupConfigurations[eventGroups[i]])
        return fail(error,"inter-group communication requires an earlier completed configuration");
    }
    int64_t demand=1;
    if(event.kind!="transfer") {
      auto instance=std::find_if(g.instances.begin(),g.instances.end(),
          [&](const auto &v){return v.event==i;});
      if(instance==g.instances.end())return fail(error,"compute has no source instance");
      demand=instance->resourceCount;
    }
    std::map<std::string,int64_t> expanded;
    if(!programResourceDemands(target,{{event.resource,demand}},expanded,error))return false;
    if(*time.endMs>*time.startMs)for(const auto &d:expanded) {
      calendars[d.first][*time.startMs]+=d.second;
      calendars[d.first][*time.endMs]-=d.second;
    }
  }
  for(const auto &r:g.resources) {
    int64_t live=0;
    for(auto change:calendars[r.id]) {
      if(change.second>0 && live>INT64_MAX-change.second)
        return fail(error,"resource reservation overflow");
      live+=change.second;
      if(live<0 || live>r.capacity)return fail(error,"overlapping resource/shared-resource reservations");
    }
    if(live)return fail(error,"unclosed resource reservation");
  }
  std::map<std::string,std::map<double,int64_t>> memory;
  for(size_t ai=0;ai<g.allocations.size();++ai) {
    const auto &a=g.allocations[ai];
    double begin=INFINITY,end=0;
    std::set<size_t> owners;
    for(size_t gi=0;gi<g.executionGroups.size();++gi) {
      const auto &group=g.executionGroups[gi];
      if(group.memory!=a.memory)continue;
      bool used=false;
      for(auto event:group.memberEvents)
        for(const auto *values:{&g.events[event].reads,&g.events[event].writes})
          for(auto copy:*values)if(copy<g.copies.size() && g.copies[copy].allocation==ai)used=true;
      if(used)owners.insert(groupConfigurations[gi]);
    }
    if(!owners.empty())for(auto k:owners) {
      begin=std::min(begin,s.configurations[k].startMs);
      end=std::max(end,s.configurations[k].endMs);
    } else {
      // Caller-visible BOs are allocated for the complete invocation.
      begin=s.configurations.front().startMs;end=s.configurations.back().endMs;
    }
    if(!validTime(begin) || !validTime(end) || end<begin || a.bytes<=0)
      return fail(error,"invalid static allocation lifetime");
    if(end>begin){memory[a.memory][begin]+=a.bytes;memory[a.memory][end]-=a.bytes;}
  }
  for(const auto &m:g.memories) {
    int64_t live=0;
    for(auto change:memory[m.id]) {
      if(change.second>0 && live>INT64_MAX-change.second)
        return fail(error,"memory occupancy overflow");
      live+=change.second;
      if(live<0 || live>m.capacityBytes)return fail(error,"parallel static allocation occupancy exceeds capacity");
    }
  }
  return true;
}

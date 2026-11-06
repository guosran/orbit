#include "Taskflow/ProgramTargetBinding.h"
#include <set>
using namespace orbit::taskflow;
namespace {
bool fail(std::string &error,const std::string &message) {
  error="program target binding: "+message;return false;
}
}
const StaticResourceDomain *orbit::taskflow::programDomain(const StaticConfig &c,llvm::StringRef id) {
  for(const auto &d:c.domains)if(d.id==id)return &d;return nullptr;
}
const StaticResource *orbit::taskflow::programResource(const StaticConfig &c,llvm::StringRef id) {
  for(const auto &r:c.resources)if(r.id==id)return &r;return nullptr;
}
const StaticMemory *orbit::taskflow::programMemory(const StaticConfig &c,llvm::StringRef id) {
  for(const auto &m:c.memories)if(m.id==id)return &m;return nullptr;
}
const StaticTransfer *orbit::taskflow::programLink(const StaticConfig &c,llvm::StringRef id) {
  for(const auto &l:c.links)if(l.id==id)return &l;return nullptr;
}
bool orbit::taskflow::programOwnerExists(const StaticConfig &c,llvm::StringRef id) {
  return !id.empty() && programDomain(c,id);
}
bool orbit::taskflow::programDomainContains(const StaticConfig &c,llvm::StringRef ancestor,llvm::StringRef descendant) {
  if(!programOwnerExists(c,ancestor) || !programOwnerExists(c,descendant))return false;
  std::set<std::string> seen;auto current=descendant.str();
  while(!current.empty()) {
    if(current==ancestor)return true;
    if(!seen.insert(current).second)return false;
    auto d=programDomain(c,current);if(!d)return false;current=d->parent;
  }
  return false;
}
bool orbit::taskflow::programResourceOwner(const StaticConfig &c,llvm::StringRef id,std::string &owner,std::string &error) {
  owner.clear();auto r=programResource(c,id);
  if(!r || !programOwnerExists(c,r->domain))return fail(error,"resource has no declared owner instance: "+id.str());
  owner=r->domain;return true;
}
bool orbit::taskflow::programMemoryOwner(const StaticConfig &c,llvm::StringRef id,std::string &owner,std::string &error) {
  owner.clear();auto m=programMemory(c,id);
  if(!m || !programOwnerExists(c,m->domain))return fail(error,"memory has no declared owner instance: "+id.str());
  owner=m->domain;return true;
}
bool orbit::taskflow::checkProgramTargetBinding(const StaticConfig &c,ProgramTargetBindingView b,int64_t capacity,std::string &error) {
  if(!validateMemoryArchitecture(c,error))return false;
  auto host=programMemory(c,b.hostMemory),local=programMemory(c,b.localMemory);
  auto compute=programResource(c,b.computeResource),dma=programResource(c,b.dmaResource),submit=programResource(c,b.submissionResource);
  if(!host || !local || host==local || !compute || compute->kind!="compute" ||
     !dma || (dma->kind!="transfer"&&dma->kind!="communication") || !submit || submit->kind!="submission")
    return fail(error,"missing or incompatible memory/resource references");
  std::string owner;
  if(!programResourceOwner(c,b.computeResource,owner,error) || !programMemoryOwner(c,b.localMemory,owner,error) ||
     !programMemoryOwner(c,b.hostMemory,owner,error) || !programResourceOwner(c,b.dmaResource,owner,error) ||
     !programResourceOwner(c,b.submissionResource,owner,error))return false;
  if(capacity<=0 || (local->capacityKnown && capacity>local->capacityBytes))return fail(error,"payload budget exceeds target execution memory capacity");
  if(!memoryAccessible(c,b.localMemory,b.computeResource))return fail(error,"selected compute cannot access execution memory");
  for(bool input:{true,false}) {
    auto link=programLink(c,input?b.inputLink:b.outputLink);
    if(!link || link->resource!=b.dmaResource || link->source!=(input?b.hostMemory:b.localMemory) ||
       link->target!=(input?b.localMemory:b.hostMemory))return fail(error,"transfer binding does not reference the selected directed path/resource");
  }
  return true;
}

bool orbit::taskflow::programResourceDemands(const StaticConfig &c,
    const std::map<std::string,int64_t> &base,
    std::map<std::string,int64_t> &expanded,std::string &error) {
  expanded=base;
  if(!validateMemoryArchitecture(c,error))return false;
  // Architecture validation rejects cycles, so at most |resources| expansion
  // rounds propagate every implication, including transitive shared parents.
  for(size_t round=0;round<=c.resources.size();++round) {
    bool changed=false;auto current=expanded;
    for(const auto &entry:current) {
      auto r=programResource(c,entry.first);
      if(!r || entry.second<=0 || entry.second>r->capacity)return fail(error,"invalid or over-capacity reservation: "+entry.first);
      for(const auto &share:r->sharedResources) {
        if(entry.second>INT64_MAX/share.second)return fail(error,"shared reservation overflow");
        int64_t need=entry.second*share.second;
        if(expanded[share.first]<need){expanded[share.first]=need;changed=true;}
      }
    }
    if(!changed)return true;
  }
  return fail(error,"resource implication closure did not converge");
}

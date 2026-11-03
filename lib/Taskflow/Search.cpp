#include "Taskflow/Search.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <map>
using namespace orbit::taskflow;
namespace j=llvm::json;
namespace {
std::string render(const j::Value &v) { std::string s; llvm::raw_string_ostream os(s); os<<v; return s; }
bool write(const std::string &path, llvm::StringRef s, std::string &e) {
  std::error_code ec; llvm::raw_fd_ostream os(path+".tmp",ec); if(ec){e=ec.message();return false;} os<<s; os.close();
  ec=llvm::sys::fs::rename(path+".tmp",path); if(ec){e=ec.message();return false;} return true;
}
bool read(const std::string &path, j::Value &v, std::string &e) {
  auto b=llvm::MemoryBuffer::getFile(path); if(!b){e=b.getError().message();return false;}
  auto p=j::parse((*b)->getBuffer()); if(!p){e=llvm::toString(p.takeError());return false;} v=std::move(*p);return true;
}
std::string hash(llvm::StringRef s) {
  auto h=llvm::SHA256::hash(llvm::ArrayRef<uint8_t>((const uint8_t*)s.data(),s.size()));
  std::string r; for(auto c:h){r+="0123456789abcdef"[c>>4];r+="0123456789abcdef"[c&15];}return r;
}
std::string fileDigest(const std::string &path) {
  auto bytes=llvm::MemoryBuffer::getFile(path);return bytes?hash((*bytes)->getBuffer()):"missing";
}
void bindFiles(const j::Value &v,std::string &out) {
  if(auto s=v.getAsString()){if(llvm::sys::fs::is_regular_file(*s))out+=s->str()+fileDigest(s->str());}
  else if(auto a=v.getAsArray())for(auto &x:*a)bindFiles(x,out);
  else if(auto ob=v.getAsObject()) {
    // Object iteration is not a canonical serialization. Sort keys so file
    // dependency bindings survive a fresh process and allocation layout.
    std::vector<std::string> keys;for(auto &x:*ob)keys.push_back(x.first.str());
    std::sort(keys.begin(),keys.end());
    for(auto &key:keys)bindFiles(*ob->get(key),out);
  }
}
struct State { Program p; std::string key, parent, action, status; double score=0; int64_t round=0; j::Value response=nullptr; };
bool scored(const State &s){return s.status=="passed";}
j::Value row(const State &s) {return j::Object{{"search_key",s.key},{"candidate_id",s.p.candidateId},{"parent",s.parent},{"action",s.action},{"round",s.round},{"status",s.status},{"score",scored(s)?j::Value(s.score):j::Value(nullptr)},{"program",printPlan(s.p)},{"evaluation",j::Value(s.response)}};}
}
std::string orbit::taskflow::canonicalSearchIdentity(const Program &p) {
  Program q=p; q.candidateId.clear(); q.transforms.clear();
  std::map<std::string,std::string> names;
  for(size_t i=0;i<q.tasks.size();++i)names[q.tasks[i].id]="t"+std::to_string(i);
  for(auto &t:q.tasks){t.id=names[t.id];for(auto &d:t.dependsOn)d=names[d];std::sort(t.dependsOn.begin(),t.dependsOn.end());}
  return fingerprint(q);
}
std::vector<SearchAction> orbit::taskflow::generateSearchActions(const Program &p, const SearchPreferences &preferences) {
  int64_t m=0,n=0,k=0;
  for(auto &t:p.tasks)for(auto &c:t.operations)if(c.kind=="matmul") {
    m=std::max(m,c.m.hi-c.m.lo);n=std::max(n,c.n.hi-c.n.lo);k=std::max(k,c.k.hi-c.k.lo);
  }
  auto half=[](int64_t v,int64_t alignment){return v>alignment*2?((v/2+alignment-1)/alignment)*alignment:(v+1)/2;};
  std::vector<SearchAction> a;
  if(n>1)a.push_back({"tile_n_"+std::to_string(half(n,preferences.n)),{0,half(n,preferences.n),0,false,false}});
  if(k>1)a.push_back({"tile_k_"+std::to_string(half(k,preferences.k)),{0,0,half(k,preferences.k),false,false}});
  if(m>1)a.push_back({"tile_m_"+std::to_string(half(m,preferences.m)),{half(m,preferences.m),0,0,false,false}});
  a.push_back({"fuse_matmul_relu",{0,0,0,true,false}});
  a.push_back({"fission_matmul_relu",{0,0,0,false,true}});
  return a;
}
bool orbit::taskflow::runStructuredSearch(const Program &source,const SearchOptions &o,std::string &e) {
  if((o.backend!="amoeba"&&o.backend!="npu"&&o.backend!="static")||o.maxRounds<1||o.maxCandidates<1||o.beamWidth<1||o.outputDir.empty()||o.evaluator.empty()) {e="invalid structured search options";return false;}
  if(!validate(source,e))return false;
  if(auto ec=llvm::sys::fs::create_directories(o.outputDir)){e=ec.message();return false;}
  j::Value config=j::Object{};if(!o.targetConfigFile.empty()&&!read(o.targetConfigFile,config,e))return false;
  if(!config.getAsObject()){e="target config must be a JSON object";return false;}
  SearchPreferences preferences;
  if(auto v=config.getAsObject()->get("search_preferences")) {
    auto object=v->getAsObject();
    if(!object){e="search_preferences must be an object";return false;}
    for(auto &entry:*object) {
      auto value=entry.second.getAsInteger();
      if(!value||*value<1||*value>1000000000){e="search preference alignment must be an integer in [1, 1000000000]";return false;}
      if(entry.first=="m_alignment")preferences.m=*value;
      else if(entry.first=="n_alignment")preferences.n=*value;
      else if(entry.first=="k_alignment")preferences.k=*value;
      else {e="unknown search preference";return false;}
    }
  }
  auto exe=llvm::sys::findProgramByName(o.evaluator);if(!exe){e="evaluator executable not found";return false;}
  auto exeBytes=llvm::MemoryBuffer::getFile(*exe);if(!exeBytes){e="cannot hash evaluator";return false;}
  std::string dependencyBinding;bindFiles(config,dependencyBinding);
  dependencyBinding+=fileDigest(llvm::sys::fs::getMainExecutable(nullptr,(void*)&runStructuredSearch));
  std::string binding=hash(dependencyBinding+canonicalSearchIdentity(source)+o.backend+render(config)+(*exeBytes)->getBuffer().str()+std::to_string(o.maxCandidates)+":"+std::to_string(o.maxRounds)+":"+std::to_string(o.beamWidth));
  if(!o.resume && !write(o.outputDir+"/binding-components.json",render(j::Object{
      {"dependencies",hash(dependencyBinding)},{"source",canonicalSearchIdentity(source)},
      {"backend",o.backend},{"config",hash(render(config))},{"evaluator",hash((*exeBytes)->getBuffer())},
      {"budget",std::to_string(o.maxCandidates)+":"+std::to_string(o.maxRounds)+":"+std::to_string(o.beamWidth)}}),e))return false;
  std::string cp=o.checkpoint.empty()?o.outputDir+"/checkpoint.json":o.checkpoint;
  std::vector<State> archive,beam;std::set<std::string> seen; int64_t nextRound=1, evaluations=0;
  j::Array selections,attempts;
  auto save=[&](){j::Array rows,bs;for(auto &s:archive)rows.push_back(row(s));for(auto &s:beam)bs.push_back(s.key);
    return write(cp,render(j::Object{{"schema","orbit.structured.search.v1"},{"binding",binding},{"next_round",nextRound},{"evaluations",evaluations},{"archive",std::move(rows)},{"beam",std::move(bs)},{"selections",j::Array(selections)},{"attempts",j::Array(attempts)}}),e);};
  if(o.resume){j::Value saved=nullptr;if(!read(cp,saved,e))return false;auto root=saved.getAsObject();
    if(!root||root->getString("schema")!="orbit.structured.search.v1"||root->getString("binding")!=binding){e="checkpoint binding differs (source/config/evaluator/budget)";return false;}
    auto rows=root->getArray("archive"),bs=root->getArray("beam");
    if(!rows||!bs||!root->getInteger("next_round")||!root->getInteger("evaluations")){e="invalid checkpoint";return false;}
    nextRound=*root->getInteger("next_round");evaluations=*root->getInteger("evaluations");
    if(auto a=root->getArray("selections"))selections=*a;if(auto a=root->getArray("attempts"))attempts=*a;
    for(auto &r:*rows){auto x=r.getAsObject();State s;if(!x||!x->get("program")||!parsePlan(*x->get("program"),s.p,e))return false;
      s.key=x->getString("search_key").value_or("").str();s.parent=x->getString("parent").value_or("").str();s.action=x->getString("action").value_or("").str();s.status=x->getString("status").value_or("").str();s.round=x->getInteger("round").value_or(0);s.score=x->getNumber("score").value_or(0);if(auto v=x->get("evaluation"))s.response=*v;
      if(s.key!=canonicalSearchIdentity(s.p)|| (scored(s)&&(!std::isfinite(s.score)||s.score<=0))){e="checkpoint identity/score authentication failed";return false;}
      if(scored(s)) {
        auto r=s.response.getAsObject();auto score=r?r->getObject("score"):nullptr;
        if(!r||(o.backend=="static" && r->getString("evaluation_kind")!="static_estimate")||r->getString("status")!="passed"||r->getString("candidate_id")!=s.p.candidateId||r->getString("source_id")!=s.p.sourceId||!score||score->getString("unit")!=(o.backend=="amoeba"?"cycles":"ms")||score->getNumber("value")!=s.score){e="checkpoint evaluation authentication failed";return false;}
      }
      seen.insert(s.key);archive.push_back(std::move(s));}
    for(auto &v:*bs){bool found=false;for(auto &s:archive)if(v.getAsString()==s.key&&scored(s)){beam.push_back(s);found=true;break;}if(!found){e="invalid checkpoint beam";return false;}}
  }
  auto evaluate=[&](State &s)->bool {
    std::string dir=o.outputDir+"/candidates/"+s.key;if(auto ec=llvm::sys::fs::create_directories(dir)){e=ec.message();return false;}
    std::string plan=render(printPlan(s.p)), mlir=emitTaskflow(s.p), escaped;
    llvm::raw_string_ostream es(escaped);es<<'"';llvm::printEscapedString(plan,es);es<<'"';
    mlir.replace(0,8,"module attributes {orbit.plan = "+escaped+"} {");
    if(!write(dir+"/plan.json",plan,e)||!write(dir+"/taskflow.mlir",mlir,e))return false;
    if(evaluations>=o.maxCandidates){s.status="budget_unmeasured";return true;}
    if(s.p.tasks.size()>4096){s.status="unsupported";s.response=j::Object{{"reason","target task-count guard: >4096"}};return true;}
    j::Value req=j::Object{{"schema","orbit.search.evaluation.v1"},{"backend",o.backend},{"candidate_id",s.p.candidateId},{"search_key",s.key},{"source_id",s.p.sourceId},{"plan_path",dir+"/plan.json"},{"mlir_path",dir+"/taskflow.mlir"},{"output_dir",dir+"/evaluation"},{"target_config",j::Value(config)}};
    std::string request=dir+"/request.json",response=dir+"/response.json";
    if(!write(request,render(req),e))return false;
    ++evaluations;llvm::sys::fs::remove(response);
    std::string executionError;int rc=llvm::sys::ExecuteAndWait(*exe,{*exe,request,response},std::nullopt,{},0,0,&executionError);
    std::string readError; s.status="runtime_failed";
    if(rc==0&&read(response,s.response,readError)){auto r=s.response.getAsObject();
      if(r){auto status=r->getString("status");if(status&&(*status=="unsupported"||*status=="compile_failed"||*status=="runtime_failed"||*status=="numerical_failed"))s.status=status->str();
        if(status&&*status=="passed" && (o.backend!="static" || r->getString("evaluation_kind")=="static_estimate") && r->getString("candidate_id")==s.p.candidateId && r->getString("source_id")==s.p.sourceId){auto score=r->getObject("score");if(score){auto v=score->getNumber("value");auto unit=score->getString("unit");if(v&&std::isfinite(*v)&&*v>0&&unit&&*unit==(o.backend=="amoeba"?"cycles":"ms")){s.status="passed";s.score=*v;}}}
      }
    }else s.response=j::Object{{"reason",executionError+readError},{"exit_code",rc}};
    return true;
  };
  if(archive.empty()){State base; base.p=source;if(!transform(base.p,{},e))return false;base.key=canonicalSearchIdentity(base.p);base.action="identity";if(!evaluate(base))return false;archive.push_back(base);seen.insert(base.key);if(scored(base))beam.push_back(base);if(!save())return false;}
  for(;nextRound<=o.maxRounds && !beam.empty();){
    std::vector<State> children;
    for(auto &parent:beam)for(auto &action:generateSearchActions(parent.p,preferences)){
      State s;s.p=parent.p;s.parent=parent.key;s.action=action.name;s.round=nextRound;
      std::string reason;
      if(!transform(s.p,action.options,reason)||!validate(s.p,reason)){attempts.push_back(j::Object{{"round",nextRound},{"parent",parent.key},{"action",action.name},{"status","semantic_invalid"},{"reason",reason}});continue;}
      s.key=canonicalSearchIdentity(s.p);
      if(!seen.insert(s.key).second){attempts.push_back(j::Object{{"round",nextRound},{"parent",parent.key},{"action",action.name},{"search_key",s.key},{"candidate_id",s.p.candidateId},{"status","duplicate"}});
        for(auto &old:archive)if(old.key==s.key&&old.round==nextRound&&scored(old))children.push_back(old);continue;}
      if(!evaluate(s))return false;archive.push_back(s);if(scored(s))children.push_back(s);if(!save())return false;
    }
    beam=selectSearchBeam(std::move(children),o.beamWidth,0,[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,[](auto &s){return s.key;},[](auto &s){return s.action;});
    j::Array selected;for(auto &s:beam)selected.push_back(j::Object{{"search_key",s.key},{"candidate_id",s.p.candidateId},{"score",s.score},{"reason","lowest evaluated score among unique legal children; archive retains global best"}});
    selections.push_back(j::Object{{"round",nextRound},{"selected",std::move(selected)}});++nextRound;if(!save())return false;
  }
  auto ranked=selectSearchBeam(archive,archive.size(),0,[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,[](auto &s){return s.key;},[](auto &s){return s.action;});
  j::Array rows;for(auto &s:archive)rows.push_back(row(s));j::Array ranking;for(auto &s:ranked)ranking.push_back(s.key);
  return write(o.outputDir+"/search.json",render(j::Object{{"schema","orbit.structured.search.v1"},{"backend",o.backend},{"score_unit",o.backend=="amoeba"?"cycles":"ms"},{"binding",binding},{"evaluations",evaluations},{"archive",std::move(rows)},{"ranking",std::move(ranking)},{"selections",std::move(selections)},{"attempts",std::move(attempts)}}),e);
}

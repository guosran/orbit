#include "Taskflow/Search.h"
#include "Taskflow/StaticEvaluation.h"
#include "Taskflow/GenericMapping.h"
#include "Taskflow/NativeTransforms.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
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
// Evaluation identity is semantic JSON, independent of object insertion order.
std::string canonicalJSON(const j::Value &v) {
  if (auto object=v.getAsObject()) {
    std::vector<std::string> keys;for(auto &entry:*object)keys.push_back(entry.first.str());
    std::sort(keys.begin(),keys.end());std::string result="{";
    for(auto &key:keys) {if(result.size()>1)result+=",";result+=render(key)+":"+canonicalJSON(*object->get(key));}
    return result+"}";
  }
  if(auto array=v.getAsArray()) {
    std::string result="[";for(auto &item:*array){if(result.size()>1)result+=",";result+=canonicalJSON(item);}return result+"]";
  }
  return render(v);
}
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
struct State { Program p; std::string nativeMLIR; std::string key, parent, action, status, evaluationIdentity; std::vector<std::string> actionChain; double score=0; int64_t round=0; j::Value response=nullptr; };
bool scored(const State &s){return s.status=="passed";}
std::string materializedMLIR(const Program &p) {
  std::string mlir=emitTaskflow(p), escaped;
  llvm::raw_string_ostream es(escaped);
  es << '"'; llvm::printEscapedString(render(printPlan(p)),es); es << '"';
  mlir.replace(0,8,"module attributes {orbit.plan = "+escaped+"} {");
  return mlir;
}
j::Value row(const State &s) {
  j::Array chain;for(auto &action:s.actionChain)chain.push_back(action);
  return j::Object{{"search_key",s.key},{"candidate_id",s.nativeMLIR.empty()?s.p.candidateId:s.key},{"evaluation_identity",s.evaluationIdentity},{"parent",s.parent},{"action",s.action},{"action_chain",std::move(chain)},{"round",s.round},{"status",s.status},{"score",scored(s)?j::Value(s.score):j::Value(nullptr)},{"program",s.nativeMLIR.empty()?printPlan(s.p):j::Value(nullptr)},{"native_mlir",s.nativeMLIR.empty()?j::Value(nullptr):j::Value(s.nativeMLIR)},{"evaluation",j::Value(s.response)}};
}
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
  const bool nativeAir = o.backend == "air" && (o.evaluator.empty() || o.evaluator == "native");
  const bool nativeStatic = (o.backend == "static" || nativeAir) && (o.evaluator.empty() || o.evaluator == "native");
  if((o.backend!="amoeba"&&o.backend!="npu"&&o.backend!="static"&&o.backend!="air")||o.maxRounds<1||o.maxCandidates<1||o.beamWidth<1||o.outputDir.empty()||(!nativeStatic && o.evaluator.empty())) {e="invalid structured search options";return false;}
  const bool nativeBody = !o.nativeSourceMLIR.empty();
  if(nativeBody && !nativeAir) {e="native-body search currently requires backend=air native evaluator";return false;}
  mlir::DialectRegistry nativeRegistry;
  nativeRegistry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,mlir::arith::ArithDialect,mlir::func::FuncDialect,mlir::math::MathDialect,mlir::memref::MemRefDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext nativeContext(nativeRegistry);
  auto parseNative=[&](const std::string &text){return mlir::parseSourceString<mlir::ModuleOp>(text,&nativeContext);};
  auto nativeTransform=[&](std::string &text,bool split,std::string &reason,
                            const TransformOptions *tiles=nullptr,
                            const SearchAction *mappingAction=nullptr){
    auto module=parseNative(text);GenericAnalysis analysis;
    if(!module||!analyzeGenericProgram(*module,analysis,reason))return false;
    if(!normalizeNativeMappingChoice(*module,reason))return false;
    if(split&&!splitNativeOuterLoops(*module,reason))return false;
    if(tiles&&!setNativeAIRTileSizes(*module,tiles->tileM,tiles->tileN,tiles->tileK,reason))return false;
    if(mappingAction && mappingAction->mapping &&
       !setNativeMappingChoice(*module,*mappingAction->mapping,reason))return false;
    if(mappingAction && !mappingAction->implementation.empty() &&
       !setNativeImplementation(*module,mappingAction->implementation,
                                mappingAction->vectorLanes,reason))return false;
    if(mappingAction && mappingAction->partitionExtent>=0) {
      if(mappingAction->partitionExtent)
        module->getOperation()->setAttr("orbit.program_partition_extent",
            mlir::IntegerAttr::get(mlir::IntegerType::get(&nativeContext,64),
                                   mappingAction->partitionExtent));
      else module->getOperation()->removeAttr("orbit.program_partition_extent");
    }
    if(!analyzeGenericProgram(*module,analysis,reason))return false;
    text.clear();llvm::raw_string_ostream os(text);module->print(os);return true;
  };
  std::string sourceText=o.nativeSourceMLIR;
  if(nativeBody) {if(!nativeTransform(sourceText,false,e))return false;} else if(!validate(source,e))return false;
  const std::string sourceKey=nativeBody ? hash(sourceText) : canonicalSearchIdentity(source);
  auto stateCandidate=[&](const State &s){return nativeBody?s.key:s.p.candidateId;};
  auto stateSource=[&](const State &s){return nativeBody?sourceKey:s.p.sourceId;};
  auto stateIdentity=[&](const State &s){return nativeBody?hash(s.nativeMLIR):canonicalSearchIdentity(s.p);};
  auto stateMLIR=[&](const State &s){return nativeBody?s.nativeMLIR:(nativeAir?emitNativeTaskflow(s.p,e):materializedMLIR(s.p));};

  if(auto ec=llvm::sys::fs::create_directories(o.outputDir)){e=ec.message();return false;}
  j::Value config=j::Object{};if(!o.targetConfigFile.empty()&&!read(o.targetConfigFile,config,e))return false;
  if(!config.getAsObject()){e="target config must be a JSON object";return false;}
  int64_t diversitySlots=0;
  if(auto setting=config.getAsObject()->get("search_diversity_slots")) {
    auto count=setting->getAsInteger();
    if(!count || *count<0 || *count>o.beamWidth) {e="search_diversity_slots must be an integer in [0, beam width]";return false;}
    diversitySlots=*count;
  }
  auto structuralConfiguration=[&](const State &state) {
    if(!nativeBody)return state.action;
    auto module=parseNative(state.nativeMLIR);
    // Keep transformed bodies, tile/domain/resource/implementation choices;
    // vary data organization within each structural class without treating
    // every residency action name as a new structural configuration.
    module->getOperation()->removeAttr("orbit.mapping");
    std::string text;llvm::raw_string_ostream stream(text);module->print(stream);
    return hash(text);
  };
  std::vector<SearchAction> nativeActions{{"split_outer_loops",{}}};
  if (auto candidates=config.getAsObject()->getArray("native_air_tile_candidates")) {
    nativeActions.clear();
    for (auto &candidate:*candidates) {
      auto triple=candidate.getAsArray();
      if (!triple || triple->size()!=3) { e="native AIR tile candidate must have three sizes"; return false; }
      int64_t sizes[3];
      for (unsigned i=0;i<3;++i) {
        auto size=(*triple)[i].getAsInteger();
        if(!size || *size<=0) {e="native AIR tile sizes must be positive integers";return false;}
        sizes[i]=*size;
      }
      nativeActions.push_back({"air_tiles_"+std::to_string(sizes[0])+"_"+
          std::to_string(sizes[1])+"_"+std::to_string(sizes[2]),
          {sizes[0],sizes[1],sizes[2],false,false}});
    }
  }
  bool discoverMappings = false;
  if (auto setting=config.getAsObject()->get("native_mapping_candidates")) {
    if (setting->getAsString() != "discover") {
      e="native_mapping_candidates currently requires 'discover'";return false;
    }
    discoverMappings = true;
    // Mapping proposals supplement structural proposals in the same beam.
    SearchAction local{"mapping_phase_local", {}};
    local.mapping = ProgramMappingChoice{};
    nativeActions.push_back(std::move(local));
  }
  if (auto candidates=config.getAsObject()->getArray("native_implementation_candidates")) {
    for (auto &candidate:*candidates) {
      auto object=candidate.getAsObject();
      if (!object || !object->getString("kind") || !object->getInteger("lanes")) {
        e="implementation candidate requires kind and lanes"; return false;
      }
      SearchAction action{"implementation_"+object->getString("kind")->str()+
          "_"+std::to_string(*object->getInteger("lanes")),{}};
      action.implementation=object->getString("kind")->str();
      action.vectorLanes=*object->getInteger("lanes");
      nativeActions.push_back(std::move(action));
    }
  }
  if(auto candidates=config.getAsObject()->getArray("native_program_partition_candidates")) {
    for(const auto &candidate:*candidates) {
      auto extent=candidate.getAsInteger();
      if(!extent || *extent<0 || *extent>INT32_MAX) {
        e="program partition candidates must be integers in [0,INT32_MAX]";return false;
      }
      SearchAction action{"program_partition_"+std::to_string(*extent),{}};
      action.partitionExtent=*extent;nativeActions.push_back(std::move(action));
    }
  }
  auto nativeActionsFor=[&](const State &state) {
    auto actions=nativeActions;
    if (discoverMappings) {
      auto module=parseNative(state.nativeMLIR); GenericAnalysis analysis;
      std::string diagnostic;
      if (module && analyzeGenericProgram(*module,analysis,diagnostic))
        for (auto choice:discoverGenericMappingChoices(analysis)) {
          if (choice.residency=="phase-local") continue;
          SearchAction action{"mapping_"+choice.dataOrganization+"_storage_"+
              std::to_string(choice.storage)+"_reuse_"+
              std::to_string(choice.reuseLoop)+"_panel_"+
              std::to_string(choice.panelLoop),{}};
          action.mapping=std::move(choice); actions.push_back(std::move(action));
        }
    }
    return actions;
  };
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
  std::string executable, evaluatorIdentity = "native-cpp-program-data-organization-v8";
  if (!nativeStatic) {
    auto found = llvm::sys::findProgramByName(o.evaluator);
    if (!found) { e="evaluator executable not found"; return false; }
    executable=*found;
    auto bytes=llvm::MemoryBuffer::getFile(executable);
    if (!bytes) { e="cannot hash evaluator"; return false; }
    evaluatorIdentity=(*bytes)->getBuffer().str();
  }
  j::Value staticConfig=nullptr;
  if (nativeStatic) {
    if (auto path=config.getAsObject()->getString("mapping_config")) {
      if (!read(path->str(),staticConfig,e)) return false;
    } else staticConfig=config;
  }
  std::string dependencyBinding;bindFiles(config,dependencyBinding);
  dependencyBinding+=fileDigest(llvm::sys::fs::getMainExecutable(nullptr,(void*)&runStructuredSearch));
  const std::string evaluationBinding=hash(dependencyBinding+o.backend+canonicalJSON(config)+evaluatorIdentity);
  const auto evaluationIdentityFor=[&](const State &s){return hash(evaluationBinding+stateIdentity(s));};
  std::string binding=hash(evaluationBinding+sourceKey+std::to_string(o.maxCandidates)+":"+std::to_string(o.maxRounds)+":"+std::to_string(o.beamWidth));
  if(!o.resume && !write(o.outputDir+"/binding-components.json",render(j::Object{
      {"dependencies",hash(dependencyBinding)},{"source",sourceKey},
      {"backend",o.backend},{"config",hash(canonicalJSON(config))},{"evaluator",hash(evaluatorIdentity)},
      {"evaluation_binding",evaluationBinding},
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
    auto savedSelections=root->getArray("selections"),savedAttempts=root->getArray("attempts");
    if(nextRound<1 || nextRound-1>o.maxRounds || evaluations<0 || evaluations>o.maxCandidates ||
       !savedSelections || !savedAttempts || savedSelections->size()!=size_t(nextRound-1)) {
      e="checkpoint round/evaluation/history consistency failed";return false;
    }
    selections=*savedSelections;attempts=*savedAttempts;
    for(auto &r:*rows){auto x=r.getAsObject();State s;if(!x)return false;
      if(nativeBody){s.nativeMLIR=x->getString("native_mlir").value_or("").str();if(!nativeTransform(s.nativeMLIR,false,e))return false;}else if(!x->get("program")||!parsePlan(*x->get("program"),s.p,e))return false;
      s.key=x->getString("search_key").value_or("").str();s.parent=x->getString("parent").value_or("").str();s.action=x->getString("action").value_or("").str();s.status=x->getString("status").value_or("").str();s.round=x->getInteger("round").value_or(0);s.score=x->getNumber("score").value_or(0);if(auto v=x->get("evaluation"))s.response=*v;
      s.evaluationIdentity=x->getString("evaluation_identity").value_or("").str();
      if(auto chain=x->getArray("action_chain"))for(auto &action:*chain) {
        auto name=action.getAsString();if(!name){e="invalid action chain";return false;}
        s.actionChain.push_back(name->str());
      }
      if(s.key!=stateIdentity(s)|| (scored(s)&&(!std::isfinite(s.score)||s.score<=0))){e="checkpoint identity/score consistency failed";return false;}
      if(s.evaluationIdentity!=evaluationIdentityFor(s)){e="checkpoint target/model evaluation identity mismatch";return false;}
      if(scored(s)) {
        auto r=s.response.getAsObject();auto score=r?r->getObject("score"):nullptr;
        if(!r||(o.backend=="static" && r->getString("evaluation_kind")!="static_estimate")||r->getString("status")!="passed"||r->getString("candidate_id")!=stateCandidate(s)||r->getString("source_id")!=stateSource(s)||!score||score->getString("unit")!=(o.backend=="amoeba"?"cycles":"ms")||score->getNumber("value")!=s.score){e="checkpoint evaluation consistency failed";return false;}
        if (nativeStatic) {
          if(r->getString("evaluation_identity")!=s.evaluationIdentity){e="checkpoint feedback evaluation identity mismatch";return false;}
          std::string diagnostic;
          auto replanned=nativeAir ? evaluateGenericMLIR(stateMLIR(s),staticConfig,diagnostic) : evaluateStaticMLIR(materializedMLIR(s.p),staticConfig,diagnostic);
          auto result=replanned.getAsObject();
          auto best=result ? result->getObject("best_plan") : nullptr;
          if(!best || result->getString("status")=="unsupported" ||
              best->getNumber(nativeAir ? "estimated_ms" : "makespan_ms")!=s.score ||
              r->getString("plan_id")!=best->getString("plan_id") ||
              r->getString("evaluator_kind")!="native_cpp") {
            e="native checkpoint score failed independent re-evaluation";return false;
          }
        }
      }
      if(!seen.insert(s.key).second){e="duplicate checkpoint archive identity";return false;}
      archive.push_back(std::move(s));}
    size_t baselineCount=0;
    int64_t countedEvaluations=0;
    for (const auto &state:archive) {
      // A checkpoint may include part of nextRound: save() is called after
      // each child, before that round's selection. Older selections are final.
      if(state.round<0 || state.round>o.maxRounds || state.round>nextRound) {
        e="checkpoint candidate round out of range";return false;
      }
      if(state.status!="budget_unmeasured" && state.p.tasks.size()<=4096)
        ++countedEvaluations;
      if (state.parent.empty()) {
        ++baselineCount;
        if(state.key!=sourceKey || state.round!=0 || state.action!="identity" || !state.actionChain.empty()) {
          e="checkpoint baseline differs from canonical source";return false;
        }
      } else {
        auto parent=llvm::find_if(archive,[&](const auto &s){return s.key==state.parent;});
        if (parent==archive.end()) {e="action chain parent missing";return false;}
        if(!scored(*parent) || state.round!=parent->round+1) {
          e="checkpoint parent/child round consistency failed";return false;
        }
        bool selectedParent=state.round==1 ? parent->key==sourceKey : false;
        if(state.round>1 && size_t(state.round-2)<selections.size()) {
          auto selection=selections[size_t(state.round-2)].getAsObject();
          if(auto selected=selection ? selection->getArray("selected") : nullptr)
            for(auto &entry:*selected) if(auto row=entry.getAsObject())
              selectedParent |= row->getString("search_key")==parent->key;
        }
        if(!selectedParent) {e="checkpoint parent was not selected in preceding round";return false;}
        auto chain=parent->actionChain;chain.push_back(state.action);
        if (chain!=state.actionChain) {e="action chain disagrees with parent";return false;}
        auto actions=nativeBody ? nativeActionsFor(*parent) : generateSearchActions(parent->p,preferences);
        auto action=llvm::find_if(actions,[&](const auto &a){return a.name==state.action;});
        if(action==actions.end()) {e="checkpoint action is not a current source-derived proposal";return false;}
        State replay=*parent;
        std::string diagnostic;
        bool valid=nativeBody ? nativeTransform(replay.nativeMLIR,action->name=="split_outer_loops",diagnostic,
            action->options.tileM?&action->options:nullptr,&*action)
            : (transform(replay.p,action->options,diagnostic) && validate(replay.p,diagnostic));
        if(!valid || stateIdentity(replay)!=state.key) {
          e="checkpoint action replay differs from child identity";return false;
        }
      }
    }
    if(baselineCount!=1 || countedEvaluations!=evaluations) {
      e="checkpoint baseline/evaluation count consistency failed";return false;
    }
    std::vector<std::string> expectedBeam;
    auto baseline=llvm::find_if(archive,[](const auto &s){return s.parent.empty();});
    if(scored(*baseline)) expectedBeam.push_back(baseline->key);
    for(size_t i=0;i<selections.size();++i) {
      if(expectedBeam.empty()) {e="checkpoint completed a round without a scored parent beam";return false;}
      // A completed round must account for every current proposal, including
      // unmeasured budget rows. Otherwise an empty forged selection could skip
      // work while appearing internally self-consistent.
      for(const auto &parentKey:expectedBeam) {
        auto parent=llvm::find_if(archive,[&](const auto &s){return s.key==parentKey;});
        auto actions=nativeBody ? nativeActionsFor(*parent) : generateSearchActions(parent->p,preferences);
        for(const auto &action:actions) {
          State replay=*parent;std::string diagnostic;
          bool valid=nativeBody ? nativeTransform(replay.nativeMLIR,action.name=="split_outer_loops",diagnostic,
              action.options.tileM?&action.options:nullptr,&action)
              : (transform(replay.p,action.options,diagnostic) && validate(replay.p,diagnostic));
          if(!valid)continue;
          auto key=stateIdentity(replay);
          if(llvm::none_of(archive,[&](const auto &s){return s.key==key && s.round<=int64_t(i+1);})) {
            e="checkpoint completed round omits a current proposal";return false;
          }
        }
      }
      auto selection=selections[i].getAsObject();
      auto selected=selection ? selection->getArray("selected") : nullptr;
      if(!selection || selection->getInteger("round")!=int64_t(i+1) || !selected || selected->size()>size_t(o.beamWidth)) {
        e="checkpoint selection history consistency failed";return false;
      }
      std::vector<State> candidates;
      for(const auto &state:archive) if(state.round==int64_t(i+1)) candidates.push_back(state);
      auto recomputed=selectSearchBeam(std::move(candidates),o.beamWidth,diversitySlots,
          [](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,
          [](auto &s){return s.key;},structuralConfiguration);
      if(recomputed.size()!=selected->size()) {e="checkpoint selected beam differs from scored round";return false;}
      expectedBeam.clear();
      for(size_t j=0;j<selected->size();++j) {
        auto row=(*selected)[j].getAsObject();
        if(!row || row->getString("search_key")!=recomputed[j].key || row->getNumber("score")!=recomputed[j].score) {
          e="checkpoint selected beam differs from cost/diversity selection";return false;
        }
        expectedBeam.push_back(recomputed[j].key);
      }
    }
    if(bs->size()>size_t(o.beamWidth) || bs->size()!=expectedBeam.size()) {
      e="checkpoint beam width/history consistency failed";return false;
    }
    std::set<std::string> beamKeys;
    for(size_t i=0;i<bs->size();++i) {
      auto key=(*bs)[i].getAsString();
      if(!key || !beamKeys.insert(key->str()).second || *key!=expectedBeam[i]) {
        e="checkpoint beam identity/order consistency failed";return false;
      }
      auto state=llvm::find_if(archive,[&](const auto &s){return s.key==*key;});
      if(state==archive.end() || !scored(*state)) {e="invalid checkpoint beam";return false;}
      beam.push_back(*state);
    }
  }
  auto evaluate=[&](State &s)->bool {
    s.evaluationIdentity=evaluationIdentityFor(s);
    std::string dir=o.outputDir+"/candidates/"+s.key;if(auto ec=llvm::sys::fs::create_directories(dir)){e=ec.message();return false;}
    std::string plan=render(nativeBody ? j::Value(j::Object{{"kind","native_mlir_interface_only"},{"body_sha256",s.key}}) : printPlan(s.p)), mlir=stateMLIR(s);
    if(mlir.empty())return false;
    if(!write(dir+"/plan.json",plan,e)||!write(dir+"/taskflow.mlir",mlir,e))return false;
    if(evaluations>=o.maxCandidates){s.status="budget_unmeasured";return true;}
    if(s.p.tasks.size()>4096){s.status="unsupported";s.response=j::Object{{"reason","target task-count guard: >4096"}};return true;}
    j::Value req=j::Object{{"schema","orbit.search.evaluation.v1"},{"backend",o.backend},{"candidate_id",nativeBody?s.key:s.p.candidateId},{"search_key",s.key},{"evaluation_identity",s.evaluationIdentity},{"source_id",stateSource(s)},{"plan_path",dir+"/plan.json"},{"mlir_path",dir+"/taskflow.mlir"},{"output_dir",dir+"/evaluation"},{"target_config",j::Value(config)}};
    std::string request=dir+"/request.json",response=dir+"/response.json";
    if(!write(request,render(req),e))return false;
    ++evaluations;llvm::sys::fs::remove(response);
    std::string executionError;
    int rc=0;
    if (nativeStatic) {
      auto result=nativeAir ? evaluateGenericMLIR(mlir,staticConfig,executionError) : evaluateStaticMLIR(mlir,staticConfig,executionError);
      if (!write(dir+"/mapping-result.json",render(result),e)) return false;
      auto object=result.getAsObject();
      auto best=object ? object->getObject("best_plan") : nullptr;
      bool unsupported=object && object->getString("status")=="unsupported";
      auto value=best && !unsupported ? best->getNumber(nativeAir ? "estimated_ms" : "makespan_ms") : std::nullopt;
      j::Object feedback{{"schema","orbit.search.feedback.v1"},{"backend",o.backend},
          {"candidate_id",nativeBody?s.key:s.p.candidateId},{"source_id",stateSource(s)},
          {"evaluation_identity",s.evaluationIdentity},
          {"evaluation_kind","static_estimate"},{"evaluator_kind","native_cpp"},
          {"executed",false},{"status",value ? "passed" : (!unsupported && nativeAir && best ? "uncalibrated" : "unsupported")},
          {"evidence_path",dir+"/mapping-result.json"}};
      if(value) {
        feedback["plan_id"]=best->getString("plan_id").value_or("").str();
        feedback["score"]=j::Object{{"value",*value},{"unit","ms"},
          {"provenance","static-plan-estimate"},{"physical_hardware_measurement",false}};
      }
      else feedback["diagnostics"]=std::move(result);
      if(!write(response,render(j::Value(std::move(feedback))),e)) return false;
    } else rc=llvm::sys::ExecuteAndWait(executable,{executable,request,response},std::nullopt,{},0,0,&executionError);
    std::string readError; s.status="runtime_failed";
    if(rc==0&&read(response,s.response,readError)){auto r=s.response.getAsObject();
      if(r){auto status=r->getString("status");if(status&&(*status=="unsupported"||*status=="uncalibrated"||*status=="compile_failed"||*status=="runtime_failed"||*status=="numerical_failed"))s.status=status->str();
        if(status&&*status=="passed" && (o.backend!="static" || r->getString("evaluation_kind")=="static_estimate") && r->getString("candidate_id")==stateCandidate(s) && r->getString("source_id")==stateSource(s)){auto score=r->getObject("score");if(score){auto v=score->getNumber("value");auto unit=score->getString("unit");if(v&&std::isfinite(*v)&&*v>0&&unit&&*unit==(o.backend=="amoeba"?"cycles":"ms")){s.status="passed";s.score=*v;}}}
      }
    }else s.response=j::Object{{"reason",executionError+readError},{"exit_code",rc}};
    return true;
  };
  if(archive.empty()){State base; base.p=source;base.nativeMLIR=sourceText;if(!nativeBody&&!transform(base.p,{},e))return false;base.key=stateIdentity(base);base.action="identity";if(!evaluate(base))return false;archive.push_back(base);seen.insert(base.key);if(scored(base))beam.push_back(base);if(!save())return false;}
  for(;nextRound<=o.maxRounds && !beam.empty();){
    std::vector<State> children;
    for(auto &parent:beam)for(auto &action:(nativeBody ? nativeActionsFor(parent) : generateSearchActions(parent.p,preferences))){
      State s;s.p=parent.p;s.nativeMLIR=parent.nativeMLIR;s.parent=parent.key;s.action=action.name;s.round=nextRound;
      s.actionChain=parent.actionChain;s.actionChain.push_back(action.name);
      std::string reason;
      if(nativeBody ? !nativeTransform(s.nativeMLIR,action.name=="split_outer_loops",reason,
                         action.options.tileM?&action.options:nullptr,&action) : (!transform(s.p,action.options,reason)||!validate(s.p,reason))){attempts.push_back(j::Object{{"round",nextRound},{"parent",parent.key},{"action",action.name},{"status","semantic_invalid"},{"reason",reason}});continue;}
      s.key=stateIdentity(s);
      if(!seen.insert(s.key).second){attempts.push_back(j::Object{{"round",nextRound},{"parent",parent.key},{"action",action.name},{"search_key",s.key},{"candidate_id",nativeBody?s.key:s.p.candidateId},{"status","duplicate"}});
        for(auto &old:archive)if(old.key==s.key&&old.round==nextRound&&scored(old))children.push_back(old);continue;}
      if(!evaluate(s))return false;archive.push_back(s);if(scored(s))children.push_back(s);if(!save())return false;
    }
    beam=selectSearchBeam(std::move(children),o.beamWidth,diversitySlots,[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,[](auto &s){return s.key;},structuralConfiguration);
    j::Array selected;for(auto &s:beam)selected.push_back(j::Object{{"search_key",s.key},{"candidate_id",nativeBody?s.key:s.p.candidateId},{"score",s.score},{"structural_configuration",structuralConfiguration(s)},{"reason",diversitySlots ? "evaluated legal child selected by cost plus reserved structural-diversity slots; archive retains global best" : "lowest evaluated score among unique legal children; archive retains global best"}});
    selections.push_back(j::Object{{"round",nextRound},{"selected",std::move(selected)}});++nextRound;if(!save())return false;
  }
  auto ranked=selectSearchBeam(archive,archive.size(),0,[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,[](auto &s){return s.key;},[](auto &s){return s.action;});
  j::Array rows;for(auto &s:archive)rows.push_back(row(s));j::Array ranking;for(auto &s:ranked)ranking.push_back(s.key);
  return write(o.outputDir+"/search.json",render(j::Object{{"schema","orbit.structured.search.v1"},{"backend",o.backend},{"score_unit",o.backend=="amoeba"?"cycles":"ms"},{"binding",binding},{"evaluations",evaluations},{"archive",std::move(rows)},{"ranking",std::move(ranking)},{"selections",std::move(selections)},{"attempts",std::move(attempts)}}),e);
}

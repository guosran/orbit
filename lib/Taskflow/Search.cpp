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
#include <numeric>
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
// Requested source configuration is diagnostic evidence, never a valid plan.
// Failed capacity requests must remain replayable without inventing best_plan.
j::Object nativeRequestedConfiguration(mlir::ModuleOp module) {
  j::Array sizes;
  if(auto attr=module->getAttrOfType<mlir::DenseI64ArrayAttr>("orbit.air_tile_sizes"))
    for(auto size:attr.asArrayRef())sizes.push_back(size);
  ProgramMappingChoice choice;std::string diagnostic;
  bool readable=readNativeMappingChoice(module,choice,diagnostic);
  auto kind=module->getAttrOfType<mlir::StringAttr>("orbit.implementation");
  auto lanes=module->getAttrOfType<mlir::IntegerAttr>("orbit.vector_lanes");
  j::Object result{{"scope","requested canonical source attributes; not a legal mapping witness"},
      {"tile_sizes",std::move(sizes)},{"data_organization",choice.dataOrganization},
      {"residency",choice.residency},{"grouping",choice.grouping},
      {"storage",choice.storage},{"reuse_loop",choice.reuseLoop},{"panel_loop",choice.panelLoop},
      {"implementation",kind?kind.str():"scalar"},{"vector_lanes",lanes?lanes.getInt():1},
      {"mapping_readable",readable},{"diagnostic",diagnostic}};
  if(!choice.taskGroup.empty()){j::Array v;for(auto t:choice.taskGroup)v.push_back(int64_t(t));result["task_group"]=std::move(v);}
  if(!choice.retainedStorages.empty()){j::Array v;for(auto t:choice.retainedStorages)v.push_back(int64_t(t));result["retained_storages"]=std::move(v);}
  if(!choice.residencyGroups.empty()){j::Array groups;for(const auto &g:choice.residencyGroups){j::Array tasks,storages;for(auto t:g.tasks)tasks.push_back(int64_t(t));for(auto x:g.retainedStorages)storages.push_back(int64_t(x));groups.push_back(j::Object{{"tasks",std::move(tasks)},{"retained_storages",std::move(storages)}});}result["residency_groups"]=std::move(groups);}

  if(auto devices=module->getAttrOfType<mlir::ArrayAttr>("orbit.heterogeneous_devices")) {
    j::Array values;for(auto value:devices)if(auto s=mlir::dyn_cast<mlir::StringAttr>(value))values.push_back(s.str());
    result["heterogeneous_devices"]=std::move(values);
    j::Array cuts;if(auto boundaries=module->getAttrOfType<mlir::DenseI64ArrayAttr>("orbit.heterogeneous_cuts"))for(auto cut:boundaries.asArrayRef())cuts.push_back(cut);
    result["region_cuts"]=std::move(cuts);
  }
  return result;
}
j::Value uncalibratedMappingSummary(const j::Value &result) {
  auto object=result.getAsObject();auto best=object?object->getObject("best_plan"):nullptr;
  if(!object || !best)return nullptr;
  j::Object plan{{"estimated_ms",nullptr}};
  for(auto key:{"plan_id","analysis_status","codegen_status","cost_status"})
    plan[key]=best->getString(key).value_or("").str();
  return j::Object{{"status",object->getString("status").value_or("").str()},
      {"reason",object->getString("reason").value_or("").str()},
      {"best_plan",std::move(plan)},
      {"scope","compact legal uncalibrated plan flags; complete mapping-result.json retained separately"}};
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
  auto feedback=s.response.getAsObject();
  const std::string graph=feedback?feedback->getString("program_graph_id").value_or("").str():"";
  const std::string candidate=feedback?feedback->getString("program_candidate_id").value_or("").str():"";
  return j::Object{{"search_key",s.key},{"candidate_id",s.nativeMLIR.empty()?s.p.candidateId:s.key},{"evaluation_identity",s.evaluationIdentity},{"program_graph_id",graph},{"program_candidate_id",candidate},{"parent",s.parent},{"action",s.action},{"action_chain",std::move(chain)},{"round",s.round},{"status",s.status},{"score",scored(s)?j::Value(s.score):j::Value(nullptr)},{"program",s.nativeMLIR.empty()?printPlan(s.p):j::Value(nullptr)},{"native_mlir",s.nativeMLIR.empty()?j::Value(nullptr):j::Value(s.nativeMLIR)},{"evaluation",j::Value(s.response)}};
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
std::vector<SearchAction> orbit::taskflow::generateNativeTileActions(
    const GenericAnalysis &analysis, const std::array<int64_t,3> &current,
    const GenericTargetCapabilities &capabilities,
    const SearchPreferences &preferences, size_t limit) {
  std::array<int64_t,3> dimensions{1,1,1};
  for(const auto &task:analysis.tasks)
    for(size_t axis=0;axis<std::min<size_t>(3,task.loops.size());++axis) {
      const auto &loop=task.loops[axis];
      if(loop.step>0 && loop.upper>loop.lower)
        dimensions[axis]=std::max(dimensions[axis],(loop.upper-loop.lower-1)/loop.step+1);
    }
  const std::array<int64_t,3> preferred{preferences.m,preferences.n,preferences.k};
  std::vector<SearchAction> actions;std::set<std::array<int64_t,3>> seen;
  // Spatial N first preserves the narrow-panel opportunity under a small
  // proposal budget. Axes denote the existing native loop-index interface.
  for(size_t axis: {size_t(1),size_t(2),size_t(0)}) {
    int64_t targetAlignment=capabilities.tileAlignment[axis];
    if(targetAlignment<=0 || preferred[axis]<=0)continue;
    int64_t factor=preferred[axis]/std::gcd(preferred[axis],targetAlignment);
    if(factor>INT64_MAX/targetAlignment)continue;
    int64_t alignment=factor*targetAlignment;
    auto add=[&](int64_t extent) {
      if(actions.size()>=limit || extent<=0 || extent>dimensions[axis])return;
      // A whole dimension smaller than alignment is a scalar proposal; the
      // selected implementation's own tail proof remains the planner's job.
      if(extent!=dimensions[axis])extent=(extent/alignment)*alignment;
      if(extent<=0)return;
      auto tile=current;tile[axis]=extent;
      if(tile==current || !seen.insert(tile).second)return;
      actions.push_back({"auto_tiles_"+std::to_string(tile[0])+"_"+
          std::to_string(tile[1])+"_"+std::to_string(tile[2]),
          {tile[0],tile[1],tile[2],false,false}});
    };
    add(std::max<int64_t>(1,current[axis]/2));
    if(current[axis]<=INT64_MAX/2)add(std::min(dimensions[axis],current[axis]*2));
    add(std::min(dimensions[axis],alignment));
    add(dimensions[axis]);
  }
  return actions;
}
bool orbit::taskflow::runStructuredSearch(const Program &source,const SearchOptions &o,std::string &e) {
  const bool nativeAir = o.backend == "air" && (o.evaluator.empty() || o.evaluator == "native");
  const bool nativeAmoeba = o.backend=="amoeba" && bool(o.nativeEvaluator);
  const bool nativeHeterogeneous=o.backend=="cpu-npu" && bool(o.nativeEvaluator);
  const bool nativePlugin=nativeAmoeba||nativeHeterogeneous;
  const bool nativeStatic = (o.backend == "static" || nativeAir) && (o.evaluator.empty() || o.evaluator == "native");
  const bool nativeInternal=nativeStatic||nativePlugin;
  if((o.backend!="amoeba"&&o.backend!="npu"&&o.backend!="static"&&o.backend!="air"&&o.backend!="cpu-npu")||o.maxRounds<1||o.maxCandidates<1||o.beamWidth<1||o.outputDir.empty()||(!nativeInternal && o.evaluator.empty()) || (nativePlugin && o.nativeBackendIdentity.empty())) {e="invalid structured search options";return false;}
  const bool nativeBody = !o.nativeSourceMLIR.empty();
  if(nativeBody && !nativeAir && !nativePlugin) {e="native-body search requires a registered target evaluator";return false;}
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
    // Normalize pre-existing implementation attributes as well as actions.
    // Default scalar/1 and its explicit legacy spelling are one candidate.
    auto implementation=module->getOperation()->getAttr("orbit.implementation");
    auto vectorLanes=module->getOperation()->getAttr("orbit.vector_lanes");
    if(implementation || vectorLanes) {
      auto kind=mlir::dyn_cast_or_null<mlir::StringAttr>(implementation);
      auto lanes=mlir::dyn_cast_or_null<mlir::IntegerAttr>(vectorLanes);
      if((implementation && !kind) || (vectorLanes && !lanes)) {
        reason="native implementation kind/lanes attributes have invalid types";return false;
      }
      if(!setNativeImplementation(*module,kind?kind.str():"scalar",lanes?lanes.getInt():1,reason))return false;
    }
    if(split&&!splitNativeOuterLoops(*module,reason))return false;
    if(mappingAction && !mappingAction->backendAction.empty() &&
        (!o.nativeApplyAction || !o.nativeApplyAction(*module,*mappingAction,reason)))return false;
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
  if(nativePlugin)for(auto key:{"native_auto_candidates","native_air_tile_candidates","native_mapping_candidates","native_implementation_candidates","native_program_partition_candidates"})
    if(config.getAsObject()->get(key)){e="Amoeba program-graph target does not support AIR action option: "+std::string(key);return false;}
  j::Value staticConfig=nullptr;
  if (nativeInternal) {
    if (auto path=config.getAsObject()->getString("mapping_config")) {
      if (!read(path->str(),staticConfig,e)) return false;
    } else staticConfig=config;
  }
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
    if(nativeHeterogeneous)module->getOperation()->removeAttr("orbit.heterogeneous_cuts");
    std::string text;llvm::raw_string_ostream stream(text);module->print(stream);
    return hash(text);
  };
  bool automatic=false;std::string automaticMode="joint";size_t maxTileProposals=12,maxResidencyProposals=8,maxResidencyGroups=4;int64_t explorationSlots=0;
  GenericTargetCapabilities proposalCapabilities;
  SearchPreferences preferences;
  j::Object automaticProposals;
  if(auto setting=config.getAsObject()->get("native_auto_candidates")) {
    auto object=setting->getAsObject();
    if(!nativeBody || !object) {e="native_auto_candidates requires native source and an object";return false;}
    automatic=true;
    if(auto mode=object->getString("mode"))automaticMode=mode->str();
    if(automaticMode!="joint" && automaticMode!="tiling-only" && automaticMode!="implementation-residency") {
      e="native auto mode must be joint, tiling-only or implementation-residency";return false;
    }
    if(auto count=object->get("max_tile_proposals")) {
      auto value=count->getAsInteger();
      if(!value || *value<1 || *value>64) {e="max_tile_proposals must be an integer in [1,64]";return false;}
      maxTileProposals=size_t(*value);
    }
    for(auto field:{"max_residency_proposals","max_residency_groups"})if(auto count=object->get(field)){
      auto value=count->getAsInteger();if(!value||*value<1||*value>64){e=std::string(field)+" must be an integer in [1,64]";return false;}
      (llvm::StringRef(field)=="max_residency_proposals"?maxResidencyProposals:maxResidencyGroups)=size_t(*value);
    }
    if(auto count=object->get("exploration_slots")) {
      auto value=count->getAsInteger();
      if(!value || *value<0 || *value>o.beamWidth) {e="exploration_slots must be an integer in [0,beam width]";return false;}
      explorationSlots=*value;
    }
    for(auto &entry:*object)if(entry.first!="mode" && entry.first!="max_tile_proposals" && entry.first!="exploration_slots" && entry.first!="max_residency_proposals" && entry.first!="max_residency_groups") {
      e="unknown native_auto_candidates field";return false;
    }
    if(!getGenericTargetCapabilities(staticConfig,proposalCapabilities,e))return false;
    if(proposalCapabilities.tileAlignment.size()!=3 || proposalCapabilities.capacityBytes<=0) {
      e="native automatic proposals require three tile alignments and positive target capacity";return false;
    }
  }
  std::vector<SearchAction> nativeActions;
  if(nativePlugin)nativeActions=o.nativeActions;
  else if(!automatic)nativeActions.push_back({"split_outer_loops",{}});
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
  bool discoverMappings = automatic && automaticMode!="tiling-only";
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
  if(automatic && automaticMode!="tiling-only") {
    SearchAction local{"auto_mapping_phase_local",{}};
    local.mapping=ProgramMappingChoice{};nativeActions.push_back(std::move(local));
  }
  auto nativeActionsFor=[&](const State &state) {
    auto actions=nativeActions;
    if(automatic) {
      auto module=parseNative(state.nativeMLIR);GenericAnalysis analysis;std::string diagnostic;
      j::Array proposals;
      if(module && analyzeGenericProgram(*module,analysis,diagnostic)) {
        if(automaticMode!="implementation-residency") {
          std::array<int64_t,3> current{1,1,1};
          if(auto sizes=module->getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>("orbit.air_tile_sizes")) {
            if(sizes.size()==3)std::copy(sizes.asArrayRef().begin(),sizes.asArrayRef().end(),current.begin());
          } else {
            GenericMappingPlan baseline;
            if(planGenericMapping(analysis,staticConfig,baseline,diagnostic))
              for(const auto &phase:baseline.phases)for(const auto &bound:phase.loops)
                if(bound.loop<3)current[bound.loop]=std::max(current[bound.loop],
                    (bound.upper-bound.lower-1)/analysis.tasks[phase.task].loops[bound.loop].step+1);
          }
          for(auto action:generateNativeTileActions(analysis,current,proposalCapabilities,preferences,maxTileProposals)) {
            auto candidate=parseNative(state.nativeMLIR);GenericAnalysis facts;GenericMappingPlan plan;
            diagnostic.clear();
            bool legal=setNativeAIRTileSizes(*candidate,action.options.tileM,action.options.tileN,action.options.tileK,diagnostic) &&
                analyzeGenericProgram(*candidate,facts,diagnostic) && planGenericMapping(facts,staticConfig,plan,diagnostic);
            std::string candidateText;llvm::raw_string_ostream candidateStream(candidateText);
            candidate->print(candidateStream);candidateStream.flush();
            j::Object proposal{{"action",action.name},{"kind","tiling"},
                {"candidate_id",hash(candidateText)},
                {"requested_configuration",nativeRequestedConfiguration(*candidate)},
                {"planner_supported",legal},{"reason",diagnostic},
                {"target_capability_identity",proposalCapabilities.identity},
                {"capacity_bytes",proposalCapabilities.capacityBytes},
                {"resident_bytes",legal?j::Value(plan.residentBytes):j::Value(nullptr)}};
            if(!legal)proposal["native_mlir"]=candidateText;
            proposals.push_back(std::move(proposal));
            // All unavailable proposals are reported. Only supported complete
            // source/target plans consume the finite evaluation budget.
            if(legal)actions.push_back(std::move(action));
          }
        }
        if(automaticMode!="tiling-only")for(const auto &implementation:proposalCapabilities.implementations) {
          SearchAction action{"auto_implementation_"+implementation.kind+"_"+std::to_string(implementation.lanes),{}};
          action.implementation=implementation.kind;action.vectorLanes=implementation.lanes;
          actions.push_back(std::move(action));
        }
      }
      automaticProposals[state.key]=std::move(proposals);
      actions.erase(std::remove_if(actions.begin(),actions.end(),[&](const auto &action) {
        return automaticMode=="tiling-only" ? (!action.implementation.empty() || bool(action.mapping) || action.partitionExtent>=0)
             : automaticMode=="implementation-residency" && action.options.tileM;
      }),actions.end());
    }
    if (discoverMappings && (!automatic || automaticMode!="tiling-only")) {
      auto module=parseNative(state.nativeMLIR); GenericAnalysis analysis;
      std::string diagnostic;
      if (module && analyzeGenericProgram(*module,analysis,diagnostic))
        for (auto choice:discoverGenericMappingChoices(analysis,maxResidencyProposals,maxResidencyGroups)) {
          if (choice.residency=="phase-local") continue;
          std::string name=choice.grouping=="shape-major"?"mapping_shape_major":choice.grouping=="tile-major"?"mapping_tile_major":choice.residency=="whole-program"?"mapping_whole_program":
              choice.residency=="tiled-phase-local"?"mapping_tiled_phase_local":
              "mapping_"+choice.dataOrganization+"_storage_"+
              std::to_string(choice.storage)+"_reuse_"+
              std::to_string(choice.reuseLoop)+"_panel_"+std::to_string(choice.panelLoop);
          if(!choice.residencyGroups.empty()){name="mapping_residency";for(const auto &group:choice.residencyGroups){name+="_tasks";for(auto t:group.tasks)name+="_"+std::to_string(t);name+="_data";for(auto x:group.retainedStorages)name+="_"+std::to_string(x);}}
          SearchAction action{name,{}};action.mapping=choice;
          if(automatic) {
            auto candidate=parseNative(state.nativeMLIR);GenericAnalysis facts;GenericMappingPlan plan;diagnostic.clear();
            bool legal=setNativeMappingChoice(*candidate,choice,diagnostic) &&
                analyzeGenericProgram(*candidate,facts,diagnostic) && planGenericMapping(facts,staticConfig,plan,diagnostic);
            std::string text;llvm::raw_string_ostream stream(text);candidate->print(stream);stream.flush();
            j::Object proposal{{"action",name},{"kind","data-organization"},{"candidate_id",hash(text)},
                {"requested_configuration",nativeRequestedConfiguration(*candidate)},{"planner_supported",legal},
                {"reason",diagnostic},{"target_capability_identity",proposalCapabilities.identity},
                {"capacity_bytes",proposalCapabilities.capacityBytes},
                {"resident_bytes",legal?j::Value(plan.residentBytes):j::Value(nullptr)}};
            if(!legal)proposal["native_mlir"]=text;
            automaticProposals[state.key].getAsArray()->push_back(std::move(proposal));
            if(!legal)continue;
          }
          actions.push_back(std::move(action));
        }
    }
    return actions;
  };
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
  auto explorable=[&](const State &state) {
    if(!automatic || explorationSlots<=0 || state.status!="uncalibrated")return false;
    auto feedback=state.response.getAsObject();
    auto diagnostics=feedback?feedback->getObject("diagnostics"):nullptr;
    auto plan=diagnostics?diagnostics->getObject("best_plan"):nullptr;
    return feedback && feedback->getString("evaluator_kind")=="native_cpp" &&
        feedback->getString("status")=="uncalibrated" && !feedback->get("score") &&
        diagnostics && diagnostics->getString("status")!="unsupported" && plan &&
        !plan->getNumber("estimated_ms");
  };
  auto parentEligible=[&](const State &state){return scored(state)||explorable(state);};
  auto selectParents=[&](std::vector<State> candidates) {
    auto less=[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;};
    int64_t scoreWidth=o.beamWidth-explorationSlots;
    auto selected=selectSearchBeam(candidates,scoreWidth,std::min(diversitySlots,scoreWidth),less,
        scored,[](auto &s){return s.key;},structuralConfiguration);
    std::set<std::string> keys;for(const auto &state:selected)keys.insert(state.key);
    std::stable_sort(candidates.begin(),candidates.end(),[](auto &a,auto &b){return a.key<b.key;});
    int64_t explored=0;
    for(const auto &state:candidates)if(explorable(state) && explored<explorationSlots && keys.insert(state.key).second) {
      selected.push_back(state);++explored;
    }
    auto remaining=selectSearchBeam(candidates,o.beamWidth,diversitySlots,less,scored,
        [](auto &s){return s.key;},structuralConfiguration);
    for(const auto &state:remaining)if(selected.size()<size_t(o.beamWidth) && keys.insert(state.key).second)
      selected.push_back(state);
    return selected;
  };
  std::string executable, evaluatorIdentity = "native-cpp-program-data-organization-v8";
  if (nativePlugin) evaluatorIdentity=o.nativeBackendIdentity;
  else if (!nativeInternal) {
    auto found = llvm::sys::findProgramByName(o.evaluator);
    if (!found) { e="evaluator executable not found"; return false; }
    executable=*found;
    auto bytes=llvm::MemoryBuffer::getFile(executable);
    if (!bytes) { e="cannot hash evaluator"; return false; }
    evaluatorIdentity=(*bytes)->getBuffer().str();
  }
  std::string dependencyBinding;bindFiles(config,dependencyBinding);
  dependencyBinding+=fileDigest(llvm::sys::fs::getMainExecutable(nullptr,(void*)&runStructuredSearch));
  const std::string evaluationBinding=hash(dependencyBinding+o.backend+canonicalJSON(config)+evaluatorIdentity);
  const auto evaluationIdentityFor=[&](const State &s){return hash(evaluationBinding+stateIdentity(s));};
  auto evaluateNative=[&](const std::string &text,const std::string &directory,std::string &reason) {
    return nativePlugin ? o.nativeEvaluator(text,staticConfig,directory,reason) :
      (nativeAir ? evaluateGenericMLIR(text,staticConfig,reason) : evaluateStaticMLIR(text,staticConfig,reason));
  };
  const char *scoreField=nativeAmoeba?"makespan_cycles":((nativeAir||nativeHeterogeneous)?"estimated_ms":"makespan_ms");
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
  auto checkGraphArtifacts=[&](const State &state,const j::Value &result) {
    if(!nativeAir && !nativePlugin)return true;
    auto object=result.getAsObject();auto expected=object?object->get("program_graph"):nullptr;
    auto dot=object?object->getString("program_graph_dot"):std::nullopt;
    auto id=object?object->getString("program_graph_id"):std::nullopt;
    auto feedback=state.response.getAsObject();
    j::Value savedGraph=nullptr;std::string reason;
    auto graphFile=o.outputDir+"/candidates/"+state.key+"/program-graph.json";
    auto dotFile=llvm::MemoryBuffer::getFile(o.outputDir+"/candidates/"+state.key+"/program-graph.dot");
    std::string programCandidate=id?staticFingerprint(j::Object{{"source_id",stateSource(state)},{"program_graph_id",id->str()}}):"";
    if(!expected || !dot || !id || !feedback || feedback->getString("program_graph_id")!=id ||
        feedback->getString("program_candidate_id")!=programCandidate ||
        !read(graphFile,savedGraph,reason) || canonicalJSON(savedGraph)!=canonicalJSON(*expected) ||
        !dotFile || (*dotFile)->getBuffer()!=*dot) {
      e="checkpoint program graph artifact/identity failed independent re-evaluation";return false;
    }
    return true;
  };
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
      if(nativeAir) if(auto feedback=s.response.getAsObject()) {
        if(x->getString("program_graph_id").value_or("")!=feedback->getString("program_graph_id").value_or("") ||
           x->getString("program_candidate_id").value_or("")!=feedback->getString("program_candidate_id").value_or("")) {
          e="checkpoint archive common graph/candidate identity differs from feedback";return false;
        }
      }
      if(auto chain=x->getArray("action_chain"))for(auto &action:*chain) {
        auto name=action.getAsString();if(!name){e="invalid action chain";return false;}
        s.actionChain.push_back(name->str());
      }
      if(s.key!=stateIdentity(s)|| (scored(s)&&(!std::isfinite(s.score)||s.score<=0))){e="checkpoint identity/score consistency failed";return false;}
      if(s.evaluationIdentity!=evaluationIdentityFor(s)){e="checkpoint target/model evaluation identity mismatch";return false;}
      if(scored(s)) {
        auto r=s.response.getAsObject();auto score=r?r->getObject("score"):nullptr;
        if(!r||(o.backend=="static" && r->getString("evaluation_kind")!="static_estimate")||r->getString("status")!="passed"||r->getString("candidate_id")!=stateCandidate(s)||r->getString("source_id")!=stateSource(s)||!score||score->getString("unit")!=(o.backend=="amoeba"?"cycles":"ms")||score->getNumber("value")!=s.score){e="checkpoint evaluation consistency failed";return false;}
        if (nativeInternal) {
          if(r->getString("evaluation_identity")!=s.evaluationIdentity){e="checkpoint feedback evaluation identity mismatch";return false;}
          std::string diagnostic;
          std::string revalidation=o.outputDir+"/candidates/"+s.key+"/revalidation";
          if(nativePlugin){llvm::SmallString<256> fresh;if(auto code=llvm::sys::fs::createUniqueDirectory(revalidation,fresh)){e=code.message();return false;}revalidation=fresh.str().str();}
          auto replanned=evaluateNative(stateMLIR(s),revalidation,diagnostic);
          if(!checkGraphArtifacts(s,replanned))return false;
          auto result=replanned.getAsObject();
          auto best=result ? result->getObject("best_plan") : nullptr;
          if(!best || result->getString("status")=="unsupported" ||
              best->getNumber(scoreField)!=s.score ||
              r->getString("plan_id")!=best->getString("plan_id") ||
              r->getString("evaluator_kind")!="native_cpp") {
            e="native checkpoint score failed independent re-evaluation";return false;
          }
          if((nativeAir && best->getObject("direct_air_cost")) || nativeHeterogeneous) {
            const bool observed=best->getString("cost_status")=="measured";
            const bool predicted=best->getString("cost_status")=="calibrated";
            const bool mechanism=best->getString("cost_status")=="assumption-driven-estimate" || best->getString("cost_status")=="mechanism-estimate-with-declared-provenance";
            if(r->getString("evaluation_kind")!=(mechanism?"mechanism_estimate":observed?"measured_feedback":(predicted?"model_prediction":"static_estimate")) ||
               score->getString("provenance")!=(mechanism?"declared-event-mechanism-estimate":observed?"exact-observation-table":(predicted?(nativeHeterogeneous?"frozen-heterogeneous-region-model":"calibrated-direct-air-model"):"static-plan-estimate")) ||
               score->getBoolean("physical_hardware_measurement")!=observed ||
               r->getBoolean("executed")!=false) {
              e="native checkpoint measurement/prediction provenance mismatch";return false;
            }
          }
        }
      }
      if(automatic && explorationSlots>0 && s.status=="uncalibrated") {
        if(!x->get("score") || render(*x->get("score"))!="null") {
          e="uncalibrated exploration checkpoint contains a fabricated score";return false;
        }
        auto feedback=s.response.getAsObject();std::string diagnostic;
        auto replanned=evaluateGenericMLIR(stateMLIR(s),staticConfig,diagnostic);
        if(!checkGraphArtifacts(s,replanned))return false;
        auto result=replanned.getAsObject();auto best=result?result->getObject("best_plan"):nullptr;
        auto old=feedback?feedback->get("diagnostics"):nullptr;
        if(!explorable(s) || !feedback || feedback->getString("candidate_id")!=stateCandidate(s) ||
           feedback->getString("source_id")!=stateSource(s) ||
           feedback->getString("evaluation_identity")!=s.evaluationIdentity ||
           !result || result->getString("status")=="unsupported" || !best ||
           best->getNumber("estimated_ms") || !old) {
          e="uncalibrated exploration checkpoint failed independent plan re-evaluation";return false;
        }
        if(auto digest=feedback->getString("mapping_result_canonical_sha256")) {
          if(*digest!=hash(canonicalJSON(replanned)) ||
             feedback->getString("plan_id")!=best->getString("plan_id") ||
             feedback->getString("feedback_storage")!="native-auto-uncalibrated-summary-v1" ||
             feedback->getBoolean("physical_hardware_measurement")!=false ||
             feedback->getBoolean("executed")!=false ||
             canonicalJSON(*old)!=canonicalJSON(uncalibratedMappingSummary(replanned))) {
            e="uncalibrated exploration compact plan/digest consistency failed";return false;
          }
        } else if(canonicalJSON(*old)!=canonicalJSON(replanned)) {
          e="uncalibrated exploration checkpoint failed independent plan re-evaluation";return false;
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
        if(!parentEligible(*parent) || state.round!=parent->round+1) {
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
    if(parentEligible(*baseline)) expectedBeam.push_back(baseline->key);
    for(size_t i=0;i<selections.size();++i) {
      if(expectedBeam.empty()) {e="checkpoint completed a round without an eligible parent beam";return false;}
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
      auto recomputed=selectParents(std::move(candidates));
      if(recomputed.size()!=selected->size()) {e="checkpoint selected beam differs from scored round";return false;}
      expectedBeam.clear();
      for(size_t j=0;j<selected->size();++j) {
        auto row=(*selected)[j].getAsObject();
        if(!row || row->getString("search_key")!=recomputed[j].key ||
           (scored(recomputed[j]) ? row->getNumber("score")!=recomputed[j].score :
             (!row->get("score") || render(*row->get("score"))!="null" ||
              row->getString("reason")!="uncalibrated-exploration"))) {
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
      if(state==archive.end() || !parentEligible(*state)) {e="invalid checkpoint beam";return false;}
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
    if(nativeBody) {
      auto module=parseNative(mlir);
      (*req.getAsObject())["requested_configuration"]=nativeRequestedConfiguration(*module);
    }
    std::string request=dir+"/request.json",response=dir+"/response.json";
    if(!write(request,render(req),e))return false;
    ++evaluations;llvm::sys::fs::remove(response);
    std::string executionError;
    int rc=0;
    if (nativeInternal) {
      auto result=evaluateNative(mlir,dir+"/evaluation",executionError);
      if (!write(dir+"/mapping-result.json",render(result),e)) return false;
      if(auto value=result.getAsObject()) {
        if(auto graph=value->get("program_graph"))
          if(!write(dir+"/program-graph.json",render(*graph),e))return false;
        if(auto dot=value->getString("program_graph_dot"))
          if(!write(dir+"/program-graph.dot",*dot,e))return false;
      }
      auto object=result.getAsObject();
      auto best=object ? object->getObject("best_plan") : nullptr;
      bool unsupported=object && object->getString("status")=="unsupported";
      auto value=best && !unsupported ? best->getNumber(scoreField) : std::nullopt;
      j::Object feedback{{"schema","orbit.search.feedback.v1"},{"backend",o.backend},
          {"candidate_id",nativeBody?s.key:s.p.candidateId},{"source_id",stateSource(s)},
          {"evaluation_identity",s.evaluationIdentity},
          {"evaluation_kind","static_estimate"},{"evaluator_kind","native_cpp"},
          {"executed",false},{"status",value ? "passed" : (!unsupported && nativeAir && best ? "uncalibrated" : "unsupported")},
          {"evidence_path",dir+"/mapping-result.json"}};
      if(object && object->getString("program_graph_id")) {
        auto graph=object->getString("program_graph_id")->str();
        auto programCandidate=staticFingerprint(j::Object{{"source_id",stateSource(s)},{"program_graph_id",graph}});
        feedback["program_graph_id"]=graph;feedback["program_candidate_id"]=programCandidate;
        (*req.getAsObject())["program_graph_id"]=graph;
        (*req.getAsObject())["program_candidate_id"]=programCandidate;
        if(!write(request,render(req),e))return false;
      }
      if(value) {
        const bool observed=nativeAir && best->getString("cost_status")=="measured";
        const bool predicted=(nativeAir||nativeHeterogeneous) && best->getString("cost_status")=="calibrated";
        const bool mechanism=best->getString("cost_status")=="assumption-driven-estimate" || best->getString("cost_status")=="mechanism-estimate-with-declared-provenance";
        feedback["evaluation_kind"]=mechanism?"mechanism_estimate":observed?"measured_feedback":(predicted?"model_prediction":"static_estimate");
        feedback["plan_id"]=best->getString("plan_id").value_or("").str();
        feedback["score"]=j::Object{{"value",*value},{"unit",nativeAmoeba?"cycles":"ms"},
          {"provenance",nativeAmoeba?"amoeba-program-graph-provider-and-production-schedule":(mechanism?"declared-event-mechanism-estimate":observed?"exact-observation-table":(predicted?(nativeHeterogeneous?"frozen-heterogeneous-region-model":"calibrated-direct-air-model"):"static-plan-estimate"))},
          {"physical_hardware_measurement",observed}};
        if(observed) feedback["measurement_reuse"]="SHA-bound prior complete-command observations; this evaluation does not submit hardware";
      }
      else if(automatic && explorationSlots>0 && nativeAir && best && !unsupported) {
        feedback["feedback_storage"]="native-auto-uncalibrated-summary-v1";
        feedback["mapping_result_canonical_sha256"]=hash(canonicalJSON(result));
        feedback["plan_id"]=best->getString("plan_id").value_or("").str();
        feedback["physical_hardware_measurement"]=false;
        feedback["diagnostics"]=uncalibratedMappingSummary(result);
      } else feedback["diagnostics"]=std::move(result);
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
  if(archive.empty()){State base; base.p=source;base.nativeMLIR=sourceText;if(!nativeBody&&!transform(base.p,{},e))return false;base.key=stateIdentity(base);base.action="identity";if(!evaluate(base))return false;archive.push_back(base);seen.insert(base.key);if(parentEligible(base))beam.push_back(base);if(!save())return false;}
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
        for(auto &old:archive)if(old.key==s.key&&old.round==nextRound&&parentEligible(old))children.push_back(old);continue;}
      if(!evaluate(s))return false;archive.push_back(s);if(parentEligible(s))children.push_back(s);if(!save())return false;
    }
    beam=selectParents(std::move(children));
    j::Array selected;for(auto &s:beam)selected.push_back(j::Object{{"search_key",s.key},{"candidate_id",nativeBody?s.key:s.p.candidateId},{"score",scored(s)?j::Value(s.score):j::Value(nullptr)},{"structural_configuration",structuralConfiguration(s)},{"reason",explorable(s)?"uncalibrated-exploration":diversitySlots ? "evaluated legal child selected by cost plus reserved structural-diversity slots; archive retains global best" : "lowest evaluated score among unique legal children; archive retains global best"}});
    selections.push_back(j::Object{{"round",nextRound},{"selected",std::move(selected)}});++nextRound;if(!save())return false;
  }
  if(automatic && !write(o.outputDir+"/automatic-proposals.json",render(j::Object{
      {"schema","orbit.native.automatic-proposals.v1"},{"mode",automaticMode},{"exploration_slots",explorationSlots},
      {"target_capability_identity",proposalCapabilities.identity},
      {"parents",std::move(automaticProposals)}}),e))return false;
  auto ranked=selectSearchBeam(archive,archive.size(),0,[](auto &a,auto &b){return a.score!=b.score?a.score<b.score:a.key<b.key;},scored,[](auto &s){return s.key;},[](auto &s){return s.action;});
  j::Array rows;for(auto &s:archive)rows.push_back(row(s));j::Array ranking;for(auto &s:ranked)ranking.push_back(s.key);
  return write(o.outputDir+"/search.json",render(j::Object{{"schema","orbit.structured.search.v1"},{"backend",o.backend},{"score_unit",o.backend=="amoeba"?"cycles":"ms"},{"binding",binding},{"evaluations",evaluations},{"archive",std::move(rows)},{"ranking",std::move(ranking)},{"selections",std::move(selections)},{"attempts",std::move(attempts)}}),e);
}

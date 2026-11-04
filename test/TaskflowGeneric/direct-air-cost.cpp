#include "Taskflow/GenericMapping.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace orbit::taskflow;
// Coefficients below are synthetic API contracts, never hardware calibration.
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  mlir::DialectRegistry registry;
  registry.insert<mlir::taskflow::TaskflowDialect,mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,mlir::math::MathDialect,mlir::memref::MemRefDialect,
      mlir::func::FuncDialect,mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  GenericAnalysis a; GenericMappingPlan plain,planned;std::string error;
  unsigned checks=0;bool ok=true;
  auto expect=[&](bool value,const char *name){++checks;if(!value){ok=false;llvm::errs()<<name<<": "<<error<<"\n";}error.clear();};
  if(!module || !analyzeGenericProgram(*module,a,error)) {llvm::errs()<<error;return 1;}
  llvm::json::Value config=llvm::json::Object{{"generic_air",llvm::json::Object{{"tile_sizes",llvm::json::Array{4}}}}};
  if(!planGenericMapping(a,config,plain,error)){llvm::errs()<<error;return 1;}
  GenericCostFeatures f;
  expect(collectGenericCostFeatures(a,plain,f,error),"native source facts");
  expect(f.values["source_scalar_f32_operations"]==16 && f.values["source_f32_product_operations"]==0 &&
         f.values["source_math_exp_f32_operations"]==0 && f.values["source_scalar_reads"]==32 &&
         f.values["source_scalar_writes"]==16 && f.values["dma_bytes"]==192 &&
         f.values["command_count"]==1 && f.values["hardware_stage_count"]==1,"actual add counts");
  expect(!serializeGenericMapping(a,plain).getAsObject()->get("direct_air_cost"),"default serialization unchanged");
  llvm::json::Value target=llvm::json::Object{{"id","synthetic"},{"toolchain","test"},{"runtime","test"},{"boundary","prepared-command-host-to-host"}};
  auto model=[&](){
    llvm::json::Array sources,impl,ops;
    for(auto &s:f.sourceFingerprints)sources.push_back(s);
    for(auto &s:f.implementations)impl.push_back(s);
    for(auto &s:f.operations)ops.push_back(s.first);
    return llvm::json::Value(llvm::json::Object{
      {"schema","orbit.direct-air.cost.v1"},{"kind","linear-composite"},{"calibration","calibrated"},
      {"model_id","synthetic-contract"},{"model_version","1"},{"training_manifest_digest",std::string(64,'0')},
      {"boundary","prepared-command-host-to-host"},
      {"target",llvm::json::Object{{"id","synthetic"},{"toolchain","test"},{"runtime","test"}}},
      {"domain",llvm::json::Object{{"source_fingerprints",std::move(sources)},{"implementations",std::move(impl)},
        {"operations",std::move(ops)},{"feature_ranges",llvm::json::Object{{"hardware_stage_count",llvm::json::Array{1,1}}}}}},
      {"terms",llvm::json::Array{llvm::json::Object{{"feature","command_count"},{"coefficient_ms",0.1}},
         llvm::json::Object{{"feature","source_scalar_f32_operations"},{"coefficient_ms",0.01}},
         llvm::json::Object{{"feature","hardware_stage_count"},{"coefficient_ms",0.2}},
         llvm::json::Object{{"feature","dma_bytes"},{"coefficient_ms",0.001}}}}});
  };
  GenericDirectAIRCost c;
  auto m=model();
  expect(evaluateGenericDirectAIRCost(f,m,target,c,error) && c.valueMs && std::abs(*c.valueMs-0.652)<1e-12,"four-term synthetic composite");
  auto apply=[&](const llvm::json::Value &value){
    auto cfg=llvm::json::Value(config);auto cfgAir=cfg.getAsObject()->getObject("generic_air");
    (*cfgAir)["cost_model"]=llvm::json::Value(value);(*cfgAir)["cost_target"]=llvm::json::Value(target);
    return planGenericMapping(a,cfg,planned,error);
  };
  expect(apply(m) && checkGenericMapping(a,planned,error),"checked opt-in plan");
  auto original=planned;
  expect(planned.planId!=plain.planId && planned.hasCost,"model bound to identity");
  auto report=serializeGenericMapping(a,planned);
  expect(report.getAsObject()->get("theoretical_coefficients")->kind()==llvm::json::Value::Null,"composite not legacy theory");
  auto mutation=[&](const char *name,auto change){auto p=original;change(p);expect(!checkGenericMapping(a,p,error),name);};
  mutation("feature mutation",[](auto &p){++p.directCost.features.values["dma_bytes"];});
  mutation("source identity mutation",[](auto &p){p.directCost.features.sourceFingerprints={"foreign"};});
  mutation("feature digest mutation",[](auto &p){p.directCost.featureDigest="foreign";});
  mutation("model digest mutation",[](auto &p){p.directCost.modelDigest="foreign";});
  mutation("estimate mutation",[](auto &p){p.estimatedMs+=1;});
  mutation("component double charge",[](auto &p){p.theoreticalPerByteMs=1;});
  mutation("provider value mutation",[](auto &p){p.directCost.valueMs=7;});
  auto bad=[&](const char *name,auto change){auto v=model();change(*v.getAsObject());expect(!evaluateGenericDirectAIRCost(f,v,target,c,error),name);};
  bad("negative coefficient",[](auto &v){(*v.getArray("terms"))[0].getAsObject()->operator[]("coefficient_ms")=-1.0;});
  bad("too many terms",[](auto &v){v.getArray("terms")->push_back(llvm::json::Value((*v.getArray("terms"))[0]));});
  bad("duplicate term",[](auto &v){(*v.getArray("terms"))[1]=llvm::json::Value((*v.getArray("terms"))[0]);});
  bad("unknown feature",[](auto &v){(*v.getArray("terms"))[0].getAsObject()->operator[]("feature")="unknown";});
  bad("missing provenance",[](auto &v){v.erase("training_manifest_digest");});
  auto ood=[&](const char *name,auto change){auto v=model();change(*v.getAsObject());expect(evaluateGenericDirectAIRCost(f,v,target,c,error) && c.status=="out_of_domain" && !c.valueMs,name);};
  ood("source OOD",[](auto &v){(*v.getObject("domain"))["source_fingerprints"]=llvm::json::Array{"foreign"};});
  ood("implementation OOD",[](auto &v){(*v.getObject("domain"))["implementations"]=llvm::json::Array{"foreign"};});
  ood("operation OOD",[](auto &v){(*v.getObject("domain"))["operations"]=llvm::json::Array{};});
  ood("range OOD",[](auto &v){(*v.getObject("domain"))["feature_ranges"]=llvm::json::Object{{"hardware_stage_count",llvm::json::Array{2,3}}};});
  ood("runtime OOD",[](auto &v){(*v.getObject("target"))["runtime"]="foreign";});
  auto multi=f;multi.values["command_count"]=2;
  expect(evaluateGenericDirectAIRCost(multi,m,target,c,error) && !c.valueMs && c.status=="out_of_domain","partition host cost unknown");
  llvm::json::Value featuresOnly=llvm::json::Object{{"schema","orbit.direct-air.cost.v1"},{"kind","features-only"}};
  expect(apply(featuresOnly) && !planned.hasCost && planned.directCost.status=="uncalibrated" && checkGenericMapping(a,planned,error),"features only unknown");
  auto m2=model();(*m2.getAsObject())["model_version"]="2";
  expect(apply(m2) && planned.planId!=original.planId,"model version identity");
  auto invalid=f;invalid.values["dma_bytes"]=-1;
  expect(!evaluateGenericDirectAIRCost(invalid,m,target,c,error),"negative native fact");

  auto tightened=model();
  llvm::json::Object perSource;
  for(const auto &source:f.sourceFingerprints)perSource[source]=llvm::json::Object{
    {"max_tile_loop_0_iterations",llvm::json::Array{4,4}},
    {"dma_bytes",llvm::json::Array{192,192}}, {"serial_tiled",llvm::json::Array{1,1}}};
  (*tightened.getAsObject()->getObject("domain"))["source_feature_ranges"]=std::move(perSource);
  expect(evaluateGenericDirectAIRCost(f,tightened,target,c,error) && c.valueMs,"source configuration accepted");
  auto tileMismatch=f;tileMismatch.values["max_tile_loop_0_iterations"]=8;
  expect(evaluateGenericDirectAIRCost(tileMismatch,tightened,target,c,error) && !c.valueMs && c.status=="out_of_domain","source tile OOD");
  auto emptySource=llvm::json::Value(tightened);
  (*emptySource.getAsObject()->getObject("domain"))["source_feature_ranges"]=llvm::json::Object{};
  expect(!evaluateGenericDirectAIRCost(f,emptySource,target,c,error),"empty source configuration rejects");
  auto badSource=llvm::json::Value(tightened);
  (*badSource.getAsObject()->getObject("domain"))["source_feature_ranges"]=llvm::json::Object{{"undeclared",llvm::json::Object{{"dma_bytes",llvm::json::Array{0,1}}}}};
  expect(!evaluateGenericDirectAIRCost(f,badSource,target,c,error),"undeclared source configuration rejects");
  auto missingSource=f;missingSource.sourceFingerprints.push_back("additional");
  auto missingModel=llvm::json::Value(tightened);
  missingModel.getAsObject()->getObject("domain")->getArray("source_fingerprints")->push_back("additional");
  expect(evaluateGenericDirectAIRCost(missingSource,missingModel,target,c,error) && !c.valueMs,"unconfigured source OOD");
  auto sameUnpriced=original;sameUnpriced.planId="different-price-identity";
  sameUnpriced.estimatedMs=99;sameUnpriced.theoreticalPerByteMs=7;sameUnpriced.costProvenance="different price";
  GenericCostFeatures unchanged;
  expect(collectGenericCostFeatures(a,sameUnpriced,unchanged,error) && unchanged.executionFingerprint==f.executionFingerprint,"price independent execution identity");
  auto altered=plain;altered.capacityBytes+=1;GenericCostFeatures alteredFeatures;
  expect(collectGenericCostFeatures(a,altered,alteredFeatures,error) && alteredFeatures.executionFingerprint!=f.executionFingerprint,"resource execution identity");
  auto observed=[&](){return llvm::json::Value(llvm::json::Object{
    {"schema","orbit.direct-air.cost.v1"},{"kind","observed-table"},{"model_id","synthetic-observations"},{"model_version","1"},
    {"boundary","prepared-command-host-to-host"},{"target",llvm::json::Value(llvm::json::Object(*m.getAsObject()->getObject("target")))},
    {"rows",llvm::json::Array{llvm::json::Object{{"execution_fingerprint",f.executionFingerprint},
      {"median_ms",3.25},{"raw_manifest_sha256",std::string(64,'a')},{"sample_count",6},
      {"boundary","prepared-command-host-to-host"},{"target",llvm::json::Value(llvm::json::Object(*m.getAsObject()->getObject("target")))}}}}});};
  auto observations=observed();
  expect(evaluateGenericDirectAIRCost(f,observations,target,c,error) && c.status=="measured" && c.valueMs==3.25,"exact measured feedback");
  expect(apply(observations) && planned.hasCost && planned.directCost.status=="measured" && checkGenericMapping(a,planned,error),"observed checked plan");
  expect(planned.directCost.features.executionFingerprint==f.executionFingerprint,"priced plan stable execution key");
  auto miss=f;miss.executionFingerprint=std::string(64,'b');
  expect(evaluateGenericDirectAIRCost(miss,observations,target,c,error) && c.status=="out_of_domain" && !c.valueMs,"count-identical execution miss unknown");
  auto malformedObserved=[&](const char *name,auto change){auto v=observed();change(*v.getAsObject());expect(!evaluateGenericDirectAIRCost(f,v,target,c,error),name);};
  malformedObserved("duplicate observation",[](auto &v){v.getArray("rows")->push_back(llvm::json::Value((*v.getArray("rows"))[0]));});
  malformedObserved("bad raw digest",[](auto &v){(*v.getArray("rows"))[0].getAsObject()->operator[]("raw_manifest_sha256")="unbound";});
  malformedObserved("negative observed median",[](auto &v){(*v.getArray("rows"))[0].getAsObject()->operator[]("median_ms")=-1;});
  malformedObserved("zero sample count",[](auto &v){(*v.getArray("rows"))[0].getAsObject()->operator[]("sample_count")=0;});
  malformedObserved("observation target differs",[](auto &v){(*v.getArray("rows"))[0].getAsObject()->getObject("target")->operator[]("runtime")="foreign";});
  malformedObserved("observation boundary differs",[](auto &v){(*v.getArray("rows"))[0].getAsObject()->operator[]("boundary")="device-only";});
  auto foreignTarget=llvm::json::Value(target);(*foreignTarget.getAsObject())["runtime"]="foreign";
  expect(evaluateGenericDirectAIRCost(f,observations,foreignTarget,c,error) && !c.valueMs,"observation runtime OOD");
  auto observed2=observed();(*observed2.getAsObject())["model_version"]="2";
  auto observedPlanId=planned.planId;
  expect(apply(observed2) && planned.planId!=observedPlanId && planned.directCost.features.executionFingerprint==f.executionFingerprint,"observations checkpoint identity");
  auto implementationModel=[&](){
    auto value=model();auto &o=*value.getAsObject();o["kind"]="implementation-composite";
    llvm::json::Object ranges;
    for(const auto &entry:f.values)ranges[entry.first]=llvm::json::Array{entry.second,entry.second};
    llvm::json::Array ops;for(const auto &entry:f.operations)ops.push_back(entry.first);
    o["domain"]=llvm::json::Object{{"family_fingerprints",llvm::json::Array{f.regionFamilyFingerprint}},
      {"architecture_fingerprints",llvm::json::Array{f.architectureFingerprint}},
      {"implementations",llvm::json::Array{"scalar:1","scalar-product-vector-add-f32:8"}},
      {"operations",std::move(ops)},{"allow_tails",false},{"feature_ranges",std::move(ranges)}};
    o["terms"]=llvm::json::Array{
      llvm::json::Object{{"feature","command_count"},{"coefficient_ms",0.1}},
      llvm::json::Object{{"feature","implemented_scalar_f32_product_operations"},{"coefficient_ms",0.01}},
      llvm::json::Object{{"feature","implemented_mixed8_f32_product_operations"},{"coefficient_ms",0.005}},
      llvm::json::Object{{"feature","hardware_stage_count"},{"coefficient_ms",0.2}}};
    return value;
  };
  auto im=implementationModel();
  expect(evaluateGenericDirectAIRCost(f,im,target,c,error) && c.valueMs && std::abs(*c.valueMs-.3)<1e-12,"implementation composite four terms");
  auto otherSource=f;otherSource.sourceFingerprints={std::string(64,'c')};
  expect(evaluateGenericDirectAIRCost(otherSource,im,target,c,error) && c.valueMs,"family admits new source fingerprint");
  auto implementationOOD=[&](const char *name,auto change){auto facts=f;change(facts);
    expect(evaluateGenericDirectAIRCost(facts,im,target,c,error) && !c.valueMs && c.status=="out_of_domain",name);};
  implementationOOD("family OOD",[](auto &v){v.regionFamilyFingerprint=std::string(64,'d');});
  implementationOOD("architecture OOD",[](auto &v){v.architectureFingerprint=std::string(64,'d');});
  implementationOOD("new tile outside envelope",[](auto &v){++v.values["max_tile_loop_0_iterations"];});
  implementationOOD("tail excluded",[](auto &v){v.values["source_has_tile_tail"]=1;});
  implementationOOD("mixed4 excluded",[](auto &v){v.implementations={"scalar-product-vector-add-f32:4"};});
  implementationOOD("exp not silently product",[](auto &v){v.operations["math.exp:f32"]=1;});
  auto missingEnvelope=implementationModel();missingEnvelope.getAsObject()->getObject("domain")->getObject("feature_ranges")->erase("dma_bytes");
  expect(!evaluateGenericDirectAIRCost(f,missingEnvelope,target,c,error),"implementation envelope mandatory");
  auto wrongTerms=implementationModel();(*wrongTerms.getAsObject()->getArray("terms"))[1].getAsObject()->operator[]("feature")="dma_bytes";
  expect(!evaluateGenericDirectAIRCost(f,wrongTerms,target,c,error),"implementation terms frozen not double charged");
  auto badRegions=plain;badRegions.communication.executionRegions[0].submissionCount=0;
  expect(!collectGenericCostFeatures(a,badRegions,alteredFeatures,error),"nonpositive public submission rejected");
  auto twoRegions=plain;twoRegions.communication.executionRegions.push_back(twoRegions.communication.executionRegions[0]);
  expect(collectGenericCostFeatures(a,twoRegions,alteredFeatures,error) && alteredFeatures.values["command_count"]==2 && alteredFeatures.values["execution_region_count"]==2,"public submissions counted once per region");
  std::string source;llvm::raw_string_ostream sourceStream(source);module->print(sourceStream);sourceStream.flush();
  auto replaceAll=[](std::string text,const std::string &from,const std::string &to){size_t at=0;while((at=text.find(from,at))!=std::string::npos){text.replace(at,from.size(),to);at+=to.size();}return text;};
  // These checks isolate family derivation from target legality: no transformed
  // program is executed or represented as a successfully checked mapping.
  auto familyFrom=[&](const std::string &text){
    auto m=mlir::parseSourceString<mlir::ModuleOp>(text,&context);GenericAnalysis analysis;GenericCostFeatures facts;
    if(!m || !analyzeGenericProgram(*m,analysis,error) || !collectGenericCostFeatures(analysis,plain,facts,error))return std::string();
    return facts.regionFamilyFingerprint;
  };
  auto bigger=replaceAll(replaceAll(source,"16xf32","24xf32"),"to 16","to 24");
  expect(familyFrom(bigger)==f.regionFamilyFingerprint,"family erases static extents only");
  auto renamed=replaceAll(source,"@add","@renamed");
  expect(familyFrom(renamed)==f.regionFamilyFingerprint,"family erases task and function names");
  auto clone=mlir::cast<mlir::ModuleOp>(module->clone());GenericAnalysis cloneAnalysis;
  expect(analyzeGenericProgram(clone,cloneAnalysis,error),"family mutation source analysis");
  mlir::OpBuilder builder(&context);builder.setInsertionPointToStart(&cloneAnalysis.tasks[0].operation->getRegion(0).front());
  auto constant=builder.create<mlir::arith::ConstantFloatOp>(clone.getLoc(),llvm::APFloat(1.0f),builder.getF32Type());
  GenericCostFeatures oneConstant,twoConstant;
  collectGenericCostFeatures(cloneAnalysis,plain,oneConstant,error);
  constant->setAttr("value",builder.getF32FloatAttr(2.0));
  collectGenericCostFeatures(cloneAnalysis,plain,twoConstant,error);
  expect(oneConstant.regionFamilyFingerprint!=twoConstant.regionFamilyFingerprint,"family preserves arithmetic constants");
  auto loop=mlir::cast<mlir::affine::AffineForOp>(cloneAnalysis.tasks[0].loops[0].operation);
  loop.setStep(2);GenericCostFeatures changedStep;collectGenericCostFeatures(cloneAnalysis,plain,changedStep,error);
  expect(changedStep.regionFamilyFingerprint!=twoConstant.regionFamilyFingerprint,"family preserves reduction or spatial step");
  loop.setStep(1);
  auto load=mlir::cast<mlir::affine::AffineLoadOp>(cloneAnalysis.tasks[0].accesses[0].operation);
  load.getProperties().map=mlir::AffineMapAttr::get(mlir::AffineMap::get(1,0,{15-builder.getAffineDimExpr(0)},&context));
  GenericCostFeatures changedMap;collectGenericCostFeatures(cloneAnalysis,plain,changedMap,error);
  expect(changedMap.regionFamilyFingerprint!=twoConstant.regionFamilyFingerprint,"family preserves affine access maps");
  auto root=cloneAnalysis.storages[0].root;auto oldType=root.getType();
  root.setType(mlir::MemRefType::get({16},builder.getF32Type(),mlir::StridedLayoutAttr::get(&context,0,{2})));
  GenericCostFeatures changedLayout;collectGenericCostFeatures(cloneAnalysis,plain,changedLayout,error);
  expect(changedLayout.regionFamilyFingerprint!=changedMap.regionFamilyFingerprint,"family preserves memory layout");root.setType(oldType);
  mlir::func::FuncOp(cloneAnalysis.function)->setAttr("orbit.noalias",builder.getStringAttr("changed-contract"));
  GenericCostFeatures changedAlias;collectGenericCostFeatures(cloneAnalysis,plain,changedAlias,error);
  expect(changedAlias.regionFamilyFingerprint!=changedMap.regionFamilyFingerprint,"family preserves alias contract");
  clone.erase();
  llvm::outs()<<llvm::json::Value(llvm::json::Object{{"status",ok?"passed":"failed"},{"checks",int64_t(checks)},{"calibration","synthetic contracts only"}})<<"\n";
  return ok?0:1;
}

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraJointScheduling
  AnalyticalTaskCandidateCommon.cpp
  AnalyticalTaskCandidateManifest.cpp
  AnalyticalTaskCostCatalog.cpp
  AnalyzeRecResMiiPass.cpp
  AnalyticalMLPInference.cpp
  MapperFeatureExtractor.cpp
  BindExactMemrefShapePass.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  EnumerateExactTaskSchedulesPass.cpp
  ExactScheduleSpace.cpp
  CommunicationExactScheduleSpace.cpp
  FixedScheduleVerifier.cpp
  ProductionFixedScheduleCommunication.cpp
  EnumerateJointSemanticRewritePass.cpp
  EnumerateJointGraphClosurePass.cpp
  JointNeighborhoodSearchPass.cpp
  JointNeighborhoodActions.cpp
  PruneJointGraphCandidatesPass.cpp
  EnumerateApplicationRewriteActionsPass.cpp
  MaterializeJointSemanticRewritePass.cpp
  JointReductionRewritePasses.cpp
  MaterializeNeuraJointRewritePass.cpp
  MaterializeNeuraKReductionPass.cpp
  BindStaticScalarArgumentPass.cpp
  ImportInput0CallerNoAliasPass.cpp
  LowerJointTaskflowForHostPass.cpp
  ProductionSramNativeUpperBound.cpp
  ProductionSramNativeCapacityGate.cpp
  ExtractJointTaskGraphFactsPass.cpp
  MaterializeAnalyticalTaskCandidatePass.cpp
  MaterializeJointTaskReplicasPass.cpp
  EnumerateReplicaActionsPass.cpp
  MapJointSchedulingTasksPass.cpp
  ScoreAnalyticalTaskCandidatesPass.cpp
  RebindAnalyticalTaskCostCatalogPass.cpp
  InheritReplicaAnalyticalTaskCostCatalogPass.cpp
  PredictAnalyticalTaskCostCatalogPass.cpp
  GlobalStageRankerPass.cpp
  ScoreExactJointTaskCandidatesPass.cpp
  SpatialTaskCandidateSpace.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraOrchestrationUtil
  MLIRArithDialect
  MLIRBytecodeWriter
  MLIRFuncDialect
  MLIRIR
  MLIRNeura
  MLIRNeuraConversion
  MLIRNeuraTransforms
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTaskflow
  ${dialect_libs}
  LLVMSupport
)

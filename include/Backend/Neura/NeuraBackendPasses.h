// NeuraBackendPasses.h - Passes owned by the Neura backend

#ifndef AMOEBA_BACKEND_NEURA_PASSES_H
#define AMOEBA_BACKEND_NEURA_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "Backend/Neura/Orchestration/JointScheduling/ProveStaticActiveTransferShapesPass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {

// Registers the existing affine -> Taskflow -> Neura convenience pipeline.
void registerTaskflowConversionPassPipeline();

// Passes defined in NeuraBackendPasses.td.
#define GEN_PASS_DECL
#include "Backend/Neura/NeuraBackendPasses.h.inc"

std::unique_ptr<Pass> createConvertTaskflowToNeuraPass();
std::unique_ptr<Pass> createConstructHyperblockFromTaskPass();
std::unique_ptr<Pass> createBindSourceIterationDomainPass();
std::unique_ptr<Pass> createVerifySourceIterationDomainPartitionsPass();
std::unique_ptr<Pass> createClassifyTaskAndCounterPass();
std::unique_ptr<Pass> createOrchestrateTasksOnAcceleratorsPass();
std::unique_ptr<Pass> createAnalyzeRecResMiiPass();
std::unique_ptr<Pass> createBindExactMemrefShapePass();
std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass();
std::unique_ptr<Pass> createEnumerateExactTaskSchedulesPass();
std::unique_ptr<Pass> createScoreAnalyticalTaskCandidatesPass();
std::unique_ptr<Pass> createScoreExactJointTaskCandidatesPass();
std::unique_ptr<Pass> createSearchJointNeighborhoodPass();
std::unique_ptr<Pass> createReplayJointNeighborhoodActionsPass();
std::unique_ptr<Pass> createPrepareJointDiagnosticCandidatePass();
std::unique_ptr<Pass> createMaterializeAnalyticalTaskCandidatePass();
std::unique_ptr<Pass> createMaterializeJointTaskReplicasPass();
std::unique_ptr<Pass> createEnumerateReplicaActionsPass();
std::unique_ptr<Pass> createMapJointSchedulingTasksPass();
std::unique_ptr<Pass> createRetimingOriginalAmoebaFixedDecisionsPass();
std::unique_ptr<Pass> createOriginalAmoebaProfileCostAdapterPass();
std::unique_ptr<Pass> createExtractJointTaskGraphFactsPass();
std::unique_ptr<Pass> createEnumerateJointSemanticRewritePass();
std::unique_ptr<Pass> createEnumerateJointGraphClosurePass();
std::unique_ptr<Pass> createPruneJointGraphCandidatesPass();
std::unique_ptr<Pass> createEnumerateApplicationRewriteActionsPass();
std::unique_ptr<Pass> createMaterializeJointSemanticRewritePass();
std::unique_ptr<Pass> createTileOutputMPass();
std::unique_ptr<Pass> createTileOutputNPass();
std::unique_ptr<Pass> createTileReductionKSequentialPass();
std::unique_ptr<Pass> createTileReductionKParallelLinearPass();
std::unique_ptr<Pass> createTileReductionKParallelTreePass();
std::unique_ptr<Pass> createFuseProducerConsumerPass();
std::unique_ptr<Pass> createFuseTileLocalPairPass();
std::unique_ptr<Pass> createFuseReductionConsumerPass();
std::unique_ptr<Pass> createMaterializeNumericReductionPass();
std::unique_ptr<Pass> createMaterializeCompletionJoinPass();
std::unique_ptr<Pass> createMaterializeNeuraJointRewritePass();
std::unique_ptr<Pass> createMaterializeNeuraKReductionPass();
std::unique_ptr<Pass> createBindStaticScalarArgumentPass();
std::unique_ptr<Pass> createImportInput0CallerNoAliasPass();
std::unique_ptr<Pass> createRebindAnalyticalTaskCostCatalogPass();
std::unique_ptr<Pass> createInheritReplicaAnalyticalTaskCostCatalogPass();
std::unique_ptr<Pass> createPredictAnalyticalTaskCostCatalogPass();
// Recompute formal max-four analytical bounds from the current C++ mapper
// bodies and validate every source-bound cost-catalogue row. This is used
// when an imported catalogue advertises unsupported model-domain shapes.
bool verifyCurrentModelDomainCostCatalog(mlir::ModuleOp module,
                                         llvm::StringRef function,
                                         llvm::StringRef catalogPath,
                                         std::string &error);
std::unique_ptr<Pass> createGlobalStageRankerPass();
std::unique_ptr<Pass> createLowerJointTaskflowToHostSCFPass();
std::unique_ptr<Pass> createFuseTaskPass();
std::unique_ptr<Pass> createFissionTaskPass();
std::unique_ptr<Pass> createVerifyTaskflowFissionSourceReplayPass();
std::unique_ptr<Pass> createFissionRayCarriedReductionPass();
std::unique_ptr<Pass> createTileTaskPass();
std::unique_ptr<Pass> createResourceAwareTaskOptimizationPass();

#define GEN_PASS_REGISTRATION
#include "Backend/Neura/NeuraBackendPasses.h.inc"

} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_PASSES_H

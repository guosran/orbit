add_mlir_library(MLIRAmoebaNeuraOrchestrationUtil
    orchestration_utils.cpp
    JointScheduling/InterTaskNetwork.cpp
    JointScheduling/TaskCommunicationModel.cpp
    JointScheduling/ProgramGraphCommunication.cpp
    JointScheduling/TaskEdgeContract.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    MLIRTaskflowStructured
    MLIRAmoebaNeuraBackendOptions
    LLVMSupport
)

add_subdirectory(AnalyticalBasedTaskOrchestration)
add_subdirectory(JointScheduling)
add_subdirectory(RoutingCriticalPathOrchestration)

add_mlir_library(MLIRAmoebaNeuraOrchestration
    Orchestration.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRAmoebaNeuraOrchestrationUtil
    MLIRAmoebaNeuraRoutingCriticalPathOrchestration
    MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
    MLIRAmoebaNeuraJointScheduling
)

add_mlir_library(MLIRTaskflowStructured StructuredProgram.cpp Search.cpp ProgramAllSearch.cpp ProgramAllHeterogeneousCost.cpp ProgramMatrixImplementation.cpp ProgramBeamSearch.cpp ProgramGraphSearch.cpp
  MemoryArchitecture.cpp ProgramTargetBinding.cpp ProgramOperatorSource.cpp ProgramGraph.cpp ProgramResourceSchedule.cpp ProgramJointDomain.cpp ProgramJointSolver.cpp ProgramJointMILPSolver.cpp ProgramJointPlan.cpp AmoebaMapping.cpp HeterogeneousMapping.cpp HeterogeneousImplementation.cpp HeterogeneousSearch.cpp ProgramHeterogeneousSearch.cpp MechanismCost.cpp ProgramTiming.cpp ProgramInvocationCost.cpp ProgramRegionCompositeCost.cpp ProgramInvocationEvaluation.cpp ProgramMechanismCost.cpp GenericMechanismTiming.cpp GenericProgramTiming.cpp GenericNativeCompositeTiming.cpp GenericProgramContrast.cpp SyncState.cpp SyncBoundaryMapping.cpp SyncCost.cpp DeviceTransferCost.cpp DeviceTransferAudit.cpp StaticAnalysis.cpp StaticContract.cpp StaticCommunication.cpp
  StaticMapper.cpp StaticChecker.cpp StaticEvaluation.cpp StaticRegion.cpp
  StaticRegionChecker.cpp GenericAnalysis.cpp GenericReuse.cpp GenericConstraintSearch.cpp GenericMapping.cpp GenericCommunication.cpp ProgramDataOrganization.cpp ProgramPartition.cpp NativeTransforms.cpp PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRTaskflow LLVMSupport MLIRAffineDialect MLIRArithDialect
  MLIRSCFDialect MLIRSideEffectInterfaces MLIRMathDialect MLIRMemRefDialect MLIRFuncDialect MLIRParser MLIRIR MLIRSupport)
# Optional solver library remains independent of MLIR and available as a stub.
add_library(ORBITConstraintSearch STATIC ConstraintSearch.cpp NPUConstraintSchedule.cpp ProgramSearch.cpp)
target_include_directories(ORBITConstraintSearch PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(ORBIT_ENABLE_ORTOOLS)
  target_link_libraries(ORBITConstraintSearch PUBLIC ortools::ortools)
  target_compile_definitions(ORBITConstraintSearch PRIVATE ORBIT_HAS_ORTOOLS=1)
endif()
if(ORBIT_ENABLE_SCIP)
  target_link_libraries(MLIRTaskflowStructured PUBLIC SCIP::libscip)
  target_link_libraries(obj.MLIRTaskflowStructured PRIVATE SCIP::libscip)
  set_source_files_properties(ProgramJointMILPSolver.cpp PROPERTIES
    COMPILE_DEFINITIONS ORBIT_HAS_SCIP=1 COMPILE_OPTIONS "-fexceptions")
endif()
target_link_libraries(MLIRTaskflowStructured PUBLIC ORBITConstraintSearch)

# These translation units contain validation exceptions within their public
# bool/error boundary; no exception crosses into MLIR/LLVM code.
set_source_files_properties(StaticContract.cpp StaticCommunication.cpp StaticChecker.cpp StaticRegion.cpp StaticRegionChecker.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")

if(ORBIT_ENABLE_ORTOOLS)
  target_link_libraries(obj.MLIRTaskflowStructured PRIVATE ortools::ortools)
  set_source_files_properties(ProgramJointSolver.cpp PROPERTIES COMPILE_DEFINITIONS ORBIT_HAS_ORTOOLS=1 COMPILE_OPTIONS "-fexceptions")
endif()

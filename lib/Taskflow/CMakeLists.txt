add_mlir_library(MLIRTaskflowStructured StructuredProgram.cpp Search.cpp
  MemoryArchitecture.cpp ProgramTargetBinding.cpp ProgramGraph.cpp AmoebaMapping.cpp HeterogeneousMapping.cpp HeterogeneousImplementation.cpp HeterogeneousSearch.cpp MechanismCost.cpp ProgramTiming.cpp ProgramMechanismCost.cpp GenericMechanismTiming.cpp GenericProgramTiming.cpp GenericProgramContrast.cpp SyncState.cpp SyncBoundaryMapping.cpp SyncCost.cpp DeviceTransferCost.cpp DeviceTransferAudit.cpp StaticAnalysis.cpp StaticContract.cpp StaticCommunication.cpp
  StaticMapper.cpp StaticChecker.cpp StaticEvaluation.cpp StaticRegion.cpp
  StaticRegionChecker.cpp GenericAnalysis.cpp GenericMapping.cpp GenericCommunication.cpp ProgramDataOrganization.cpp ProgramPartition.cpp NativeTransforms.cpp PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRTaskflow LLVMSupport MLIRAffineDialect MLIRArithDialect
  MLIRSCFDialect MLIRSideEffectInterfaces MLIRMathDialect MLIRMemRefDialect MLIRFuncDialect MLIRParser MLIRIR MLIRSupport)
# These translation units contain validation exceptions within their public
# bool/error boundary; no exception crosses into MLIR/LLVM code.
set_source_files_properties(StaticContract.cpp StaticCommunication.cpp StaticChecker.cpp StaticRegion.cpp StaticRegionChecker.cpp
  PROPERTIES COMPILE_OPTIONS "-fexceptions")

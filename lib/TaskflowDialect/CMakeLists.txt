# Semantic dialect and verifier are independent of every backend, even in a
# full legacy build. Legacy pipeline registration is a separate target below.
add_mlir_dialect_library(MLIRTaskflow
  Taskflow.cpp TaskflowOps.cpp ../Taskflow/OutputCoordinateProof.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS MLIRTaskflowIncGen MLIRTaskflowTransformsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSupport MLIRInferTypeOpInterface
                   MLIRArithDialect MLIRMemRefDialect)

if(ORBIT_ENABLE_NEURA)
  add_mlir_library(MLIRTaskflowLegacyPipelines
    TaskflowPasses.cpp
    PARTIAL_SOURCES_INTENDED
    DEPENDS MLIRAmoebaConversionIncGen MLIRTaskflowTransformsIncGen
    LINK_LIBS PUBLIC MLIRTaskflow MLIRAmoebaConversion MLIRPass)
endif()
add_subdirectory(Transforms)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaTaskflowFission
  FissionTaskPass.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTaskflow
  MLIRIR
  MLIRPass
  MLIRSupport
  LLVMSupport
)

add_mlir_library(MLIRAmoebaNeuraOptimizations
  FuseTaskPass.cpp
  FissionRayCarriedReductionPass.cpp
  ResourceAwareTaskOptimizationPass.cpp
  TileTaskPass.cpp

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaTaskflowFission
  MLIRAmoebaNeuraConversion
  MLIRAmoebaNeuraTransforms
  MLIRTaskflow
  MLIRNeura
  MLIRNeuraTransforms
  MLIRNeuraConversion
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  ${dialect_libs}
  LLVMSupport
)

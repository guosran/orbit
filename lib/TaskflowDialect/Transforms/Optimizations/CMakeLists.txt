add_mlir_library(MLIRTaskflowTransforms
  AffineLoopTreeSerializationPass.cpp
  AffineLoopPerfectionPass.cpp
  MemoryAccessStreamingFusion.cpp

  DEPENDS
  MLIRTaskflowTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  LLVMSupport
)

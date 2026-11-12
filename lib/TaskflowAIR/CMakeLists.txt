add_mlir_library(MLIRTaskflowAIR AIRBackend.cpp ProgramGraphAIR.cpp ProgramGraphAIRSelection.cpp ProgramGraphAIRLoop.cpp ProgramGraphAIE.cpp ProgramMatrixAIR.cpp ProgramMatrixRuntime.cpp AIRImplementation.cpp AIEImplementation.cpp NPUIndependentShards.cpp NPUShardsAIE.cpp ProgramJointAIR.cpp ProgramJointAIE.cpp
  LINK_LIBS PUBLIC MLIRTaskflowStructured MLIRTaskflow MLIRArithDialect
  MLIRMathTransforms MLIRTransforms MLIRMemRefDialect MLIRFuncDialect MLIRAffineToStandard MLIRAffineTransforms MLIRPass MLIRIR MLIRSupport)
# The fixed LLVM/MLIR installation is built without C++ RTTI.
target_compile_options(obj.MLIRTaskflowAIR PRIVATE -fno-rtti)

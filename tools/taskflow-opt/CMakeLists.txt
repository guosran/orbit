add_executable(taskflow-opt taskflow-opt.cpp)
target_link_libraries(taskflow-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow
  MLIRAffineDialect MLIRArithDialect MLIRMathDialect MLIRMemRefDialect MLIRFuncDialect
  MLIRParser MLIRIR MLIRSupport)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(check-taskflow-core
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/TaskflowCore/check.py
          --tool $<TARGET_FILE:taskflow-opt>
  DEPENDS taskflow-opt
  COMMENT "Checking independent Taskflow semantics and fail-closed verification")

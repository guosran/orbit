add_executable(taskflow-opt taskflow-opt.cpp)
target_link_libraries(taskflow-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow
  MLIRAffineDialect MLIRArithDialect MLIRMathDialect MLIRMemRefDialect MLIRFuncDialect
  MLIRParser MLIRIR MLIRSupport MLIRSCFDialect)
if(ORBIT_ENABLE_AIR)
  target_link_libraries(taskflow-opt PRIVATE MLIRTaskflowAIR)
  target_compile_definitions(taskflow-opt PRIVATE ORBIT_ENABLE_AIR=1)
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(check-taskflow-core
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/TaskflowCore/check.py
          --tool $<TARGET_FILE:taskflow-opt>
  DEPENDS taskflow-opt
  COMMENT "Checking independent Taskflow semantics and fail-closed verification")

add_executable(taskflow-generic-mapping-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/mapping.cpp)
target_link_libraries(taskflow-generic-mapping-test PRIVATE MLIRTaskflowStructured)
add_executable(taskflow-tiled-mapping-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/tiled-mapping.cpp)
target_link_libraries(taskflow-tiled-mapping-test PRIVATE MLIRTaskflowStructured)
add_executable(taskflow-program-mapping-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-mapping.cpp)
target_link_libraries(taskflow-program-mapping-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-program-partition-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-partition.cpp)
target_link_libraries(taskflow-program-partition-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-data-organization-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/data-organization.cpp)
target_link_libraries(taskflow-data-organization-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-direct-air-cost-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/direct-air-cost.cpp)
target_link_libraries(taskflow-direct-air-cost-test PRIVATE MLIRTaskflowStructured)

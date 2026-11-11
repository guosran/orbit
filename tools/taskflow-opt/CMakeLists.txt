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

add_executable(taskflow-program-graph-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-graph.cpp)
target_link_libraries(taskflow-program-graph-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-amoeba-program-graph-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/amoeba-program-graph.cpp)
target_link_libraries(taskflow-amoeba-program-graph-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-heterogeneous-mapping-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/heterogeneous-mapping.cpp)
target_link_libraries(taskflow-heterogeneous-mapping-test PRIVATE MLIRTaskflowStructured MLIRParser)

add_executable(taskflow-mechanism-cost-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/mechanism-cost.cpp)
target_link_libraries(taskflow-mechanism-cost-test PRIVATE MLIRTaskflowStructured MLIRParser)

add_executable(taskflow-sync-state-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/sync-state.cpp)
target_link_libraries(taskflow-sync-state-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-sync-boundary-mapping-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/sync-boundary-mapping.cpp)
target_link_libraries(taskflow-sync-boundary-mapping-test PRIVATE MLIRTaskflowStructured MLIRParser)

add_executable(taskflow-sync-cost-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/sync-cost.cpp)
target_link_libraries(taskflow-sync-cost-test PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-sync-opt taskflow-sync-opt.cpp)
target_link_libraries(taskflow-sync-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser)

add_executable(taskflow-sync-replay taskflow-sync-replay.cpp)
target_link_libraries(taskflow-sync-replay PRIVATE MLIRTaskflowStructured)

add_executable(taskflow-device-transfer-cost-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/device-transfer-cost.cpp)
target_link_libraries(taskflow-device-transfer-cost-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser)

add_executable(taskflow-transfer-opt taskflow-transfer-opt.cpp)
target_link_libraries(taskflow-transfer-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)

add_executable(taskflow-program-timing-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-timing.cpp)
target_link_libraries(taskflow-program-timing-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser)

add_executable(taskflow-contrast-opt taskflow-contrast-opt.cpp)
target_link_libraries(taskflow-contrast-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)

add_executable(taskflow-program-contrast-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-contrast.cpp)
target_link_libraries(taskflow-program-contrast-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)

add_executable(taskflow-instance-grouping-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/instance-grouping.cpp)
target_link_libraries(taskflow-instance-grouping-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)

add_executable(taskflow-program-mechanism-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-mechanism.cpp)
target_link_libraries(taskflow-program-mechanism-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)

add_executable(taskflow-target-binding-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/target-binding.cpp)
target_link_libraries(taskflow-target-binding-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)
add_executable(taskflow-subgraph-grouping-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/subgraph-grouping.cpp)
target_link_libraries(taskflow-subgraph-grouping-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)
add_executable(taskflow-reuse-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/reuse.cpp)
target_link_libraries(taskflow-reuse-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)
add_executable(taskflow-shared-input-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/shared-input.cpp)
target_link_libraries(taskflow-shared-input-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)
add_executable(taskflow-reuse-constraints-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/reuse-constraints.cpp)
target_link_libraries(taskflow-reuse-constraints-test PRIVATE MLIRTaskflowStructured MLIRTaskflow MLIRParser MLIRSCFDialect)
add_executable(taskflow-cpsat-opt taskflow-cpsat-opt.cpp)
target_compile_options(taskflow-cpsat-opt PRIVATE -fexceptions)
target_link_libraries(taskflow-cpsat-opt PRIVATE ORBITConstraintSearch LLVMSupport)

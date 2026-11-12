add_executable(taskflow-opt taskflow-opt.cpp)
target_link_libraries(taskflow-opt PRIVATE MLIRTaskflowStructured MLIRTaskflow
  MLIRAffineDialect MLIRArithDialect MLIRMathDialect MLIRMemRefDialect MLIRFuncDialect
  MLIRParser MLIRIR MLIRSupport MLIRSCFDialect)
if(ORBIT_ENABLE_AIR)
  target_link_libraries(taskflow-opt PRIVATE MLIRTaskflowAIR)
  target_compile_definitions(taskflow-opt PRIVATE ORBIT_ENABLE_AIR=1)
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(taskflow-json-ownership-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/json-ownership.cpp)
target_link_libraries(taskflow-json-ownership-test PRIVATE LLVMSupport)
add_custom_target(check-taskflow-json-ownership
  COMMAND $<TARGET_FILE:taskflow-json-ownership-test>
  DEPENDS taskflow-json-ownership-test)
add_executable(taskflow-program-all-heterogeneous-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-all-heterogeneous.cpp)
target_link_libraries(taskflow-program-all-heterogeneous-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-program-all-heterogeneous
  COMMAND $<TARGET_FILE:taskflow-program-all-heterogeneous-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-all-heterogeneous-test)
add_executable(taskflow-program-all-inactive-bindings-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-all-inactive-bindings.cpp)
target_link_libraries(taskflow-program-all-inactive-bindings-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-program-all-inactive-bindings
  COMMAND $<TARGET_FILE:taskflow-program-all-inactive-bindings-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-all-inactive-bindings-test)
add_executable(taskflow-program-all-heterogeneous-cost-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-all-heterogeneous-cost.cpp)
target_link_libraries(taskflow-program-all-heterogeneous-cost-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-program-all-heterogeneous-cost
  COMMAND $<TARGET_FILE:taskflow-program-all-heterogeneous-cost-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-all-heterogeneous-cost-test)
add_executable(taskflow-heterogeneous-implementation-projection-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/heterogeneous-implementation-projection.cpp)
target_link_libraries(taskflow-heterogeneous-implementation-projection-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-heterogeneous-implementation-projection
  COMMAND $<TARGET_FILE:taskflow-heterogeneous-implementation-projection-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/heterogeneous-matrix-relu.mlir
  DEPENDS taskflow-heterogeneous-implementation-projection-test)
set(ORBIT_NPU_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python with NumPy and ml_dtypes for typed execution contract tests")
add_custom_target(check-taskflow-heterogeneous-runtime
  COMMAND ${ORBIT_NPU_PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/execute_heterogeneous_v32.py)
set(ORBIT_CALIBRATION_PYTHON_EXECUTABLE "${ORBIT_NPU_PYTHON_EXECUTABLE}" CACHE FILEPATH
  "Python with NumPy and SciPy for offline calibration contract tests")
execute_process(COMMAND "${ORBIT_CALIBRATION_PYTHON_EXECUTABLE}" -c "import numpy; import scipy"
  RESULT_VARIABLE ORBIT_CALIBRATION_IMPORT_RESULT
  OUTPUT_QUIET ERROR_QUIET)
if(ORBIT_CALIBRATION_IMPORT_RESULT EQUAL 0)
  add_custom_target(check-taskflow-heterogeneous-calibration
    COMMAND "${ORBIT_CALIBRATION_PYTHON_EXECUTABLE}"
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/calibrate_heterogeneous_v32.py)
else()
  message(STATUS "Calibration contracts SKIP: configured Python lacks NumPy/SciPy")
  add_custom_target(check-taskflow-heterogeneous-calibration
    COMMAND ${CMAKE_COMMAND} -E echo
      "SKIP: calibration contracts require NumPy/SciPy in ORBIT_CALIBRATION_PYTHON_EXECUTABLE")
endif()
if(ORBIT_ENABLE_AIR)
  add_executable(taskflow-program-matrix-test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-matrix.cpp)
  target_link_libraries(taskflow-program-matrix-test PRIVATE
    MLIRTaskflowAIR MLIRTaskflowStructured MLIRTaskflow MLIRParser)
endif()
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

if(ORBIT_ENABLE_ORTOOLS)
  set(ORBIT_JOINT_TEST_MODE --require-solver)
else()
  set(ORBIT_JOINT_TEST_MODE --unavailable)
endif()
add_custom_target(check-taskflow-program-joint COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_joint_v23.py $<TARGET_FILE:taskflow-opt> ${ORBIT_JOINT_TEST_MODE} DEPENDS taskflow-opt)

add_executable(taskflow-program-joint-milp-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-joint-milp.cpp)
target_link_libraries(taskflow-program-joint-milp-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_executable(taskflow-program-joint-domain-audit-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-joint-domain-audit.cpp)
target_link_libraries(taskflow-program-joint-domain-audit-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
set(ORBIT_JOINT_COMPARE_MODES)
if(NOT ORBIT_ENABLE_SCIP)
  list(APPEND ORBIT_JOINT_COMPARE_MODES --scip-off)
endif()
if(NOT ORBIT_ENABLE_ORTOOLS)
  list(APPEND ORBIT_JOINT_COMPARE_MODES --cp-sat-off)
endif()
add_custom_target(check-taskflow-program-joint-milp
  COMMAND $<TARGET_FILE:taskflow-program-joint-milp-test>
  COMMAND $<TARGET_FILE:taskflow-program-joint-domain-audit-test>
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_joint_milp_v31.py
    $<TARGET_FILE:taskflow-opt> ${ORBIT_JOINT_COMPARE_MODES}
  DEPENDS taskflow-opt taskflow-program-joint-milp-test
    taskflow-program-joint-domain-audit-test)

add_executable(taskflow-program-search-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-search.cpp)
target_link_libraries(taskflow-program-search-test PRIVATE ORBITConstraintSearch LLVMSupport)
add_executable(taskflow-program-beam-search-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-beam-search.cpp)
target_link_libraries(taskflow-program-beam-search-test PRIVATE MLIRTaskflowStructured LLVMSupport)
add_executable(taskflow-program-routing-search-test EXCLUDE_FROM_ALL ../../test/TaskflowGeneric/program-routing-search.cpp)
target_link_libraries(taskflow-program-routing-search-test PRIVATE MLIRTaskflowStructured MLIRParser)

add_executable(taskflow-program-graph-search-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-graph-search.cpp)
target_link_libraries(taskflow-program-graph-search-test PRIVATE MLIRTaskflowStructured MLIRParser)

if(ORBIT_ENABLE_ORTOOLS)
  set(ORBIT_PUBLIC_SEARCH_TEST_MODE)
else()
  set(ORBIT_PUBLIC_SEARCH_TEST_MODE --solver-off)
endif()
set(ORBIT_PUBLIC_SEARCH_BOUNDARY_COMMAND)
if(ORBIT_ENABLE_AIR)
  set(ORBIT_PUBLIC_SEARCH_BOUNDARY_COMMAND
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_search_boundary_v32.py
      --tool $<TARGET_FILE:taskflow-opt>)
endif()
add_custom_target(check-taskflow-public-search
  COMMAND $<TARGET_FILE:taskflow-program-search-test>
  COMMAND $<TARGET_FILE:taskflow-program-beam-search-test>
  COMMAND $<TARGET_FILE:taskflow-program-routing-search-test>
  COMMAND $<TARGET_FILE:taskflow-program-graph-search-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/gemm-scale.mlir
  COMMAND $<TARGET_FILE:taskflow-program-mechanism-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/broadcast.mlir
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_all_search_v25.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_search_identity_v26.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_replay_v27.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  ${ORBIT_PUBLIC_SEARCH_BOUNDARY_COMMAND}
  DEPENDS taskflow-opt taskflow-program-search-test taskflow-program-beam-search-test
    taskflow-program-routing-search-test taskflow-program-graph-search-test
    taskflow-program-mechanism-test)

if(ORBIT_ENABLE_AIR)
  add_custom_target(check-taskflow-public-search-boundary
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_search_boundary_v32.py
      --tool $<TARGET_FILE:taskflow-opt>
    DEPENDS taskflow-opt)
endif()

if(ORBIT_ENABLE_AIR)
  add_custom_target(check-taskflow-public-graph-air
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/public_graph_air_v27.py
      --tool $<TARGET_FILE:taskflow-opt>
    DEPENDS taskflow-opt)
endif()

add_executable(taskflow-program-heterogeneous-search-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-heterogeneous-search.cpp)
target_link_libraries(taskflow-program-heterogeneous-search-test PRIVATE MLIRTaskflowStructured MLIRParser)
add_custom_target(check-taskflow-program-heterogeneous-search
  COMMAND $<TARGET_FILE:taskflow-program-heterogeneous-search-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-heterogeneous-search-test)

add_executable(taskflow-program-invocation-cost-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-invocation-cost.cpp)
target_link_libraries(taskflow-program-invocation-cost-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-program-invocation-cost
  COMMAND $<TARGET_FILE:taskflow-program-invocation-cost-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-invocation-cost-test)

add_custom_target(check-taskflow-program-invocation-cli
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_invocation_v28.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  DEPENDS taskflow-opt)

add_executable(taskflow-program-region-composite-cost-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-region-composite-cost.cpp)
target_link_libraries(taskflow-program-region-composite-cost-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-program-region-composite-cost
  COMMAND $<TARGET_FILE:taskflow-program-region-composite-cost-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-program-region-composite-cost-test)

add_executable(taskflow-generic-native-composite-timing-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/generic-native-composite-timing.cpp)
target_link_libraries(taskflow-generic-native-composite-timing-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-generic-native-composite-timing
  COMMAND $<TARGET_FILE:taskflow-generic-native-composite-timing-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-generic-native-composite-timing-test)
add_custom_target(check-taskflow-native-composite-cli
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_native_composite_v30.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  DEPENDS taskflow-opt)

add_custom_target(check-taskflow-native-errors
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_native_errors_v31.py
    --tool $<TARGET_FILE:taskflow-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  DEPENDS taskflow-opt)

add_executable(taskflow-task-implementations-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/task-implementations.cpp)
target_link_libraries(taskflow-task-implementations-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
if(ORBIT_ENABLE_AIR)
  target_link_libraries(taskflow-task-implementations-test PRIVATE MLIRTaskflowAIR)
  target_compile_definitions(taskflow-task-implementations-test PRIVATE ORBIT_ENABLE_AIR=1)
endif()
add_custom_target(check-taskflow-task-implementations
  COMMAND $<TARGET_FILE:taskflow-task-implementations-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/task-implementations.mlir
  DEPENDS taskflow-task-implementations-test)

if(ORBIT_ENABLE_AIR)
  add_custom_target(check-taskflow-heterogeneous-cli-tiles
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/heterogeneous_cli_tiles_v32.py
      --tool $<TARGET_FILE:taskflow-opt>
    DEPENDS taskflow-opt)
endif()

add_custom_target(check-taskflow-heterogeneous-calibration-index
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/heterogeneous_calibration_index_v32.py)
if(ORBIT_CALIBRATION_IMPORT_RESULT EQUAL 0)
  add_custom_target(check-taskflow-heterogeneous-feature-design
    COMMAND "${ORBIT_CALIBRATION_PYTHON_EXECUTABLE}"
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/heterogeneous_feature_design_v32.py)
else()
  add_custom_target(check-taskflow-heterogeneous-feature-design
    COMMAND ${CMAKE_COMMAND} -E echo
      "SKIP: feature design contracts require NumPy/SciPy in ORBIT_CALIBRATION_PYTHON_EXECUTABLE")
endif()

add_executable(taskflow-task-implementation-cost-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/task-implementation-cost.cpp)
target_link_libraries(taskflow-task-implementation-cost-test PRIVATE
  MLIRTaskflowStructured MLIRTaskflow MLIRParser)
add_custom_target(check-taskflow-task-implementation-cost
  COMMAND $<TARGET_FILE:taskflow-task-implementation-cost-test>
    ${PROJECT_SOURCE_DIR}/test/TaskflowAIR/fixtures/heterogeneous.mlir
  DEPENDS taskflow-task-implementation-cost-test)

# The rounding refinement is an internal IR helper. Its CPU contract does not
# require AIR libraries, vendor tools, a device or a hardware runtime.
add_executable(taskflow-bf16-rne-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-graph-air-bf16-rne.cpp)
target_link_libraries(taskflow-bf16-rne-test PRIVATE
  MLIRArithDialect MLIRFuncDialect MLIRParser MLIRTransforms)
add_custom_target(check-taskflow-bf16-rne
  COMMAND $<TARGET_FILE:taskflow-bf16-rne-test>
  DEPENDS taskflow-bf16-rne-test)

add_custom_target(check-taskflow-solver-option-ranges
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/solver_option_ranges_v32.py
    --tool $<TARGET_FILE:taskflow-cpsat-opt> ${ORBIT_PUBLIC_SEARCH_TEST_MODE}
  DEPENDS taskflow-cpsat-opt)

if(ORBIT_ENABLE_AIR)
  add_custom_target(check-taskflow-matrix-runtime-static
    COMMAND ${Python3_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program_matrix_runtime_static_v32.py
      --tool $<TARGET_FILE:taskflow-opt>
      --input ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/matrix-runtime-v32/constant-feed.generic.mlir
      --contract ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/matrix-runtime-v32/constant-feed.contract.json
      --circuit-input ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/matrix-runtime-v32/circuit-relu.generic.mlir
      --circuit-contract ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/fixtures/matrix-runtime-v32/circuit-relu.contract.json
    DEPENDS taskflow-opt)
endif()

# Generic IR compatibility is a CPU parser contract, independent of AIR/XRT.
add_executable(taskflow-runtime-compatibility-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/test/TaskflowGeneric/program-runtime-compatibility.cpp)
target_link_libraries(taskflow-runtime-compatibility-test PRIVATE
  MLIRArithDialect MLIRFuncDialect MLIRParser)
add_custom_target(check-taskflow-runtime-compatibility
  COMMAND $<TARGET_FILE:taskflow-runtime-compatibility-test>
  DEPENDS taskflow-runtime-compatibility-test)

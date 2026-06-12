cmake_minimum_required(VERSION 3.16)
project(LiQuID VERSION 0.7.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Options ───────────────────────────────────────────────────────────────────
option(LIQUID_OPENMP  "Enable OpenMP parallelism"     ON)
option(LIQUID_TESTS   "Build test suite"              ON)
option(LIQUID_EXAMPLES "Build example programs"       ON)
option(LIQUID_SANITIZE "Enable ASan/UBSan (Debug)"   OFF)

# ── Compiler flags ────────────────────────────────────────────────────────────
set(LIQUID_WARN  -Wall -Wextra -Wpedantic -Wno-unused-parameter)
set(LIQUID_REL   -O3 -DNDEBUG -march=native)
set(LIQUID_DBG   -O0 -g -DLIQUID_DEBUG)
set(LIQUID_SAN   -fsanitize=address,undefined -fno-omit-frame-pointer)

# ── OpenMP ────────────────────────────────────────────────────────────────────
if(LIQUID_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found — parallel ensemble enabled")
    else()
        message(STATUS "OpenMP not found — serial execution only")
    endif()
endif()

# ── Core library (rng.cpp is the only compiled unit) ─────────────────────────
add_library(liquid STATIC src/core/rng.cpp)
target_include_directories(liquid PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(liquid PUBLIC cxx_std_17)
target_compile_options(liquid PRIVATE
    ${LIQUID_WARN}
    $<$<CONFIG:Release>:${LIQUID_REL}>
    $<$<CONFIG:Debug>:${LIQUID_DBG}>
    $<$<BOOL:${LIQUID_SANITIZE}>:${LIQUID_SAN}>)
if(OpenMP_CXX_FOUND AND LIQUID_OPENMP)
    target_link_libraries(liquid PUBLIC OpenMP::OpenMP_CXX)
endif()
if(LIQUID_SANITIZE)
    target_link_options(liquid PUBLIC ${LIQUID_SAN})
endif()

# ── Helper: add a test executable ────────────────────────────────────────────
function(liquid_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE liquid)
    target_compile_options(${name} PRIVATE
        ${LIQUID_WARN} -Wno-unused-variable
        -O1 -g -DLIQUID_DEBUG)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(liquid_validation name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE liquid)
    target_compile_options(${name} PRIVATE
        ${LIQUID_WARN} -Wno-unused-variable
        $<$<CONFIG:Release>:${LIQUID_REL}>
        $<$<CONFIG:Debug>:${LIQUID_DBG}>)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(LIQUID_TESTS)
    enable_testing()

    # Unit tests
    liquid_test(test_rng               tests/unit/test_rng.cpp)
    liquid_test(test_linalg            tests/unit/test_linalg.cpp)
    liquid_test(test_opensystem        tests/unit/test_opensystem.cpp)
    liquid_test(test_trajectory_state  tests/unit/test_trajectory_state.cpp)
    liquid_test(test_running_statistics tests/unit/test_running_statistics.cpp)
    liquid_test(test_ensemble          tests/unit/test_ensemble.cpp)
    liquid_test(test_dopri45           tests/unit/test_dopri45.cpp)
    liquid_test(test_sparse            tests/unit/test_sparse.cpp)
    liquid_test(test_ml_policy         tests/unit/test_ml_policy.cpp)
    liquid_test(test_parameter_sweep   tests/unit/test_parameter_sweep.cpp)

    # Validation tests
    liquid_validation(val_two_level_decay    tests/validation/two_level_decay.cpp)
    liquid_validation(val_driven_two_level   tests/validation/driven_two_level.cpp)
    liquid_validation(val_ensemble_convergence tests/validation/ensemble_convergence.cpp)
    liquid_validation(val_cavity_qed         tests/validation/cavity_qed.cpp)
    liquid_validation(val_smart_stopping     tests/validation/smart_stopping.cpp)
    liquid_validation(val_ml_policy          tests/validation/ml_policy_validation.cpp)
    liquid_validation(val_parameter_sweep    tests/validation/parameter_sweep_validation.cpp)
endif()

# ── Examples ──────────────────────────────────────────────────────────────────
if(LIQUID_EXAMPLES)
    add_executable(jc_steady_state examples/jc_steady_state.cpp)
    target_link_libraries(jc_steady_state PRIVATE liquid)
    target_compile_options(jc_steady_state PRIVATE
        ${LIQUID_WARN} -Wno-unused-variable
        $<$<CONFIG:Release>:${LIQUID_REL}>)
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "LiQuID v${PROJECT_VERSION} configured")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP      : ${OpenMP_CXX_FOUND}")
message(STATUS "  Tests       : ${LIQUID_TESTS}")
message(STATUS "  Examples    : ${LIQUID_EXAMPLES}")
message(STATUS "  Sanitizers  : ${LIQUID_SANITIZE}")

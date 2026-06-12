cmake_minimum_required(VERSION 3.16)
project(LiQuID VERSION 0.1.0 LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
set(LIQUID_WARNING_FLAGS
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
)

set(LIQUID_RELEASE_FLAGS
    -O3 -DNDEBUG -march=native -ffast-math
)

set(LIQUID_DEBUG_FLAGS
    -O0 -g -DLIQUID_DEBUG
)

# ── Interface library (header-only core) ──────────────────────────────────────
add_library(liquid_headers INTERFACE)
target_include_directories(liquid_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(liquid_headers INTERFACE cxx_std_17)

# ── Main library ──────────────────────────────────────────────────────────────
add_library(liquid STATIC
    src/core/rng.cpp
)
target_link_libraries(liquid PUBLIC liquid_headers)
target_compile_options(liquid PRIVATE
    ${LIQUID_WARNING_FLAGS}
    $<$<CONFIG:Release>:${LIQUID_RELEASE_FLAGS}>
    $<$<CONFIG:Debug>:${LIQUID_DEBUG_FLAGS}>
)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()

# Unit tests
add_executable(test_unit
    tests/unit/test_rng.cpp
    tests/unit/test_linalg.cpp
    tests/unit/test_opensystem.cpp
    tests/unit/test_trajectory_state.cpp
    tests/unit/test_running_statistics.cpp
    tests/unit/test_ensemble.cpp
)
target_link_libraries(test_unit PRIVATE liquid)
target_compile_options(test_unit PRIVATE
    ${LIQUID_WARNING_FLAGS}
    -O1 -g -DLIQUID_DEBUG
)
add_test(NAME unit COMMAND test_unit)

# Validation tests
add_executable(test_validation
    tests/validation/two_level_decay.cpp
    tests/validation/driven_two_level.cpp
    tests/validation/ensemble_convergence.cpp
)
target_link_libraries(test_validation PRIVATE liquid)
target_compile_options(test_validation PRIVATE
    ${LIQUID_WARNING_FLAGS}
    -O2 -DNDEBUG
)
add_test(NAME validation COMMAND test_validation)

message(STATUS "LiQuID build configured")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")

cmake_minimum_required(VERSION 3.10)

project(tricubic_interpolation)

set(CMAKE_CXX_STANDARD 17)
option(NRICP_ENABLE_CUDA "Build CUDA backend components" OFF)

# Avoid Windows headers defining min/max macros that conflict with std::min/std::max
if (WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# Add compiler warnings for GCC and Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    add_compile_options(
        -Wall        # Enable most warning flags
        -Wextra      # Enable extra warning flags
        -Wpedantic   # Enable pedantic mode
        -Werror      # Treat warnings as errors
    )
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(nanoflann)
find_package(PDAL REQUIRED)
find_package(cxxopts 3.0.0 REQUIRED)
find_package(fmt REQUIRED)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

include_directories(${PDAL_INCLUDE_DIRS})
link_directories(${PDAL_LIBRARY_DIRS})
add_definitions(${PDAL_DEFINITIONS})

set(LIB_EIGEN Eigen3::Eigen)
set(LIB_NANOFLANN nanoflann::nanoflann)
set(LIB_FMT fmt::fmt)
set(LIB_CXXOPTS cxxopts::cxxopts)

set(BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/bin") # binaries will be copied to this directory
#message("BIN_DIR=" ${BIN_DIR})

find_package(Threads REQUIRED)

# libnonrigid_icp library
add_library(libnonrigid_icp STATIC
    src/lib/io_utils.cpp
    src/lib/io_utils.hpp
    src/lib/pt_cloud.cpp
    src/lib/pt_cloud.hpp
    src/lib/translation_grid.cpp
    src/lib/translation_grid.hpp
    src/lib/correspondences.cpp
    src/lib/correspondences.hpp
    src/lib/optimization.cpp
    src/lib/optimization.hpp)

target_link_libraries(libnonrigid_icp PUBLIC ${LIB_EIGEN} ${LIB_NANOFLANN} ${PDAL_LIBRARIES})
target_include_directories(libnonrigid_icp PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/lib)
target_include_directories(libnonrigid_icp PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(libnonrigid_icp PROPERTIES DEBUG_POSTFIX _d)

if(NRICP_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures for nricp kernels")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)

    add_library(libnonrigid_icp_cuda STATIC
        src/lib/cuda/cuda_runtime.cu
        src/lib/cuda/cuda_runtime.hpp
        src/lib/cuda/cuda_transform.cu
        src/lib/cuda/cuda_transform.hpp)
    target_link_libraries(libnonrigid_icp_cuda PUBLIC CUDA::cudart)
    target_include_directories(libnonrigid_icp_cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    target_compile_definitions(libnonrigid_icp_cuda PUBLIC NRICP_ENABLE_CUDA=1)
    set_target_properties(libnonrigid_icp_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        DEBUG_POSTFIX _d)
endif()

# nonrigid-icp executable
add_executable(nonrigid-icp src/prog/nonrigid_icp.cpp)
target_link_libraries(nonrigid-icp libnonrigid_icp ${LIB_CXXOPTS} ${LIB_FMT})
target_include_directories(nonrigid-icp PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(nonrigid-icp PROPERTIES DEBUG_POSTFIX _d)

# nonrigid-icp-transform executable
add_executable(nonrigid-icp-transform src/prog/nonrigid_icp_transform.cpp)
target_link_libraries(nonrigid-icp-transform libnonrigid_icp ${LIB_CXXOPTS} ${LIB_FMT})
target_include_directories(nonrigid-icp-transform PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(nonrigid-icp-transform PROPERTIES DEBUG_POSTFIX _d)

# Unit and smoke tests
set(NRICP_DATA_SMOKE_TRANSFORM "${CMAKE_BINARY_DIR}/results/data_fixed_moving_v50_b1_i15_w001_e2_n35000_test.nricp")
set(NRICP_DATA_SMOKE_TRANSFORM_REPEAT "${CMAKE_BINARY_DIR}/results/data_fixed_moving_v50_b1_i15_w001_e2_n35000_repeat_test.nricp")
set(NRICP_DATA_SMOKE_TRANSFORM_MOVING_XYZ "${CMAKE_BINARY_DIR}/results/data_fixed_moving_xyz_v50_b1_i15_w001_e2_n35000_test.nricp")
set(NRICP_DATA_SMOKE_TRANSFORMED "${CMAKE_BINARY_DIR}/results/moving_pc_transformed_v50_b1_i15_w001_e2_n35000_test.csv")
set(NRICP_DATA_SMOKE_TRANSFORMED_SINGLE_CHUNK "${CMAKE_BINARY_DIR}/results/moving_pc_transformed_v50_b1_i15_w001_e2_n35000_single_chunk_test.csv")
set(NRICP_ADDITIONAL_TEST_IDS 2 3 4 5 6)
set(NRICP_ADDITIONAL_TRANSFORMS)
set(NRICP_ADDITIONAL_GOLDENS)
set(NRICP_ADDITIONAL_FIT_TESTS)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/results")

foreach(additional_id IN LISTS NRICP_ADDITIONAL_TEST_IDS)
    set(additional_transform "${CMAKE_BINARY_DIR}/results/additional_fixed_moving${additional_id}_v50_b1_i15_w001_e2_n35000_test.nricp")
    list(APPEND NRICP_ADDITIONAL_TRANSFORMS "${additional_transform}")
    list(APPEND NRICP_ADDITIONAL_GOLDENS "${CMAKE_SOURCE_DIR}/additional_tests/gold${additional_id}.nricp")
    list(APPEND NRICP_ADDITIONAL_FIT_TESTS "nonrigid-icp.AdditionalFit${additional_id}")
endforeach()

string(REPLACE ";" "|" NRICP_ADDITIONAL_TRANSFORMS_DEF "${NRICP_ADDITIONAL_TRANSFORMS}")
string(REPLACE ";" "|" NRICP_ADDITIONAL_GOLDENS_DEF "${NRICP_ADDITIONAL_GOLDENS}")

add_executable(nricp-file-tests test/nricp_file_test.cpp)
target_link_libraries(nricp-file-tests PRIVATE libnonrigid_icp gtest_main)
target_include_directories(nricp-file-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(nricp-file-tests PRIVATE
    NRICP_FIXED_PC="${CMAKE_SOURCE_DIR}/data/fixed_pc.csv"
    NRICP_MOVING_PC="${CMAKE_SOURCE_DIR}/data/moving_pc.csv"
    NRICP_GOLDEN_TRANSFORM="${CMAKE_SOURCE_DIR}/test/golden/data_fixed_moving_v50_b1_i15_w001_e2_n35000.nricp"
    NRICP_SMOKE_TRANSFORM="${NRICP_DATA_SMOKE_TRANSFORM}"
    NRICP_SMOKE_TRANSFORM_REPEAT="${NRICP_DATA_SMOKE_TRANSFORM_REPEAT}"
    NRICP_SMOKE_TRANSFORM_MOVING_XYZ="${NRICP_DATA_SMOKE_TRANSFORM_MOVING_XYZ}"
    NRICP_SMOKE_TRANSFORMED="${NRICP_DATA_SMOKE_TRANSFORMED}"
    NRICP_SMOKE_TRANSFORMED_SINGLE_CHUNK="${NRICP_DATA_SMOKE_TRANSFORMED_SINGLE_CHUNK}"
    NRICP_ADDITIONAL_TRANSFORMS="${NRICP_ADDITIONAL_TRANSFORMS_DEF}"
    NRICP_ADDITIONAL_GOLDENS="${NRICP_ADDITIONAL_GOLDENS_DEF}")
set_target_properties(nricp-file-tests PROPERTIES DEBUG_POSTFIX _d)

if(NRICP_ENABLE_CUDA)
    add_executable(cuda-smoke-tests test/cuda_smoke_test.cpp)
    target_link_libraries(cuda-smoke-tests PRIVATE libnonrigid_icp_cuda libnonrigid_icp ${LIB_EIGEN} gtest_main)
    target_include_directories(cuda-smoke-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(cuda-smoke-tests PROPERTIES DEBUG_POSTFIX _d)
endif()

# Copy final executables to bin directory
file(MAKE_DIRECTORY ${BIN_DIR})
add_custom_command(TARGET nonrigid-icp           POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nonrigid-icp>           ${BIN_DIR})
add_custom_command(TARGET nonrigid-icp-transform POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nonrigid-icp-transform> ${BIN_DIR})

# Enable testing
enable_testing()

add_test(
    NAME nonrigid-icp.DataSmokeFit
    COMMAND $<TARGET_FILE:nonrigid-icp>
            -f ${CMAKE_SOURCE_DIR}/data/fixed_pc.csv
            -m ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -t ${NRICP_DATA_SMOKE_TRANSFORM}
            -v 50
            -b 1
            -i 15
            -w 0.01,0.01,0.01,0.01
            -e 2.0
            -n 35000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(
    NAME nonrigid-icp.DataSmokeFitRepeat
    COMMAND $<TARGET_FILE:nonrigid-icp>
            -f ${CMAKE_SOURCE_DIR}/data/fixed_pc.csv
            -m ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -t ${NRICP_DATA_SMOKE_TRANSFORM_REPEAT}
            -v 50
            -b 1
            -i 15
            -w 0.01,0.01,0.01,0.01
            -e 2.0
            -n 35000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(
    NAME nricp-file-tests.DataSmoke
    COMMAND $<TARGET_FILE:nricp-file-tests> --gtest_filter=-NricpAdditionalFits.*:NricpTransformOutputs.*
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_test(
    NAME nonrigid-icp.DataSmokeFitMovingXyzOnly
    COMMAND $<TARGET_FILE:nonrigid-icp>
            -f ${CMAKE_SOURCE_DIR}/data/fixed_pc.csv
            -m ${CMAKE_SOURCE_DIR}/data/moving_pc_xyz.csv
            -t ${NRICP_DATA_SMOKE_TRANSFORM_MOVING_XYZ}
            -v 50
            -b 1
            -i 15
            -w 0.01,0.01,0.01,0.01
            -e 2.0
            -n 35000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nricp-file-tests.DataSmoke PROPERTIES
    DEPENDS "nonrigid-icp.DataSmokeFit;nonrigid-icp.DataSmokeFitRepeat;nonrigid-icp.DataSmokeFitMovingXyzOnly")

foreach(additional_id IN LISTS NRICP_ADDITIONAL_TEST_IDS)
    add_test(
        NAME nonrigid-icp.AdditionalFit${additional_id}
        COMMAND $<TARGET_FILE:nonrigid-icp>
                -f ${CMAKE_SOURCE_DIR}/additional_tests/fixed.csv
                -m ${CMAKE_SOURCE_DIR}/additional_tests/moving${additional_id}.csv
                -t ${CMAKE_BINARY_DIR}/results/additional_fixed_moving${additional_id}_v50_b1_i15_w001_e2_n35000_test.nricp
                -v 50
                -b 1
                -i 15
                -w 0.01,0.01,0.01,0.01
                -e 2.0
                -n 35000
                -s
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

add_test(
    NAME nricp-file-tests.AdditionalData
    COMMAND $<TARGET_FILE:nricp-file-tests> --gtest_filter=NricpAdditionalFits.*
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nricp-file-tests.AdditionalData PROPERTIES DEPENDS "${NRICP_ADDITIONAL_FIT_TESTS}")

add_test(
    NAME nonrigid-icp.DataSmokeTransform
    COMMAND $<TARGET_FILE:nonrigid-icp-transform>
            -i ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -o ${NRICP_DATA_SMOKE_TRANSFORMED}
            -t ${NRICP_DATA_SMOKE_TRANSFORM}
            -c 5000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nonrigid-icp.DataSmokeTransform PROPERTIES DEPENDS nonrigid-icp.DataSmokeFit)

add_test(
    NAME nonrigid-icp.DataSmokeTransformSingleChunk
    COMMAND $<TARGET_FILE:nonrigid-icp-transform>
            -i ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -o ${NRICP_DATA_SMOKE_TRANSFORMED_SINGLE_CHUNK}
            -t ${NRICP_DATA_SMOKE_TRANSFORM}
            -c 1000000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nonrigid-icp.DataSmokeTransformSingleChunk PROPERTIES
    DEPENDS nonrigid-icp.DataSmokeFit)

add_test(
    NAME nricp-file-tests.TransformOutputs
    COMMAND $<TARGET_FILE:nricp-file-tests> --gtest_filter=NricpTransformOutputs.*
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nricp-file-tests.TransformOutputs PROPERTIES
    DEPENDS "nonrigid-icp.DataSmokeTransform;nonrigid-icp.DataSmokeTransformSingleChunk")

add_test(
    NAME nonrigid-icp.GpuBackendNotImplemented
    COMMAND $<TARGET_FILE:nonrigid-icp>
            -f ${CMAKE_SOURCE_DIR}/data/fixed_pc.csv
            -m ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -t ${CMAKE_BINARY_DIR}/results/gpu_backend_not_implemented.nricp
            -x gpu
            -v 50
            -b 1
            -i 15
            -w 0.01,0.01,0.01,0.01
            -e 2.0
            -n 35000
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nonrigid-icp.GpuBackendNotImplemented PROPERTIES WILL_FAIL TRUE)

add_test(
    NAME nonrigid-icp-transform.GpuBackendNotImplemented
    COMMAND $<TARGET_FILE:nonrigid-icp-transform>
            -i ${CMAKE_SOURCE_DIR}/data/moving_pc.csv
            -o ${CMAKE_BINARY_DIR}/results/gpu_backend_not_implemented.csv
            -t ${CMAKE_SOURCE_DIR}/test/golden/data_fixed_moving_v50_b1_i15_w001_e2_n35000.nricp
            -x gpu
            -s
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(nonrigid-icp-transform.GpuBackendNotImplemented PROPERTIES WILL_FAIL TRUE)

if(NRICP_ENABLE_CUDA)
    add_test(
        NAME cuda-smoke-tests.Runtime
        COMMAND $<TARGET_FILE:cuda-smoke-tests>
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Add tests for the shell scripts
add_test(
    NAME nonrigid-icp.EagleScript
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/test/test-eagle.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)
add_test(
    NAME nonrigid-icp.NordbahnScript
    COMMAND /bin/bash ${CMAKE_SOURCE_DIR}/test/test-nordbahn.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

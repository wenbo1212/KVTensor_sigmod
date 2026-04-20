cmake_minimum_required(VERSION 3.15)
project(kvtensor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_MICROBENCH "Build performance-model microbenchmarks" ON)
option(ENABLE_AVX2 "Enable AVX2 SIMD optimizations for bfloat16" ON)
option(ENABLE_F16C "Enable F16C instructions for float16 conversion" ON)
option(ENABLE_ONEMKL "Enable optional oneMKL integration when available" OFF)

# Find dependencies
find_package(Threads REQUIRED)
find_package(leveldb QUIET)
find_package(dnnl REQUIRED)

# oneMKL (optional)
if(ENABLE_ONEMKL)
    find_path(MKL_INCLUDE_DIR mkl.h HINTS ENV MKLROOT PATH_SUFFIXES include)
    find_library(MKL_RT_LIB mkl_rt HINTS ENV MKLROOT PATH_SUFFIXES lib lib/intel64)
else()
    unset(MKL_INCLUDE_DIR CACHE)
    unset(MKL_RT_LIB CACHE)
endif()

# Find OpenMP (required for MKL threading)
# MKL's mkl_rt dynamically loads threading libraries, which require OpenMP
find_package(OpenMP)
if (NOT OpenMP_CXX_FOUND)
    # Try to find Intel OpenMP as fallback (often bundled with MKL)
    find_library(IOMP5_LIB 
        NAMES iomp5 libiomp5
        HINTS 
            ENV MKLROOT 
            ENV INTEL_ONEAPI_ROOT
        PATH_SUFFIXES 
            lib 
            lib/intel64
            ../compiler/latest/lib
            ../compiler/latest/lib/intel64
    )
endif()

if (ENABLE_ONEMKL AND MKL_INCLUDE_DIR AND MKL_RT_LIB)
    add_definitions(-DKVTENSOR_USE_ONEMKL)
    message(STATUS "oneMKL enabled")
elseif(ENABLE_ONEMKL)
    message(STATUS "oneMKL requested but not found; building without oneMKL")
else()
    message(STATUS "oneMKL disabled by ENABLE_ONEMKL=OFF")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(KVTENSOR_SOURCES
    src/storage/simpledb_storage.cpp
    src/storage/registry.cpp
    src/matrix/block_matrix.cpp
    src/context/operator_context.cpp
    src/bufferpool/pool_gc.cpp
    src/operators/matmul.cpp
    src/operators/rmsnorm.cpp
    src/operators/layernorm.cpp
    src/operators/attention.cpp
    src/operators/feedforward.cpp
    src/operators/elementwise.cpp
    src/operators/activations.cpp
    src/operators/sd_conv_im2col.cpp
    src/operators/sd_group_norm.cpp
    src/utils/preload_utils.cpp
    src/utils/model_utils.cpp
    src/utils/prefetch_graph.cpp
    src/utils/prefetch_graph_provider.cpp
    src/utils/profile.cpp
    src/models/llama_transformer.cpp
    src/models/llama_inference.cpp
    src/models/llama_model.cpp
    src/models/stable_diffusion_model.cpp
    src/models/stable_diffusion_inference.cpp
)

# Header files
set(KVTENSOR_HEADERS
    include/kvtensor/types.hpp
    include/kvtensor/storage.hpp
    include/kvtensor/matrix.hpp
    include/kvtensor/context.hpp
    include/kvtensor/operators.hpp
    include/kvtensor/operators_impl.hpp
    include/kvtensor/preload_utils.hpp
    include/kvtensor/model_utils.hpp
    include/kvtensor/prefetch_graph.hpp
    include/kvtensor/prefetch_graph_provider.hpp
    include/kvtensor/profile.hpp
    include/kvtensor/llama_inference.hpp
    include/kvtensor/llama.hpp
    include/kvtensor/stable_diffusion.hpp
    include/kvtensor/stable_diffusion_inference.hpp
    include/kvtensor/sd_ops.hpp
    include/kvtensor/simpledb.hpp
    include/math/arithmetic.hpp
)

# SimpleKVStore library (standalone KV store)
add_library(simpledb STATIC
    src/simpledb/simpledb.cpp
)
target_include_directories(simpledb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create library
add_library(kvtensor STATIC ${KVTENSOR_SOURCES} ${KVTENSOR_HEADERS})
target_link_libraries(kvtensor PUBLIC simpledb)

# Compiler-specific optimizations for SIMD support
# Note: These flags enable SIMD instructions at compile time. The code includes
# fallbacks for CPUs without AVX2/AVX support. However, if compiled with AVX2
# and run on a CPU without AVX2, the program may crash with an illegal instruction.
# For production, consider using runtime CPU feature detection or separate builds.
if(ENABLE_AVX2)
    # Check if compiler supports AVX2
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2_MSVC)
    
    if(COMPILER_SUPPORTS_AVX2)
        # GCC/Clang
        target_compile_options(kvtensor PRIVATE -mavx2)
        target_compile_definitions(kvtensor PRIVATE __AVX2__)
        message(STATUS "AVX2 support enabled for bfloat16 SIMD optimizations")
    elseif(COMPILER_SUPPORTS_AVX2_MSVC)
        # MSVC
        target_compile_options(kvtensor PRIVATE /arch:AVX2)
        target_compile_definitions(kvtensor PRIVATE __AVX2__)
        message(STATUS "AVX2 support enabled for bfloat16 SIMD optimizations")
    else()
        # Fallback to AVX if AVX2 not supported
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX_MSVC)
        
        if(COMPILER_SUPPORTS_AVX)
            target_compile_options(kvtensor PRIVATE -mavx)
            target_compile_definitions(kvtensor PRIVATE __AVX__)
            message(STATUS "AVX support enabled (AVX2 not available)")
        elseif(COMPILER_SUPPORTS_AVX_MSVC)
            target_compile_options(kvtensor PRIVATE /arch:AVX)
            target_compile_definitions(kvtensor PRIVATE __AVX__)
            message(STATUS "AVX support enabled (AVX2 not available)")
        else()
            message(WARNING "AVX/AVX2 not supported by compiler. Bfloat16 SIMD optimizations will be disabled.")
        endif()
    endif()
endif()

# F16C support for float16 conversion (requires AVX)
if(ENABLE_F16C)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)
    check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_F16C_MSVC)
    
    if(COMPILER_SUPPORTS_F16C)
        # GCC/Clang - F16C is part of AVX, so if AVX is enabled, F16C should work
        # But we can explicitly enable it
        target_compile_options(kvtensor PRIVATE -mf16c)
        target_compile_definitions(kvtensor PRIVATE __F16C__)
        message(STATUS "F16C support enabled for float16 conversion")
    elseif(COMPILER_SUPPORTS_F16C_MSVC)
        # MSVC - F16C is included with AVX
        target_compile_definitions(kvtensor PRIVATE __F16C__)
        message(STATUS "F16C support enabled (via AVX)")
    else()
        message(STATUS "F16C not explicitly enabled (may still work if AVX is enabled)")
    endif()
endif()

target_link_libraries(kvtensor PUBLIC leveldb::leveldb)
target_link_libraries(kvtensor PUBLIC dnnl)
if (ENABLE_ONEMKL AND MKL_INCLUDE_DIR AND MKL_RT_LIB)
    target_include_directories(kvtensor PUBLIC ${MKL_INCLUDE_DIR})
    target_link_libraries(kvtensor PUBLIC ${MKL_RT_LIB})
    
    # Link OpenMP if available (required for MKL threading)
    # MKL's mkl_rt uses dynamic loading, but the threading libraries need OpenMP symbols
    if (OpenMP_CXX_FOUND)
        target_link_libraries(kvtensor PUBLIC OpenMP::OpenMP_CXX)
        # Also add compile flags for OpenMP
        target_compile_options(kvtensor PUBLIC ${OpenMP_CXX_FLAGS})
        message(STATUS "MKL: Linked with OpenMP (${OpenMP_CXX_VERSION})")
    elseif(IOMP5_LIB)
        target_link_libraries(kvtensor PUBLIC ${IOMP5_LIB})
        message(STATUS "MKL: Linked with Intel OpenMP")
    else()
        message(WARNING "OpenMP not found. MKL threading may fail at runtime.")
        message(WARNING "Solutions: Install OpenMP or set MKL_INTERFACE_LAYER=LP64,sequential")
    endif()
else()
    # Even without MKL, OpenMP can be useful for parallel operations
    if (OpenMP_CXX_FOUND)
        target_link_libraries(kvtensor PUBLIC OpenMP::OpenMP_CXX)
        target_compile_options(kvtensor PUBLIC ${OpenMP_CXX_FLAGS})
        message(STATUS "OpenMP enabled for parallel operations (${OpenMP_CXX_VERSION})")
    endif()
endif()

target_include_directories(kvtensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Example executable
if(BUILD_EXAMPLES)
    add_executable(llama_inference_example examples/llama_inference.cpp)
    target_link_libraries(llama_inference_example PRIVATE kvtensor)
    
    add_executable(simpledb_generate_weights examples/simpledb_generate_weights.cpp)
    target_link_libraries(simpledb_generate_weights PRIVATE simpledb)

    add_executable(simpledb_generate_sd_weights examples/simpledb_generate_sd_weights.cpp)
    target_link_libraries(simpledb_generate_sd_weights PRIVATE simpledb)

    add_executable(stable_diffusion_inference_example examples/stable_diffusion_inference.cpp)
    target_link_libraries(stable_diffusion_inference_example PRIVATE kvtensor)

    add_executable(prefetch_graph_example examples/prefetch_graph_example.cpp)
    target_link_libraries(prefetch_graph_example PRIVATE kvtensor)

    add_executable(export_layers_to_bin examples/export_layers_to_bin.cpp)
    target_link_libraries(export_layers_to_bin PRIVATE kvtensor)

    if (leveldb_FOUND)
        # LevelDB benchmark tool
        add_executable(leveldb_benchmark examples/leveldb_benchmark.cpp)
        target_link_libraries(leveldb_benchmark PRIVATE leveldb)
        target_include_directories(leveldb_benchmark PRIVATE ${LEVELDB_INCLUDE_DIR})

        add_executable(simpledb_benchmark examples/simpledb_benchmark.cpp)
        target_link_libraries(simpledb_benchmark PRIVATE simpledb leveldb)
        target_include_directories(simpledb_benchmark PRIVATE ${LEVELDB_INCLUDE_DIR})
    endif()
endif()

if(BUILD_MICROBENCH)
    add_subdirectory(microbench)
endif()

# Tests
if(BUILD_TESTS)
    add_executable(kvtensor_operator_tests tests/operator_tests.cpp)
    target_link_libraries(kvtensor_operator_tests PRIVATE kvtensor)
endif()

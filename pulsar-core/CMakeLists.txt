cmake_minimum_required(VERSION 3.26)
project(pulsar_core LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA gencode target(s) for nvcc")
endif()

option(PULSAR_PROF "Compile the PULSAR_PROF phase timers into the engine" OFF)

option(BUILD_TESTS "Build the Catch2 test binaries" ON)

if(NOT Torch_DIR)
    find_package(Python REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import torch.utils; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_cmake_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _torch_query_result)
    if(NOT _torch_query_result EQUAL 0)
        message(FATAL_ERROR
            "libtorch not found. Pass -DCMAKE_PREFIX_PATH=<libtorch root> or "
            "-DTorch_DIR=<...>, or make `import torch` work for ${Python_EXECUTABLE}.")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
endif()

find_package(Torch REQUIRED)
include(GNUInstallDirs)

add_library(pulsar_core STATIC
    src/kernels/rmsnorm.cu
    src/kernels/attention.cu
    src/kernels/paged_attention.cu
    src/kernels/paged_prefill.cu
    src/kernels/w4a16_gemm.cu
    src/kernels/w4a16_marlin.cu
    src/kernels/rope.cu
    src/kernels/reposition.cu
    src/runtime/kv/active_buffer.cpp
    src/runtime/scheduler.cpp
    src/model/compiled_model.cpp
    src/model/paged_forward.cpp
    src/model/qwen2.cpp
    src/model/qwen3.cpp
    src/runtime/kv/recall.cpp
    src/runtime/kv/paged_pool.cpp
    src/runtime/kv/host_pages.cpp
    src/runtime/kv/paged_bucket.cpp
    src/runtime/kv/file_bucket.cpp
    src/runtime/kv/kv_memory.cpp
    src/runtime/sampler.cpp
    src/runtime/engine.cpp
)
target_compile_features(pulsar_core PUBLIC cxx_std_20 cuda_std_20)
# TORCH_INCLUDE_DIRS explicitly, not left to TorchConfig.cmake's directory-scoped
# include_directories(): that doesn't reach an exported target's own consumers.
target_include_directories(pulsar_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(pulsar_core PUBLIC ${TORCH_LIBRARIES})
# Some TORCH_LIBRARIES entries (e.g. torch, torch_library) are CMake targets that
# export as bare -l flags; this gives a consumer's linker the -L to resolve them.
target_link_directories(pulsar_core PUBLIC "${TORCH_INSTALL_PREFIX}/lib")
if(PULSAR_PROF)
    target_compile_definitions(pulsar_core PUBLIC PULSAR_PROF)
endif()
target_compile_options(pulsar_core PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>"
)
set_target_properties(pulsar_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)
    # Catch2 must be compiled with the same libstdc++ ABI as libtorch.
    target_compile_options(Catch2 PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>"
    )

    enable_testing()
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(Catch)

    set(_pulsar_core_tests
        kernels/test_ops
        runtime/test_scheduler
        runtime/test_recall
        runtime/test_sampler
        runtime/test_kv_memory
        runtime/test_spill_bucket
        runtime/test_engine
    )
    foreach(_path ${_pulsar_core_tests})
        get_filename_component(_test ${_path} NAME)
        add_executable(${_test} "tests/${_path}.cpp")
        target_link_libraries(${_test} PRIVATE pulsar_core Catch2::Catch2WithMain)
        set_target_properties(${_test} PROPERTIES
            BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        )
        catch_discover_tests(${_test})
    endforeach()
endif()

install(
    TARGETS pulsar_core
    EXPORT pulsar_core-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
    EXPORT pulsar_core-targets
    FILE pulsar_core-targets.cmake
    NAMESPACE pulsar_core::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pulsar_core"
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pulsar_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pulsar_core-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pulsar_core"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pulsar_core-config-version.cmake"
    VERSION 0.0.0
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/pulsar_core-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/pulsar_core-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/pulsar_core"
)

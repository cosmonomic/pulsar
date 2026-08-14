cmake_minimum_required(VERSION 3.26)
project(pulsar LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../pulsar-core"
    "${CMAKE_CURRENT_BINARY_DIR}/pulsar-core"
)

find_package(Torch REQUIRED)

add_library(pulsar_core_bindings SHARED
    ext/ops.cpp
    ext/engine.cpp
)
target_link_libraries(pulsar_core_bindings PRIVATE pulsar_core)
set_target_properties(pulsar_core_bindings PROPERTIES
    OUTPUT_NAME pulsar_core
    PREFIX ""
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

include(GNUInstallDirs)
install(TARGETS pulsar_core_bindings LIBRARY DESTINATION pulsar COMPONENT python)

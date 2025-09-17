cmake_minimum_required(VERSION 3.12)
project(nms_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/nms_ops.cpp
    python_bindings/pybind_module.cpp
)

# Create the pybind11 module
pybind11_add_module(nms_ops_cpp ${SOURCES})

# Compiler-specific options
target_compile_definitions(nms_ops_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Enable optimization
if (MSVC)
    target_compile_options(nms_ops_cpp PRIVATE /O2)
else()
    target_compile_options(nms_ops_cpp PRIVATE -O3)
endif()

# Strip unnecessary symbols to reduce file size
if (NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip debug symbols
    set_target_properties(nms_ops_cpp PROPERTIES LINK_FLAGS_RELEASE -s)
endif()
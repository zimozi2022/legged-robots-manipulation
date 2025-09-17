cmake_minimum_required(VERSION 3.14...3.22)
project(nms_ops)

# Find required packages
find_package(pybind11 REQUIRED)

# Create the extension module
pybind11_add_module(nms_ops 
    src/pybind.cpp
)

# Set C++ standard
target_compile_features(nms_ops PRIVATE cxx_std_14)

# Set optimization flags
target_compile_definitions(nms_ops PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Enable optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(nms_ops PRIVATE -O3)
endif()
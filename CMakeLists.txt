cmake_minimum_required(VERSION 3.20)
project(diffsoup VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------
# Python (provided by scikit-build-core)
# ----------------------------------
find_package(Python ${PYTHON_VERSION_HINT} COMPONENTS Interpreter Development REQUIRED)

# ----------------------------------
# nanobind (from pip, not submodule)
# ----------------------------------
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ----------------------------------
# CUDA
# ----------------------------------
find_package(CUDAToolkit REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

# ----------------------------------
# Build nanobind module inside package
# ----------------------------------
nanobind_add_module(_core
    src/main.cpp
    src/remesh.h
    src/remesh.cpp
    src/remesh_clip.h
    src/remesh_clip.cpp
    src/cuda/rasterize.cuh
    src/cuda/rasterize.cu
    src/cuda/multires.cuh
    src/cuda/multires.cu
    src/cuda/encoding.cuh
    src/cuda/encoding.cu
)

set_target_properties(_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(_core PRIVATE
    src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Optional but common for CUDA-heavy projects
set_target_properties(_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Put compiled extension inside the diffsoup package in site-packages
install(TARGETS _core
    LIBRARY DESTINATION diffsoup
    RUNTIME DESTINATION diffsoup
    ARCHIVE DESTINATION diffsoup
)

cmake_minimum_required(VERSION 3.20)
project(diffsoupviewer VERSION 0.1.0 LANGUAGES C CXX)
# NOTE: 'C' is needed for glad's generated loader source.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Everything will be linked into a shared Python extension → need PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ==================================================================
# Python  (provided by scikit-build-core; needed early for glad generation)
# ==================================================================
find_package(Python ${PYTHON_VERSION_HINT}
    COMPONENTS Interpreter Development REQUIRED)

# ==================================================================
# 1.  GLFW  —  windowing / input / OpenGL context
# ==================================================================
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)

# ==================================================================
# 2.  GLM  —  header-only maths (vec3, mat4, etc.)
# ==================================================================
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)

# ==================================================================
# 3.  glad  —  OpenGL function loader
#     Generated at configure time via the `glad2` pip package
#     (listed in pyproject.toml build-requires).
# ==================================================================
set(GLAD_OUT "${CMAKE_BINARY_DIR}/glad_generated")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m glad
            --api gl:core=4.1
            --out-path "${GLAD_OUT}"
            --reproducible
            c
    RESULT_VARIABLE _glad_rc
)
if(NOT _glad_rc EQUAL 0)
    message(FATAL_ERROR "glad generation failed (rc=${_glad_rc})")
endif()

add_library(glad_gl STATIC
    "${GLAD_OUT}/src/gl.c"
)
target_include_directories(glad_gl PUBLIC "${GLAD_OUT}/include")

# ==================================================================
# 4.  Dear ImGui  —  immediate-mode GUI
#     No CMakeLists.txt upstream, so we build it ourselves below.
# ==================================================================
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
    GIT_SHALLOW    TRUE
)

# ------------------------------------------------------------------
# Populate everything
# ------------------------------------------------------------------
FetchContent_MakeAvailable(glfw glm)

# imgui: populate without add_subdirectory (no CMakeLists.txt)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

# ------------------------------------------------------------------
# Dear ImGui: build a small static library (core + GLFW + OpenGL3 backends)
# ------------------------------------------------------------------
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(imgui_lib PUBLIC glfw glad_gl)

# ==================================================================
# nanobind  (installed via pip, located through its CLI)
# ==================================================================
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ==================================================================
# Build the extension module
# ==================================================================
nanobind_add_module(_core
    src/main.cpp
    src/viewer.cpp
    src/camera.cpp
)

set_target_properties(_core PROPERTIES
    CXX_STANDARD     17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS    NO
)

target_include_directories(_core PRIVATE
    src
)

target_link_libraries(_core PRIVATE
    glfw
    glad_gl
    glm::glm
    imgui_lib
)

# ------------------------------------------------------------------
# On macOS, GLFW needs Cocoa / IOKit / CoreVideo frameworks
# ------------------------------------------------------------------
if(APPLE)
    target_link_libraries(_core PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# ------------------------------------------------------------------
# Install into the Python package directory
# ------------------------------------------------------------------
install(TARGETS _core
    LIBRARY DESTINATION diffsoupviewer
    RUNTIME DESTINATION diffsoupviewer
    ARCHIVE DESTINATION diffsoupviewer
)

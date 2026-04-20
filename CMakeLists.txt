cmake_minimum_required(VERSION 3.20)
project(gpu_raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(raytracer
    src/main.cpp
    src/vulkan_app.cpp
    src/bvh.cpp
)

target_include_directories(raytracer PRIVATE src)
target_link_libraries(raytracer PRIVATE Vulkan::Vulkan glfw glm::glm)
target_compile_options(raytracer PRIVATE -O2 -Wall -Wextra -Wno-missing-field-initializers)
target_compile_definitions(raytracer PRIVATE SHADER_DIR="${CMAKE_BINARY_DIR}/shaders/")

# Shader compilation
find_program(GLSLC glslc HINTS Vulkan::glslc REQUIRED)

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

set(SHADER_SOURCES
    ${SHADER_SRC_DIR}/raytracer.comp
    ${SHADER_SRC_DIR}/display.vert
    ${SHADER_SRC_DIR}/display.frag
)

foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(NAME ${SHADER} NAME)
    set(SPV ${SHADER_BIN_DIR}/${NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -O ${SHADER} -o ${SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${NAME}"
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_FILES})
add_dependencies(raytracer shaders)

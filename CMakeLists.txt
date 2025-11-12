cmake_minimum_required(VERSION 3.18)

project(dl_framework CXX CUDA)

if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    message(FATAL_ERROR "Fatal: CUDA only supports MSVC on Windows")
endif ()

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(dl_framework_demo demo.cu)

set_target_properties(dl_framework_demo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(dl_framework_demo PROPERTIES
        CUDA_ARCHITECTURES "native"
)

target_include_directories(dl_framework_demo
        PUBLIC
        ${CMAKE_SOURCE_DIR}/nn/
        ${CMAKE_SOURCE_DIR}/nn/backend/
        ${CMAKE_SOURCE_DIR}/nn/codegen/generated
)

target_compile_options(dl_framework_demo
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler "/Ox /arch:AVX2 /Oi /Ot /GS-"
        >
)

target_link_libraries(dl_framework_demo PRIVATE cublas cudnn)

# todo cpu version
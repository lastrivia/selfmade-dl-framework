cmake_minimum_required(VERSION 3.18)

project(dl_framework CXX CUDA)

if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    message(FATAL_ERROR "Fatal: CUDA only supports MSVC on Windows")
endif ()

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(cs3339_demo cs3339_demo.cu)

set_target_properties(cs3339_demo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(cs3339_demo PROPERTIES
        CUDA_ARCHITECTURES "native"
)

target_include_directories(cs3339_demo
        PUBLIC
        ${CMAKE_SOURCE_DIR}/nn/
        ${CMAKE_SOURCE_DIR}/nn/codegen/generated
)

target_compile_options(cs3339_demo
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler "/Ox /arch:AVX2 /Oi /Ot /GS-"
        >
)

target_link_libraries(cs3339_demo PRIVATE cublas cudnn)

# todo cpu version
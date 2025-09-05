cmake_minimum_required(VERSION 3.10)
project(nn_demo_cpu)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn_demo_cpu demo.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nn_demo_cpu PRIVATE
            -O3
            -march=native
#            -ffast-math
            -save-temps
            -fverbose-asm
    )
    get_filename_component(GCC_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    message(STATUS "GCC bin dir: ${GCC_BIN_DIR}")

    set(LIBSTDCXX_DLL "${GCC_BIN_DIR}/libstdc++-6.dll")
    set(LIBGCC_DLL    "${GCC_BIN_DIR}/libgcc_s_seh-1.dll")
    set(LIBWPT_DLL    "${GCC_BIN_DIR}/libwinpthread-1.dll")

    add_custom_command(TARGET nn_demo_cpu POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBSTDCXX_DLL} $<TARGET_FILE_DIR:nn_demo_cpu>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBGCC_DLL}    $<TARGET_FILE_DIR:nn_demo_cpu>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBWPT_DLL}    $<TARGET_FILE_DIR:nn_demo_cpu>
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(nn_demo_cpu PRIVATE
            /Ox
            /GL
#            /fp:fast
            /arch:AVX2
            /Oi
            /Ot
            /GS-
            /Fa
            /FAs
    )
    target_link_options(nn_demo_cpu PRIVATE /LTCG)
endif()


cmake_minimum_required(VERSION 3.10)
project(dl_framework_demo)

set(CMAKE_CXX_STANDARD 20)

add_executable(dl_framework_demo demo.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(dl_framework_demo PRIVATE
            -O3
            -march=native
            -save-temps
            -fverbose-asm
    )
    get_filename_component(GCC_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    message(STATUS "GCC bin dir: ${GCC_BIN_DIR}")

    set(LIBSTDCXX_DLL "${GCC_BIN_DIR}/libstdc++-6.dll")
    set(LIBGCC_DLL    "${GCC_BIN_DIR}/libgcc_s_seh-1.dll")
    set(LIBWPT_DLL    "${GCC_BIN_DIR}/libwinpthread-1.dll")

    add_custom_command(TARGET dl_framework_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBSTDCXX_DLL} $<TARGET_FILE_DIR:dl_framework_demo>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBGCC_DLL}    $<TARGET_FILE_DIR:dl_framework_demo>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIBWPT_DLL}    $<TARGET_FILE_DIR:dl_framework_demo>
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(dl_framework_demo PRIVATE
            /Ox
            /GL
            /arch:AVX2
            /Oi
            /Ot
            /GS-
            /Fa
            /FAs
    )
    target_link_options(dl_framework_demo PRIVATE /LTCG)
endif()


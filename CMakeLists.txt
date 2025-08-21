cmake_minimum_required(VERSION 3.10)
project(nn_demo_cpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer -O3 -march=native -ffast-math")

add_executable(nn_demo_cpu main.cpp)

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
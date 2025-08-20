cmake_minimum_required(VERSION 3.31)
project(TestCpu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

add_executable(TestCpu main.cpp
)

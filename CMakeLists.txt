cmake_minimum_required(VERSION 3.16)

project(HyperWavelet)

include_directories(include)
include_directories(/usr/include/eigen3)

add_subdirectory(src)
add_subdirectory(examples)


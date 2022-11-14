cmake_minimum_required(VERSION 3.16)

project(HyperWavelet)

include_directories(include)
include_directories(/usr/include/eigen3)

add_compile_options(-O3 -std=c++17 -fopenmp)
add_link_options(-lpthread -fopenmp)

add_subdirectory(src)
add_subdirectory(examples)


cmake_minimum_required(VERSION 3.16)

project(hyperwavelet)

set(SOURCE_LIB bases.cpp colocations.cpp helpers.cpp surface.cpp mesh.cpp)

add_library(hyperwavelet STATIC ${SOURCE_LIB})

target_link_directories(hyperwavelet INTERFACE ~/local/lib/libmetis.a)
cmake_minimum_required(VERSION 3.16)

project(hyperwavelet)

set(SOURCE_LIB bases.cpp colocations.cpp helpers.cpp surface.cpp)

add_library(hyperwavelet STATIC ${SOURCE_LIB})

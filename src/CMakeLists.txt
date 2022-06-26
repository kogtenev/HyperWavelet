cmake_minimum_required(VERSION 3.16)

project(hyperwavelet)

set(SOURCE_LIB bases.cpp colocations.cpp)

add_library(hyperwavelet STATIC ${SOURCE_LIB})

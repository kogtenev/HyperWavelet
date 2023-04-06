cmake_minimum_required(VERSION 3.16)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

project(HyperWavelet)

include_directories(include)
include_directories(/usr/include/eigen3)
include_directories(~/local/include)

add_compile_options(-O3 -std=c++17 -fopenmp)
add_link_options(-lpthread -fopenmp)

add_subdirectory(src)
add_subdirectory(examples)


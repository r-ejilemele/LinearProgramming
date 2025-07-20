cmake_minimum_required(VERSION 3.4...3.18)
project(pybindtest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
include_directories("$ENV{CONDA_PREFIX}/Library/include/eigen3")
include_directories("$ENV{CONDA_PREFIX}/Library/include")
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(linear main.cpp)
target_link_libraries(linear PRIVATE "$ENV{CONDA_PREFIX}/Library/lib/gmp.lib")
target_link_libraries(linear PRIVATE "$ENV{CONDA_PREFIX}/Library/lib/mpfr.lib")
set_target_properties(linear PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "C:/Users/rejil/Documents/GitHub/LinearProgramming"
)

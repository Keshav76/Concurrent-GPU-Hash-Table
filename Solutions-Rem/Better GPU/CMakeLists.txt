cmake_minimum_required(VERSION 3.2)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/gcc-10 -std=c++14 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-std=c++14")

project(CPMTest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_SOURCE_DIR}/../../cmake/CPM.cmake)  # REPLACE THIS LINE WITH:
include($ENV{HOME}/cmake_modules/CPM.cmake)  # Now it points to your local CPM

CPMAddPackage(
  NAME bght
  GITHUB_REPOSITORY owensgroup/BGHT
  GIT_TAG main
  OPTIONS
     "build_tests OFF"
     "build_benchmarks OFF"
     "build_examples OFF"
)

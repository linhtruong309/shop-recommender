cmake_minimum_required(VERSION 3.16)
project(reco_service CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# SIMD dot products: vectorize the scoring loop (portable, no OpenMP runtime needed)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp-simd" HAS_OPENMP_SIMD)
if(HAS_OPENMP_SIMD)
  add_compile_options(-fopenmp-simd)
endif()

# HTTP server: single-header cpp-httplib
include(FetchContent)
FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
  GIT_SHALLOW TRUE
)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib)

find_package(Threads REQUIRED)

# Core library: artifact loading + top-k search
add_library(recommender src/recommender.cpp)
target_include_directories(recommender PUBLIC include)
target_link_libraries(recommender PUBLIC Threads::Threads)

# REST server
add_executable(reco_server src/server.cpp)
target_link_libraries(reco_server PRIVATE recommender httplib::httplib)

# Latency benchmark
add_executable(reco_bench src/bench.cpp)
target_link_libraries(reco_bench PRIVATE recommender)

# Unit tests
enable_testing()
add_executable(test_recommender tests/test_recommender.cpp)
target_link_libraries(test_recommender PRIVATE recommender)
add_test(NAME test_recommender COMMAND test_recommender)

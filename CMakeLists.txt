cmake_minimum_required(VERSION 3.20)
project(hot40i_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(h40_runtime
    device/runtime/src/flash_tensor_provider.cpp
    device/runtime/src/ram_arena.cpp
    device/runtime/src/expert_cache.cpp
    device/runtime/src/model_index.cpp
)
target_include_directories(h40_runtime PUBLIC device/runtime/include)
target_compile_options(h40_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(h40_io_bench device/bench/io_bench.cpp)
target_link_libraries(h40_io_bench PRIVATE h40_runtime)

add_executable(h40_runtime_smoke tests/runtime_smoke.cpp)
target_link_libraries(h40_runtime_smoke PRIVATE h40_runtime)

enable_testing()
add_test(NAME runtime_smoke COMMAND h40_runtime_smoke)

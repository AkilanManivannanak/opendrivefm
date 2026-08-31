# OpenDriveFM C++ deployment runtime and validation harness.
#
# The header-only primitives and their tests build with no dependencies, so
# `ctest` runs in CI on any machine. The LibTorch tools (runner, parity check)
# build only when Torch is found, so a missing LibTorch degrades to "fewer
# targets", never to a broken configure step.
#
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
#   cmake --build cpp/build -j
#   ctest --test-dir cpp/build --output-on-failure
cmake_minimum_required(VERSION 3.18)
project(odfm_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native is deliberately NOT default: it makes binaries non-portable and
# makes latency numbers unreproducible across machines. Opt in explicitly.
option(ODFM_NATIVE_ARCH "Build with -march=native (faster, non-portable)" OFF)
if(ODFM_NATIVE_ARCH)
  add_compile_options(-march=native)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)

add_library(odfm_core INTERFACE)
target_include_directories(odfm_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(odfm_core INTERFACE Threads::Threads)

# ── Tests: no LibTorch required ───────────────────────────────────────────────
enable_testing()
foreach(t test_spsc_ring test_latency_stats)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE odfm_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# ThreadSanitizer build of the concurrency test. A lock-free queue that passes
# a plain test can still be racy; TSan is what actually checks the memory
# orderings. Build with -DODFM_TSAN=ON.
option(ODFM_TSAN "Build a ThreadSanitizer copy of the concurrency tests" OFF)
if(ODFM_TSAN)
  add_executable(test_spsc_ring_tsan tests/test_spsc_ring.cpp)
  target_link_libraries(test_spsc_ring_tsan PRIVATE odfm_core)
  target_compile_options(test_spsc_ring_tsan PRIVATE -fsanitize=thread -g -O1)
  target_link_options(test_spsc_ring_tsan PRIVATE -fsanitize=thread)
  add_test(NAME test_spsc_ring_tsan COMMAND test_spsc_ring_tsan)
endif()

# ── LibTorch tools: optional ──────────────────────────────────────────────────
find_package(Torch QUIET)
if(Torch_FOUND)
  message(STATUS "LibTorch found: building odfm_runner and odfm_parity_check")
  foreach(tool odfm_runner odfm_parity_check)
    add_executable(${tool} src/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE odfm_core "${TORCH_LIBRARIES}")
    set_property(TARGET ${tool} PROPERTY CXX_STANDARD 17)
  endforeach()
else()
  message(STATUS "LibTorch NOT found: header-only targets and tests only. "
                 "Pass -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')")
endif()

cmake_minimum_required(VERSION 2.6)

project(argon2-gpu CXX)
set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -std=c++11;-O3;--ptxas-options=-v;-arch sm_30;-lineinfo
)

add_library(argon2-gpu-common SHARED
    lib/argon2-gpu-common/argon2params.cpp
    lib/argon2-gpu-common/blake2b.cpp
)
target_include_directories(argon2-gpu-common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-gpu-common PRIVATE
    include/argon2-gpu-common
    lib/argon2-gpu-common
)

cuda_add_library(argon2-cuda SHARED
    lib/argon2-cuda/device.cpp
    lib/argon2-cuda/globalcontext.cpp
    lib/argon2-cuda/programcontext.cpp
    lib/argon2-cuda/processingunit.cpp
    lib/argon2-cuda/kernels.cu
)
target_include_directories(argon2-cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-cuda PRIVATE
    include/argon2-cuda
    lib/argon2-cuda
)
target_link_libraries(argon2-cuda argon2-gpu-common)

add_library(argon2-opencl SHARED
    lib/argon2-opencl/device.cpp
    lib/argon2-opencl/globalcontext.cpp
    lib/argon2-opencl/kernelloader.cpp
    lib/argon2-opencl/programcontext.cpp
    lib/argon2-opencl/processingunit.cpp
)
target_include_directories(argon2-opencl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-opencl PRIVATE
    include/argon2-opencl
    lib/argon2-opencl
)
target_link_libraries(argon2-opencl
    argon2-gpu-common -lOpenCL
)

add_executable(argon2-gpu-test src/argon2-gpu-test/main.cpp)
target_include_directories(argon2-gpu-test PRIVATE src/argon2-gpu-test)
target_link_libraries(argon2-gpu-test
    argon2-cuda argon2-opencl -lOpenCL
)

add_executable(argon2-gpu-bench
    src/argon2-gpu-bench/cudaexecutive.cpp
    src/argon2-gpu-bench/openclexecutive.cpp
    src/argon2-gpu-bench/benchmark.cpp
    src/argon2-gpu-bench/main.cpp
)
target_include_directories(argon2-gpu-bench PRIVATE src/argon2-gpu-bench)
target_link_libraries(argon2-gpu-bench
    argon2-cuda argon2-opencl -lOpenCL
)

add_test(test argon2-gpu-test)

install(TARGETS argon2-opencl DESTINATION ${LIBRARY_INSTALL_DIR})
install(FILES
    include/argon2-opencl/cl.hpp
    include/argon2-opencl/opencl.h
    include/argon2-opencl/argon2-common.h
    include/argon2-opencl/argon2params.h
    include/argon2-opencl/device.h
    include/argon2-opencl/globalcontext.h
    include/argon2-opencl/programcontext.h
    include/argon2-opencl/processingunit.h
    DESTINATION ${INCLUDE_INSTALL_DIR}
)
install(TARGETS argon2-gpu-bench argon2-gpu-test DESTINATION ${BINARY_INSTALL_DIR})

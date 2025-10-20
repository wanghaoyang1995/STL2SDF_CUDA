cmake_minimum_required(VERSION 3.24)

project(stl2sdf_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \"-Xcompiler= /utf-8\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

include_directories(./common)

add_executable(stl2sdf_cuda_ascii
        src/main.cpp
        src/stl.cpp
        src/sdf.cu
        src/cuda_common.cu
)
add_executable(stl2sdf_cuda_binary
        src/main.cpp
        src/stl.cpp
        src/sdf.cu
        src/cuda_common.cu
)

target_compile_definitions(stl2sdf_cuda_binary PRIVATE USE_BINARY_STL)

set_target_properties(stl2sdf_cuda_ascii PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(stl2sdf_cuda_binary PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(stl2sdf_cuda_ascii PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_include_directories(stl2sdf_cuda_binary PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
)

add_custom_command(TARGET stl2sdf_cuda_ascii POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:stl2sdf_cuda_ascii>/data
)

add_custom_command(TARGET stl2sdf_cuda_binary POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:stl2sdf_cuda_binary>/data
)

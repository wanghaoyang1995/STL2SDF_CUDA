//
// Created by wangh on 2025/4/24.
//

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "helper_cuda.h"
#include "define.h"

#if USE_CUDA_2D_GRID
#define CUDA_THREAD_IDX ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x)
#else
#define CUDA_THREAD_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#endif

template <typename T1, typename T2>
void thrust_exclusive_scan(T2* output, T1* input, uint numElements) {
    static_assert(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>, "T1, T2 must be numeric!");

    thrust::exclusive_scan(thrust::device_ptr<T1>(input),
                           thrust::device_ptr<T1>(input + numElements),
                           thrust::device_ptr<T2>(output),
                           static_cast<T2>(0),
                           thrust::plus<T2>());  // T1隐式转换为T2
}

template <typename T1, typename T2>
void thrust_inclusive_scan(T2* output, T1* input, uint numElements) {
    static_assert(std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>, "T1, T2 must be numeric!");

    thrust::inclusive_scan(thrust::device_ptr<T1>(input),
                           thrust::device_ptr<T1>(input + numElements),
                           thrust::device_ptr<T2>(output),
                           static_cast<T2>(0),
                           thrust::plus<T2>());
}

template <typename T1, typename T2>
T2 scan_sum(T1* d_array, T2* d_array_scan, uint length) {
    T1 last_element;
    T2 last_scan_element;
    checkCudaErrors(cudaMemcpy((void *)&last_element,
                               (void *)(d_array + length - 1),
                               sizeof(T1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void *)&last_scan_element,
                               (void *)(d_array_scan + length - 1),
                               sizeof(T2), cudaMemcpyDeviceToHost));
    T2 total = static_cast<T2>(last_element) + last_scan_element;
    return total;
}

template <typename T>
inline T scan_last(T* d_array_scan, uint length) {
    T last_scan_element;
    checkCudaErrors(cudaMemcpy((void *)&last_scan_element,
                               (void *)(d_array_scan + length - 1),
                               sizeof(T), cudaMemcpyDeviceToHost));
    return last_scan_element;
}

template <typename T>
__global__ void compact_array(uint* d_comp_array, const T* d_occupied,
                               const uint* d_occupied_scan, uint num_grid_points) {
    static_assert(std::is_same_v<T, uint> || std::is_same_v<T, uchar>, "T must be uint or uchar!");
    uint idx = CUDA_THREAD_IDX;

    if (idx >= num_grid_points) return;
    if (d_occupied[idx]) {
        d_comp_array[d_occupied_scan[idx]] = idx;
    }
}

template <typename T>
__global__ void fill(T* array, uint size, T value) {
    uint idx = CUDA_THREAD_IDX;
    if (idx < size) {
        array[idx] = value;
    }
}

#endif //CUDA_COMMON_H

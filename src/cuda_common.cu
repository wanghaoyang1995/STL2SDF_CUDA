//
// Created by wangh on 2025/4/24.
//

#include "helper_math.h"
#include "cuda_common.cuh"

extern "C" void launch_fill(float* d_array, uint size, float value) {
    dim3 blocks_per_grid = cuda_thread_config(size, THREADS_PER_BLOCK);
    fill<float><<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_array, size, value);
}

extern "C" uint launch_scan_sum(uchar* d_array, uint* d_array_scan, uint length) {
    return scan_sum<uchar, uint>(d_array, d_array_scan, length);
}

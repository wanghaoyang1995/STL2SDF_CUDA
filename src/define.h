//
// Created by wangh on 2025/4/9.
//

#ifndef DEFINE_H
#define DEFINE_H

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <vector_types.h>

typedef unsigned int uint;  // 4 bytes
typedef unsigned short ushort;  // 2 bytes
typedef unsigned char uchar;  // 1 byte

#ifdef __CUDACC__
#define HOST_DEVICE_FUNC __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_FUNC
#endif

#define THREADS_PER_BLOCK 128
#define USE_CUDA_2D_GRID 0
#define THREADS_PER_BLOCK_WITH_SHARE_MEM 32
#define FLOAT_TOLERANCE 1e-6

#include <fstream>

struct Vertice {  // 4 bytes * 3
    float x, y, z;
    HOST_DEVICE_FUNC Vertice(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}

    HOST_DEVICE_FUNC Vertice operator+(const Vertice& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    HOST_DEVICE_FUNC Vertice operator-(const Vertice& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    HOST_DEVICE_FUNC float operator*(const Vertice& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    HOST_DEVICE_FUNC Vertice operator*(float f) const {
        return {f * x, f * y, f * z};
    }

    HOST_DEVICE_FUNC Vertice operator/(float f) const {
        return {f / x, f / y, f / z};
    }

    HOST_DEVICE_FUNC friend Vertice operator*(float f, const Vertice& vertice) {
        return {f * vertice.x, f * vertice.y, f * vertice.z};
    }

    HOST_DEVICE_FUNC float mag2() const {
        return x * x + y * y + z * z;
    }

    HOST_DEVICE_FUNC float mag() const {
#ifdef __CUDACC__
        return sqrt(mag2());
#else
        return std::sqrt(mag2());
#endif
    }

    HOST_DEVICE_FUNC static float dist(const Vertice& v1, const Vertice& v2) {
        return (v1 - v2).mag();
    }
};

struct Triangle {
    Vertice vertices[3];
};

struct Grid {
    uint3 grid_size;

    float3 min_bound;
    float3 grid_step;

    uint3 grid_size_mask;
    uint3 grid_size_shift;

    uint num_grid_points;

    HOST_DEVICE_FUNC uint Grid::index_1d(uint i, uint j, uint k) const {
        return i + grid_size.x * j + grid_size.x * grid_size.y * k;
    }

    HOST_DEVICE_FUNC uint3 Grid::index_3d(uint idx) const {
        uint i = idx % grid_size.x;
        uint jk = idx / grid_size.x;
        uint j = jk % grid_size.y;
        uint k = jk / grid_size.y;
        return { i, j, k };
    }

    HOST_DEVICE_FUNC uint3 Grid::index_3d_2(uint i, const uint3& shift, const uint3& mask) const {
        uint3 gridPos;
        gridPos.x = i & mask.x;
        gridPos.y = (i >> shift.y) & mask.y;
        gridPos.z = (i >> shift.z) & mask.z;
        return gridPos;
    }

    HOST_DEVICE_FUNC float3 Grid::coordinate(uint i, uint j, uint k) const {
        float x = min_bound.x + static_cast<float>(i) * grid_step.x;
        float y = min_bound.y + static_cast<float>(j) * grid_step.y;
        float z = min_bound.z + static_cast<float>(k) * grid_step.z;
        return {x, y, z};
    }

    HOST_DEVICE_FUNC float3 Grid::coordinate(uint idx) const {
        uint3 index = index_3d(idx);
        return coordinate(index.x, index.y, index.z);
    }

    HOST_DEVICE_FUNC float3 Grid::coordinate2(uint idx, const uint3& shift, const uint3& mask) const {
        uint3 index = index_3d_2(idx, shift, mask);
        return coordinate(index.x, index.y, index.z);
    }
};

inline dim3 cuda_thread_config(uint voxels_num, uint threads) {
#if USE_CUDA_2D_GRID
    dim3 grid((voxels_num + threads - 1) / threads, 1, 1);
    if (grid.x > 65535) {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }
    return grid;
#else
    return (voxels_num + threads - 1) / threads;
#endif
}

template <typename T>
void dump(const std::vector<T>& field, const Grid& grid, const char* filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        printf("dump error: cannot open file %s.\n", filename);
        return;
    }

    ofs << "# vtk DataFile Version 3.0\nSDF data\nASCII\n";
    ofs << "DATASET STRUCTURED_POINTS\n";
    ofs << "DIMENSIONS " << grid.grid_size.x << " " << grid.grid_size.y << " " << grid.grid_size.z << "\n";
    ofs << "ORIGIN 0.0 0.0 0.0\n";
    ofs << "SPACING " << grid.grid_step.x << " " << grid.grid_step.y << " " << grid.grid_step.z << "\n";
    ofs << "POINT_DATA " << grid.grid_size.x * grid.grid_size.y * grid.grid_size.z << "\n";
    ofs << "SCALARS DistanceField float 1\n";
    ofs << "LOOKUP_TABLE default\n";

    for (float i : field) {
        ofs << static_cast<float>(i) << "\n";
    }

    ofs.close();
}

#endif //DEFINE_H

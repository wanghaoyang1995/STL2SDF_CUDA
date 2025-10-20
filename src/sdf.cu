//
// Created by wangh on 2025/4/8.
//

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "cuda_common.cuh"

__device__ int orientation(float x1, float y1, float x2, float y2, float& twice_signed_area) {
    twice_signed_area = x1 * y2 - x2 * y1;
    if (abs(twice_signed_area - 0) < FLOAT_TOLERANCE) {
        if(y1 > y2) return 1;
        else if(y1 < y2) return -1;
        else if(x1 < x2) return 1;
        else if(x1 > x2) return -1;
        else return 0;
    } else {
        if (twice_signed_area > 0) return 1;
        else return -1;
    }
}

__device__ bool point_in_triangle_2d(float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3, float& lambda_a, float& lambda_b, float& lambda_c) {
    x1 -= x0; x2 -= x0; x3 -= x0;
    y1 -= y0; y2 -= y0; y3 -= y0;

    float min_x = min(min(x1, x2), x3);
    float max_x = max(max(x1, x2), x3);
    float min_y = min(min(y1, y2), y3);
    float max_y = max(max(y1, y2), y3);
    if (max_x < 0 || min_x > 0 || max_y < 0 || min_y > 0) return false;

    int sign1 = orientation(x2, y2, x3, y3, lambda_a);
    if (sign1 == 0) return false;
    int sign2 = orientation(x3, y3, x1, y1, lambda_b);
    if (sign2 != sign1) return false;
    int sign3 = orientation(x1, y1, x2, y2, lambda_c);
    if (sign3 != sign1) return false;

    float sum = lambda_a + lambda_b + lambda_c;
    lambda_a /= sum;
    lambda_b /= sum;
    lambda_c /= sum;

    return true;
}

__device__ float point_segment_distance(Vertice p0, Vertice pa, Vertice pb) {
    Vertice vab = pb - pa;
    Vertice vap = p0 - pa;

    float m2_AB = vab.mag2();
    float lambda = vab * vap / m2_AB;
    if (lambda < 0) lambda = 0;
    else if (lambda > 1) lambda = 1;

    Vertice pc = { pa.x + lambda * vab.x, pa.y + lambda * vab.y, pa.z + lambda * vab.z };
    return Vertice::dist(pc, p0);
}

__device__ float point_triangle_distance(Vertice p0, Vertice p1, Vertice p2, Vertice p3) {
    Vertice v31 = p1 - p3;
    Vertice v32 = p2 - p3;
    Vertice v30 = p0 - p3;
    float m31 = v31.mag2();
    float m32 = v32.mag2();
    float d = v31 * v32;

    float invdet = 1.0f/max(m31 * m32 - d * d, 1e-30f);
    float a = v31 * v30, b = v32 * v30;
    float w23 = invdet * (m32 * a - d * b);
    float w31 = invdet * (m31 * b - d * a);
    float w12 = 1 - w23 - w31;
    if(w23 >= 0 && w31 >= 0 && w12 >= 0){
        return Vertice::dist(p0, w23 * p1 + w31 * p2 + w12 * p3);
    } else {
        if(w23 > 0)
            return min(point_segment_distance(p0, p1, p2), point_segment_distance(p0, p1, p3));
        else if(w31>0)
            return min(point_segment_distance(p0, p1, p2), point_segment_distance(p0, p2, p3));
        else
            return min(point_segment_distance(p0, p1, p3), point_segment_distance(p0, p2, p3));
    }
}

__device__ bool point_in_tri_boundbox(const uint3& p, const Triangle& tri, const Grid* d_grid) {
    float x_min = min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x);
    float y_min = min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y);
    float z_min = min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z);
    float x_max = max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x);
    float y_max = max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y);
    float z_max = max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z);
    uint i_min = floor((x_min - d_grid->min_bound.x) / d_grid->grid_step.x);
    uint j_min = floor((y_min - d_grid->min_bound.y) / d_grid->grid_step.y);
    uint k_min = floor((z_min - d_grid->min_bound.z) / d_grid->grid_step.z);
    uint i_max = ceil((x_max - d_grid->min_bound.x) / d_grid->grid_step.x);
    uint j_max = ceil((y_max - d_grid->min_bound.y) / d_grid->grid_step.y);
    uint k_max = ceil((z_max - d_grid->min_bound.z) / d_grid->grid_step.z);
    return p.x >= i_min && p.x <= i_max &&
           p.y >= j_min && p.y <= j_max &&
           p.z >= k_min && p.z <= k_max;
}

__global__ void ray_intersection_count(
        uchar* d_ray_intersection_count,
        const Grid* d_grid,
        const Triangle* d_triangles,
        uint num_triangles,
        uint num_rays
        ) {
    uint idx = CUDA_THREAD_IDX;
    if (idx >= num_rays) return;

    uint ray_y_index = idx % d_grid->grid_size.y;
    uint ray_z_index = idx / d_grid->grid_size.y;
    float y = d_grid->min_bound.y + static_cast<float>(ray_y_index) * d_grid->grid_step.y;
    float z = d_grid->min_bound.z + static_cast<float>(ray_z_index) * d_grid->grid_step.z;

    for (uint i = 0; i < num_triangles; i++) {
        float a, b, c;
        if (point_in_triangle_2d(y, z,
                                 d_triangles[i].vertices[0].y, d_triangles[i].vertices[0].z,
                                 d_triangles[i].vertices[1].y, d_triangles[i].vertices[1].z,
                                 d_triangles[i].vertices[2].y, d_triangles[i].vertices[2].z,
                                 a, b, c
                                 )) {
            float intersection_x = a * d_triangles[i].vertices[0].x + b * d_triangles[i].vertices[1].x + c * d_triangles[i].vertices[2].x;
            uint intersection_i = static_cast<uint>(ceil((intersection_x - d_grid->min_bound.x) / d_grid->grid_step.x));
            uint count_voxel_index = d_grid->index_1d(intersection_i, ray_y_index, ray_z_index);
            d_ray_intersection_count[count_voxel_index] += 1;
        }
    }
}

__global__ void fill_sign_to_field(float* d_field,
                                   const uchar* d_ray_intersection_count,
                                   const Grid* d_grid,
                                   uint num_rays) {
    uint idx = CUDA_THREAD_IDX;
    if (idx >= num_rays) return;

    uint ray_y_index = idx % d_grid->grid_size.y;
    uint ray_z_index = idx / d_grid->grid_size.y;
    uint count = 0;
    uint base_idx = d_grid->grid_size.x * ray_y_index + d_grid->grid_size.x * d_grid->grid_size.y * ray_z_index;
    for (uint i = 0; i < d_grid->grid_size.x; i++) {
        uint count_voxel_index = base_idx + i;
        count += d_ray_intersection_count[count_voxel_index];
        if (count % 2 == 0) {
            d_field[count_voxel_index] = -d_field[count_voxel_index];
        }
    }
}

__global__ void compute_sdf_brute(
        float* d_field,
        const Grid* d_grid,
        const Triangle* d_triangles,
        uint num_triangles
) {
    uint idx = CUDA_THREAD_IDX;

    uint grid_size = d_grid->num_grid_points;
    if (idx >= grid_size || d_field[idx] < 0) return;
    float3 coord = d_grid->coordinate2(idx, d_grid->grid_size_shift, d_grid->grid_size_mask);
    Vertice p0 = { coord.x, coord.y, coord.z };
    for (int i = 0; i < num_triangles; i++) {
        float d = point_triangle_distance(p0, d_triangles[i].vertices[0], d_triangles[i].vertices[1], d_triangles[i].vertices[2]);
        if (d < d_field[idx]) d_field[idx] = d;
    }
}

__global__ void compute_sdf_jfa(
        float* d_field,
        const Grid* d_grid,
        const int* d_closest_tri,
        int* d_closest_tri_output,
        const Triangle* d_triangles,
        int step
) {
    uint idx = CUDA_THREAD_IDX;
    if (idx >= d_grid->num_grid_points) return;

    uint3 ijk = d_grid->index_3d_2(idx, d_grid->grid_size_shift, d_grid->grid_size_mask);
    float3 coord = d_grid->coordinate(ijk.x, ijk.y, ijk.z);
    Vertice p = { coord.x, coord.y, coord.z };

    int pos[] = { -step, 0, step };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int i_neighbour = static_cast<int>(ijk.x) + pos[i];
                int j_neighbour = static_cast<int>(ijk.y) + pos[j];
                int k_neighbour = static_cast<int>(ijk.z) + pos[k];
                if (i_neighbour < 0 || i_neighbour >= d_grid->grid_size.x ||
                    j_neighbour < 0 || j_neighbour >= d_grid->grid_size.y ||
                    k_neighbour < 0 || k_neighbour >= d_grid->grid_size.z
                ) continue;

                uint idx_neighbour = d_grid->index_1d(i_neighbour, j_neighbour, k_neighbour);
                int tri_idx = d_closest_tri[idx_neighbour];
                if (tri_idx == -1) continue;

                float distance = point_triangle_distance(p,
                                                         d_triangles[tri_idx].vertices[0],
                                                         d_triangles[tri_idx].vertices[1],
                                                         d_triangles[tri_idx].vertices[2]);
                if (distance < d_field[idx]) {
                    d_field[idx] = distance;
                    d_closest_tri_output[idx] = tri_idx;
                }
            }
        }
    }
}

__global__ void classify_seeds_points(uint* d_seeds, const Grid* d_grid, const Triangle* d_triangles, uint num_triangles) {
    uint idx = CUDA_THREAD_IDX;
    if (idx >= num_triangles) return;
    const Triangle& tri = d_triangles[idx];
    float x_min = min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x);
    float y_min = min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y);
    float z_min = min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z);
    float x_max = max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x);
    float y_max = max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y);
    float z_max = max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z);
    uint i_min = floor((x_min - d_grid->min_bound.x) / d_grid->grid_step.x);
    uint j_min = floor((y_min - d_grid->min_bound.y) / d_grid->grid_step.y);
    uint k_min = floor((z_min - d_grid->min_bound.z) / d_grid->grid_step.z);
    uint i_max = ceil((x_max - d_grid->min_bound.x) / d_grid->grid_step.x);
    uint j_max = ceil((y_max - d_grid->min_bound.y) / d_grid->grid_step.y);
    uint k_max = ceil((z_max - d_grid->min_bound.z) / d_grid->grid_step.z);

    for (uint i = i_min; i <= i_max; i++) {
        for (uint j = j_min; j <= j_max; j++) {
            for (uint k = k_min; k <= k_max; k++) {
                uint pindex = d_grid->index_1d(i, j, k);
                atomicMax(&d_seeds[pindex], 1);
            }
        }
    }
}

__global__ void fill_cloest_tri(int* d_cloest_tri, float* d_field, uint* d_seeds_comp, const Grid* d_grid, const Triangle* d_triangles, uint num_triangles, uint total_seeds) {
    uint idx = CUDA_THREAD_IDX;
    if (idx >= total_seeds) return;
    idx = d_seeds_comp[idx];

    uint3 ijk = d_grid->index_3d_2(idx, d_grid->grid_size_shift, d_grid->grid_size_mask);
    float3 coord = d_grid->coordinate(ijk.x, ijk.y, ijk.z);
    Vertice p0 = { coord.x, coord.y, coord.z };
    for (int i = 0; i < num_triangles; i++) {
        if (point_in_tri_boundbox(ijk, d_triangles[i], d_grid)) {
            float d = point_triangle_distance(p0, d_triangles[i].vertices[0], d_triangles[i].vertices[1], d_triangles[i].vertices[2]);
            if (d < d_field[idx]) {
                d_cloest_tri[idx] = i;
                d_field[idx] = d;
            }
        }
    }
}

extern "C" void compute_signed_field(float* d_field, const Grid* d_grid, uchar* d_ray_intersection_count, const Triangle* d_triangles, uint num_triangles, const Grid& grid) {
    uint num_rays = grid.grid_size.y * grid.grid_size.z;
    dim3 blocks_per_grid = cuda_thread_config(num_rays, THREADS_PER_BLOCK);

    ray_intersection_count<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_ray_intersection_count, d_grid, d_triangles, num_triangles,
                                                                   num_rays);
    fill_sign_to_field<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_field, d_ray_intersection_count, d_grid, num_rays);
}

extern "C" void make_sdf_brute(float* d_field, const Grid* d_grid, const Triangle* d_triangles, uint num_triangles, uint num_grid_points) {
    uint threads_per_block = THREADS_PER_BLOCK;
    dim3 blocks_per_grid = cuda_thread_config(num_grid_points, THREADS_PER_BLOCK);
    compute_sdf_brute<<<blocks_per_grid, threads_per_block>>>(d_field, d_grid, d_triangles, num_triangles);
}

extern "C" void make_mdf_jfa(float* d_field, const Grid* d_grid, int* d_closest_tri, int* d_closest_tri_output, uint* d_seeds, uint* d_seeds_scan, uint* d_seeds_comp, const Triangle* d_triangles, uint num_triangles, const Grid& field_grid) {
    dim3 blocks_per_grid_for_tris = cuda_thread_config(num_triangles, THREADS_PER_BLOCK);
    classify_seeds_points<<<blocks_per_grid_for_tris, THREADS_PER_BLOCK>>>(d_seeds, d_grid, d_triangles, num_triangles);

    dim3 blocks_per_grid_for_grid_points = cuda_thread_config(field_grid.num_grid_points, THREADS_PER_BLOCK);
    thrust_exclusive_scan<uint, uint>(d_seeds_scan, d_seeds, field_grid.num_grid_points);
    compact_array<uint><<<blocks_per_grid_for_grid_points, THREADS_PER_BLOCK>>>(d_seeds_comp, d_seeds, d_seeds_scan, field_grid.num_grid_points);
    uint total_seeds = scan_sum<uint, uint>(d_seeds, d_seeds_scan, field_grid.num_grid_points);

    dim3 blocks_per_grid_for_seeds = cuda_thread_config(total_seeds, THREADS_PER_BLOCK);
    fill_cloest_tri<<<blocks_per_grid_for_seeds, THREADS_PER_BLOCK>>>(d_closest_tri, d_field, d_seeds_comp, d_grid, d_triangles, num_triangles, total_seeds);

    thrust::copy(thrust::device_ptr<int>(d_closest_tri), thrust::device_ptr<int>(d_closest_tri + field_grid.num_grid_points), thrust::device_ptr<int>(d_closest_tri_output));

    int step = static_cast<int>(field_grid.grid_size.y / 2);
    while (step >= 1) {
        compute_sdf_jfa<<<blocks_per_grid_for_grid_points, THREADS_PER_BLOCK>>>(d_field, d_grid, d_closest_tri, d_closest_tri_output, d_triangles, step);
        thrust::copy(thrust::device_ptr<int>(d_closest_tri_output), thrust::device_ptr<int>(d_closest_tri_output + field_grid.num_grid_points), thrust::device_ptr<int>(d_closest_tri));
        step /= 2;
    }
}

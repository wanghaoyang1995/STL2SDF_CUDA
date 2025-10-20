//
// Created by wangh on 2025/4/8.
//

#include <cstdlib>
#include <vector_functions.h>

#include "helper_cuda.h"
#include "helper_timer.h"
#include "define.h"
#include "stl.h"

const char* app_name = "STL2SDF_CUDA";

// device memory
float* d_field;
Triangle* d_triangles;
Grid* d_grid;
uchar* d_ray_intersection_count;
uint* d_seeds;
uint* d_seeds_scan;
uint* d_seeds_comp;
int* d_cloest_tri;
int* d_closest_tri_output;

// host memory
std::vector<float> field;
std::vector<Triangle> triangles;
Grid field_grid;
uint jfa;
uint sign;

StopWatchInterface* timer;

extern "C" void make_sdf_brute(float* d_field, const Grid* d_grid, const Triangle* d_triangles, uint num_triangles, uint num_grid_points);
extern "C" void make_mdf_jfa(float* d_field, const Grid* d_grid, int* d_closest_tri, int* d_closest_tri_output, uint* d_seeds, uint* d_seeds_scan, uint* d_seeds_comp, const Triangle* d_triangles, uint num_triangles, const Grid& field_grid);
extern "C" void compute_signed_field(float* d_field, const Grid* d_grid, uchar* d_ray_intersection_count, const Triangle* d_triangles, uint num_triangles, const Grid& grid);
extern "C" void launch_fill(float* d_array, uint size, float value);

void create_volume_texture(float* d_field_, uint buff_size, cudaTextureObject_t& texture) {
    cudaResourceDesc tex_res;
    memset(&tex_res, 0, sizeof(cudaResourceDesc));

    tex_res.resType = cudaResourceTypeLinear;
    tex_res.res.linear.devPtr = d_field_;
    tex_res.res.linear.sizeInBytes = buff_size;
    tex_res.res.linear.desc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaTextureDesc tex_descr;
    memset(&tex_descr, 0, sizeof(cudaTextureDesc));

    tex_descr.normalizedCoords = false;
    tex_descr.filterMode = cudaFilterModePoint;
    tex_descr.addressMode[0] = cudaAddressModeClamp;
    tex_descr.readMode = cudaReadModeElementType;

    checkCudaErrors(
            cudaCreateTextureObject(&texture, &tex_res, &tex_descr, NULL));
}

void init_env() {
    cudaMalloc((void**)&d_field, field_grid.num_grid_points * sizeof(float));
    cudaMalloc((void**)&d_triangles, triangles.size() * sizeof(Triangle));
    cudaMalloc((void**)&d_grid, sizeof(Grid));
    cudaMalloc((void**)&d_ray_intersection_count, field_grid.num_grid_points * sizeof(uchar));
    cudaMalloc((void**)&d_cloest_tri, field_grid.num_grid_points * sizeof(int));
    cudaMalloc((void**)&d_closest_tri_output, field_grid.num_grid_points * sizeof(int));
    cudaMalloc((void**)&d_seeds, field_grid.num_grid_points * sizeof(uint));
    cudaMalloc((void**)&d_seeds_scan, field_grid.num_grid_points * sizeof(uint));
    cudaMalloc((void**)&d_seeds_comp, field_grid.num_grid_points * sizeof(uint));

    launch_fill(d_field, field_grid.num_grid_points, FLT_MAX);
    cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, &field_grid, sizeof(Grid), cudaMemcpyHostToDevice);
    cudaMemset(d_ray_intersection_count, 0, field_grid.num_grid_points);
    cudaMemset(d_cloest_tri, -1, field_grid.num_grid_points * sizeof(int));
    cudaMemset(d_closest_tri_output, -1, field_grid.num_grid_points * sizeof(int));
    cudaMemset(d_seeds, 0, field_grid.num_grid_points * sizeof(uint));
}

void clean_memory() {
    checkCudaErrors(cudaFree(d_field));
    checkCudaErrors(cudaFree(d_triangles));
    checkCudaErrors(cudaFree(d_grid));
    checkCudaErrors(cudaFree(d_ray_intersection_count));

    checkCudaErrors(cudaFree(d_cloest_tri));
    checkCudaErrors(cudaFree(d_closest_tri_output));
    checkCudaErrors(cudaFree(d_seeds));
    checkCudaErrors(cudaFree(d_seeds_scan));
    checkCudaErrors(cudaFree(d_seeds_comp));
}

void run(const char* stl_file, const char* output_file) {
    sdkCreateTimer(&timer);

    Vertice min_bound, max_bound;
    float last_time = 0.0f;
    timer->reset();
    timer->start();

#if defined(USE_BINARY_STL)
    STL::read_binary_stl(stl_file, triangles, min_bound, max_bound);
#else
    STL::read_ascii_stl(stl_file, triangles, min_bound, max_bound);
#endif

    printf("Loading STL: %f ms.\n", timer->getTime());

    field_grid.min_bound = make_float3(min_bound.x - 0.1f, min_bound.y - 0.1f, min_bound.z - 0.1f);
    max_bound = Vertice(max_bound.x + 0.1f, max_bound.y + 0.1f, max_bound.z + 0.1f);
    field_grid.grid_step = make_float3(
            (max_bound.x - field_grid.min_bound.x) / (float)(field_grid.grid_size.x - 1),
            (max_bound.y - field_grid.min_bound.y) / (float)(field_grid.grid_size.y - 1),
            (max_bound.z - field_grid.min_bound.z) / (float)(field_grid.grid_size.z - 1)
    );

    init_env();

    last_time = timer->getTime();
    if (jfa == 0) {
        make_sdf_brute(d_field, d_grid, d_triangles, triangles.size(), field_grid.num_grid_points);
        cudaDeviceSynchronize();
        printf("make_mdf_brute: %f ms.\n", timer->getTime() - last_time);
    } else {
        make_mdf_jfa(d_field, d_grid, d_cloest_tri, d_closest_tri_output, d_seeds, d_seeds_scan, d_seeds_comp,
            d_triangles, triangles.size(), field_grid);
        cudaDeviceSynchronize();
        printf("make_mdf_jfa: %f ms.\n", timer->getTime() - last_time);
    }

    if (sign != 0) {
        last_time = timer->getTime();
        compute_signed_field(d_field, d_grid, d_ray_intersection_count, d_triangles, triangles.size(),
                             field_grid);
        cudaDeviceSynchronize();
        printf("compute_signed_field: %f ms.\n", timer->getTime() - last_time);
    }

    timer->stop();
    printf("Total time: %f ms.\n", timer->getTime());

    field.resize(field_grid.num_grid_points);
    cudaMemcpy(field.data(), d_field, field_grid.num_grid_points * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Writing output to %s\n", output_file);
    dump(field, field_grid, output_file);

    clean_memory();
    sdkDeleteTimer(&timer);
}


int main(int argc, char** argv) {
    if (argc < 7) {
        printf("Usage:\n\tstl2sdf_cuda <object_stl> <log2_size_x> <log2_size_y> <log2_size_z> <jfa> <signed>\n");
        return 0;
    }
    printf("%s\n", app_name);

    char* endptr;
    char* stl_file = argv[1];
    uint log2_size_x = std::strtol(argv[2], &endptr, 10);
    uint log2_size_y = std::strtol(argv[3], &endptr, 10);
    uint log2_size_z = std::strtol(argv[4], &endptr, 10);
    jfa = std::strtol(argv[5], &endptr, 10);
    sign = std::strtol(argv[6], &endptr, 10);
    char* output_file = argv[7];

    field_grid.grid_size = make_uint3(1 << log2_size_x, 1 << log2_size_y, 1 << log2_size_z);
    field_grid.grid_size_mask = make_uint3(field_grid.grid_size.x - 1, field_grid.grid_size.y - 1, field_grid.grid_size.z - 1);
    field_grid.grid_size_shift = make_uint3(0, log2_size_x, log2_size_x + log2_size_y);
    field_grid.num_grid_points = field_grid.grid_size.x * field_grid.grid_size.y * field_grid.grid_size.z;

    run(stl_file, output_file);

    return 0;
}

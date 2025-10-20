### Signed Distance Field Computation for STL Based on Parallel JFA Algorithm

```bash
Usage:
stl2sdf_cuda <object_stl> <log2_size_x> <log2_size_y> <log2_size_z> <jfa> <signed>

Arguments:
<object_stl>              Path to the input STL file (triangular mesh model).

<log2_size_x>             Base-2 logarithm of the grid resolution in X dimension.
                            The actual number of grid points along X is 2^log2_size_x.

<log2_size_y>             Base-2 logarithm of the grid resolution in Y dimension.
                            The actual number of grid points along Y is 2^log2_size_y.

<log2_size_z>             Base-2 logarithm of the grid resolution in Z dimension.
                            The actual number of grid points along Z is 2^log2_size_z.

<jfa>                     Algorithm selection flag for distance field computation:
                            - 0: Use brute-force method (compute distances from each voxel to all triangles).
                            - 1 (or any non-zero): Use Jump Flooding Algorithm (JFA) for faster SDF computation.

<signed>                  Signed field flag:
                            - 0: Compute unsigned distance field (all values are non-negative).
                            - 1 (or any non-zero): Compute signed distance field (negative outside, positive inside, zero at surface), the input STL model is required to be **closed**.

<output_file>             Path to the output VTK file.
```

Example:
```bash
stl2sdf_cuda_ascii ./data/model1_ascii.stl 8 7 7 1 1 ./field1.vtk
stl2sdf_cuda_binary ./data/model2_binary.stl 7 7 7 1 0 ./field2.vtk
```

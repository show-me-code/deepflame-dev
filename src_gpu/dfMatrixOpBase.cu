#include "dfMatrixOpBase.H"
#include "dfMatrixDataBase.H"

#include <cuda_runtime.h>
#include "cuda_profiler_api.h"

__global__ void permute_vector_d2h_kernel(int num_cells, const double *input, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    output[index * 3 + 0] = input[num_cells * 0 + index];
    output[index * 3 + 1] = input[num_cells * 1 + index];
    output[index * 3 + 2] = input[num_cells * 2 + index];
}

__global__ void permute_vector_h2d_kernel(int num_cells, const double *input, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    output[num_cells * 0 + index] = input[index * 3 + 0];
    output[num_cells * 1 + index] = input[index * 3 + 1];
    output[num_cells * 2 + index] = input[index * 3 + 2];
}

__global__ void update_boundary_coeffs_zeroGradient_vector(int num, int offset,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    // valueInternalCoeffs = 1
    // valueBoundaryCoeffs = 0
    // gradientInternalCoeffs = 0
    // gradientBoundaryCoeffs = 0
    value_internal_coeffs[start_index * 3 + 0] = 1;
    value_internal_coeffs[start_index * 3 + 1] = 1;
    value_internal_coeffs[start_index * 3 + 2] = 1;
    value_boundary_coeffs[start_index * 3 + 0] = 0;
    value_boundary_coeffs[start_index * 3 + 1] = 0;
    value_boundary_coeffs[start_index * 3 + 2] = 0;
    gradient_internal_coeffs[start_index * 3 + 0] = 0;
    gradient_internal_coeffs[start_index * 3 + 1] = 0;
    gradient_internal_coeffs[start_index * 3 + 2] = 0;
    gradient_boundary_coeffs[start_index * 3 + 0] = 0;
    gradient_boundary_coeffs[start_index * 3 + 1] = 0;
    gradient_boundary_coeffs[start_index * 3 + 2] = 0;
}

__global__ void fvm_ddt_vector_kernel(int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    diag[index] += rDeltaT * rho[index] * volume[index];
    // TODO: skip moving
    source[index * 3 + 0] += rDeltaT * rho_old[index] * vf[index * 3 + 0] * volume[index];
    source[index * 3 + 1] += rDeltaT * rho_old[index] * vf[index * 3 + 1] * volume[index];
    source[index * 3 + 2] += rDeltaT * rho_old[index] * vf[index * 3 + 2] * volume[index];
}

__global__ void fvm_div_vector_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double f = phi[index];

    double lower_value = (-w) * f;
    double upper_value = (1 - w) * f;
    lower[index] += lower_value;
    upper[index] += upper_value;
    // if (index == 0) printf("index = 0, lower: %.16lf, upper:%.16lf\n", lower[index], upper[index]);

    int owner = lower_index[index];
    int neighbor = upper_index[index];
    atomicAdd(&(diag[owner]), -lower_value);
    atomicAdd(&(diag[neighbor]), -upper_value);
}

__global__ void fvm_div_vector_boundary(int num, int offset,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    double boundary_f = boundary_phi[start_index];
    internal_coeffs[start_index * 3 + 0] += boundary_f * value_internal_coeffs[start_index * 3 + 0];
    internal_coeffs[start_index * 3 + 1] += boundary_f * value_internal_coeffs[start_index * 3 + 1];
    internal_coeffs[start_index * 3 + 2] += boundary_f * value_internal_coeffs[start_index * 3 + 2];
    boundary_coeffs[start_index * 3 + 0] += boundary_f * value_boundary_coeffs[start_index * 3 + 0];
    boundary_coeffs[start_index * 3 + 1] += boundary_f * value_boundary_coeffs[start_index * 3 + 1];
    boundary_coeffs[start_index * 3 + 2] += boundary_f * value_boundary_coeffs[start_index * 3 + 2];
}

__global__ void fvm_laplacian_vector_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double w = weight[index];
    double upper_face_gamma = w * gamma[owner] + (1 - w) * gamma[neighbor];
    double upper_value = upper_face_gamma * mag_sf[index] * delta_coeffs[index];

    // laplacian doesn't use the original lower, but use lower = upper
    //double lower_face_gamma = w * gamma[neighbor] + (1 - w) * gamma[owner];
    //double lower_value = lower_face_gamma * mag_sf[index] * delta_coeffs[index];
    double lower_value = upper_value;

    lower[index] += lower_value;
    upper[index] += upper_value;

    atomicAdd(&(diag[owner]), -lower_value);
    atomicAdd(&(diag[neighbor]), -upper_value);
}

__global__ void fvm_laplacian_vector_boundary(int num, int offset,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    double boundary_value = boundary_gamma[start_index] * boundary_mag_sf[start_index];
    internal_coeffs[start_index * 3 + 0] += boundary_value * gradient_internal_coeffs[start_index * 3 + 0];
    internal_coeffs[start_index * 3 + 1] += boundary_value * gradient_internal_coeffs[start_index * 3 + 1];
    internal_coeffs[start_index * 3 + 2] += boundary_value * gradient_internal_coeffs[start_index * 3 + 2];
    boundary_coeffs[start_index * 3 + 0] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 0];
    boundary_coeffs[start_index * 3 + 1] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 1];
    boundary_coeffs[start_index * 3 + 2] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 2];
}

void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    permute_vector_d2h_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, input, output);
}

void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    permute_vector_h2d_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, input, output);
}

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lower_to_csr_index, const int *upper_to_csr_index, const int *diag_to_csr_index,
        const double *lower, const double *upper, const double *diag, const double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A, double *b)
{

}

void update_boundary_coeffs_vector(cudaStream_t stream, int num_patches,
        const int *patch_size, const int *patch_type,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        // TODO: just vector version now
        if (patch_type[i] == boundaryConditions::zeroGradient) {
            update_boundary_coeffs_zeroGradient_vector<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    value_internal_coeffs, value_boundary_coeffs, gradient_internal_coeffs, gradient_boundary_coeffs);
        } else if (patch_type[i] == boundaryConditions::fixedValue) {
            // xxx
        } else if (0) {
            // xxx
        }
        offset += patch_size[i];
    }
}

void fvm_ddt_vector(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_ddt_vector_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            rDeltaT, rho, rho_old, vf, volume, diag, source);
}

void fvm_div_vector(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvm_div_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr,
            phi, weight, lower, upper, diag);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvm_div_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_phi, value_internal_coeffs, value_boundary_coeffs,
                    internal_coeffs, boundary_coeffs);
        } else if (0) {
            // xxx
        }
        offset += patch_size[i];
    }
}

void fvm_laplacian_vector(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, // TODO: num_boundary_surfaces may not be in use
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr,
            weight, mag_sf, delta_coeffs, gamma, lower, upper, diag);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvm_laplacian_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_mag_sf, boundary_gamma, gradient_internal_coeffs, gradient_boundary_coeffs,
                    internal_coeffs, boundary_coeffs);
        } else if (0) {
            // xxx
        }
        offset += patch_size[i];
    }
}


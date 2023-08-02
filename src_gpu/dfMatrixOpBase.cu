#include "dfMatrixOpBase.H"
#include "dfMatrixDataBase.H"

#include <cuda_runtime.h>
#include "cuda_profiler_api.h"

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

__global__ void fvm_div_scalar_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double f = phi[index];

    lower[index] += (-w) * f;
    upper[index] += (1 - w) * f;

    int l = lower_index[index];
    int u = upper_index[index];
    atomicAdd(&(diag[l]), w * f);
    atomicAdd(&(diag[u]), (w - 1) * f);
}

__global__ void fvm_div_scalar_boundary(int num, int offset,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    double boundary_f = boundary_phi[start_index];
    internal_coeffs[start_index * 3 + 0] = boundary_f * value_internal_coeffs[start_index * 3 + 0];
    internal_coeffs[start_index * 3 + 1] = boundary_f * value_internal_coeffs[start_index * 3 + 1];
    internal_coeffs[start_index * 3 + 2] = boundary_f * value_internal_coeffs[start_index * 3 + 2];
    boundary_coeffs[start_index * 3 + 0] = boundary_f * value_boundary_coeffs[start_index * 3 + 0];
    boundary_coeffs[start_index * 3 + 1] = boundary_f * value_boundary_coeffs[start_index * 3 + 1];
    boundary_coeffs[start_index * 3 + 2] = boundary_f * value_boundary_coeffs[start_index * 3 + 2];
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

void fvm_div_scalar(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvm_div_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces,
            lowerAddr, upperAddr,
            phi, weight, lower, upper, diag);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvm_div_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_phi, value_internal_coeffs, value_boundary_coeffs,
                    internal_coeffs, boundary_coeffs);
        } else if (0) {
            // xxx
        }
        offset += patch_size[i];
    }
}


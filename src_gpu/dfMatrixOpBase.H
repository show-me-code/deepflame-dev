#pragma once

void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output);
void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output);

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lower_to_csr_index, const int *upper_to_csr_index, const int *diag_to_csr_index, 
        const double *lower, const double *upper, const double *diag, const double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A, double *b);

void update_boundary_coeffs_vector(cudaStream_t stream, int num_patches,
        const int *patch_size, const int *patch_type,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs);

// void fvm_ddt();

void fvm_div_scalar(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs);

// void fvm_laplacian();
// 
// void fvc_ddt();
// 
// void fvc_grad_surface();
// 
// void fvc_div_cell();


#pragma once
// #define TIME_GPU

// tools
void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output);
void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output);

void field_multiply_scalar(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output);

void fvc_to_source_vector(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source);

void ldu_to_csr_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface,
        const int* boundary_cell_face, const int *lower_to_csr_index, const int *upper_to_csr_index, const int *diag_to_csr_index,
        const double *lower, const double *upper, double *diag, double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A);

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface,
        const int* boundary_cell_face, const int *lower_to_csr_index, const int *upper_to_csr_index, const int *diag_to_csr_index,
        const double *lower, const double *upper, const double *diag, const double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A, double *b, double *diag_vec);

void update_boundary_coeffs_scalar(cudaStream_t stream, int num_patches,
        const int *patch_size, const int *patch_type,
        const double *boundary_delta_coeffs, const double *boundary_vf,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs);

void update_boundary_coeffs_vector(cudaStream_t stream, int num_boundary_surfaces, int num_patches,
        const int *patch_size, const int *patch_type,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs);

void compute_upwind_weight(cudaStream_t stream, int num_surfaces, const double *phi, double *weight);

// fvm ops

void fvm_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign);

void fvm_ddt_vector(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign = 1.);

void fvm_div_scalar(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_div_vector(cudaStream_t stream, int num_surfaces, int num_boundary_sourfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_vector(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

// fvc ops
// fvc_ddt doesn't consider to add fvc_output to source yet, which needs (fvc_output * volume * sign).
void fvc_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *vf_old,
        double *output, double sign = 1.);

void fvc_grad_vector(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const double *boundary_deltaCoeffs, double sign = 1.);

void fvc_div_surface_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, const double *ssf, const int *boundary_cell_face,
        const double *boundary_ssf, const double *volume, double *output, double sign = 1.);

void fvc_div_cell_vector(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_div_cell_tensor(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_grad_cell_scalar(cudaStream_t stream, int num_cells, int num_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *volume, double sign = 1.);

void fvc_grad_cell_scalar_withBC(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const double *boundary_deltaCoeffs, double sign = 1.);

void fvc_laplacian_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
        const double *gamma, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face,
        const double *boundary_mag_sf, const double *boundary_delta_coeffs,
        const double *boundary_gamma, const double *boundary_vf, double sign = 1.);

// others
void scale_dev2T_tensor(cudaStream_t stream, int num_cells, const double *vf1, double *vf2,
        int num_boundary_surfaces, const double *boundary_vf1, double *boundary_vf2);

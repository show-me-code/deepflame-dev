#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
// #define TIME_GPU
#define DEBUG_CHECK_LDU

// tools
void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output);
void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output);

void field_add_scalar(cudaStream_t stream,
        int num, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output);

void field_add_vector(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output, double sign = 1.);

void field_multiply_scalar(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output);

void scalar_field_multiply_vector_field(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output, double sign = 1.);

void fvc_to_source_vector(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source);

void fvc_to_source_scalar(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source, double sign = 1.);

void ldu_to_csr_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface,
        const int* boundary_cell_face, const int *ldu_to_csr_index, const int *diag_to_csr_index,
        double *ldu, double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A);

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface,
        const int* boundary_cell_face, const int *ldu_to_csr_index, const int *diag_to_csr_index,
        const double *ldu, const double *internal_coeffs, const double *boundary_coeffs, double *source, double *A);

void update_boundary_coeffs_scalar(cudaStream_t stream,
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_delta_coeffs, const double *boundary_vf, const double *boundary_weight, 
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs, const double *energy_gradient = nullptr);

void update_boundary_coeffs_vector(cudaStream_t stream, int num_boundary_surfaces, int num_patches,
        const int *patch_size, const int *patch_type, const double *boundary_vf, 
        const double *boundary_deltaCoeffs, const double *boundary_weight,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs);

void correct_boundary_conditions_scalar(cudaStream_t stream, ncclComm_t comm,
        const int *peer, int num_boundary_surfaces, int num_patches,
        const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *vf, double *boundary_vf);

void correct_boundary_conditions_vector(cudaStream_t stream, ncclComm_t comm,
        const int *neighbor_peer, int num_boundary_surfaces, int num_cells, int num_patches,
        const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *vf, double *boundary_vf);

void compute_upwind_weight(cudaStream_t stream, int num_surfaces, const double *phi, double *weight);

// fvm ops

void fvm_ddt_vol_scalar_vol_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign = 1.);

void fvm_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT, 
        const double *vf_old, const double *volume, 
        double *diag, double *source, double sign = 1.);

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

void fvm_laplacian_scalar(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_vector(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_surface_scalar_vol_scalar(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

// fvc ops
// fvc_ddt doesn't consider to add fvc_output to source yet, which needs (fvc_output * volume * sign).
void fvc_ddt_vol_scalar_vol_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *vf_old, const double *volume, 
        double *output, double sign = 1.);

void fvc_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *vf, const double *vf_old, const double *volume, double *source, double sign);

void fvc_ddt_scalar_field(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *vf, const double *vf_old, const double *volume, double *source, double sign = 1.);

void fvc_grad_vector(cudaStream_t stream, ncclComm_t comm, 
        int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *neighbor_peer, const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *boundary_weight, 
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const double *boundary_deltaCoeffs, double sign = 1.);

void fvc_div_surface_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, const double *ssf, const int *boundary_cell_face,
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_ssf, const double *volume, double *output, double sign = 1.);

void fvc_div_cell_vector(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_div_cell_tensor(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight, 
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_div_surface_scalar_vol_scalar(cudaStream_t stream, int num_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *vf, const double *ssf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_ssf, 
        double sign = 1.);

void fvc_grad_cell_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *volume, double sign = 1.);

void fvc_grad_cell_scalar_withBC(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const double *boundary_deltaCoeffs);

void fvc_laplacian_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
        const double *gamma, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face,
        const double *boundary_mag_sf, const double *boundary_delta_coeffs,
        const double *boundary_gamma, const double *boundary_vf, double sign = 1.);

void fvc_flux(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, 
        double *boundary_output, double sign);

void fvc_interpolate(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_vf, double *boundary_output, double sign);

// others
void scale_dev2T_tensor(cudaStream_t stream, int num_cells, const double *vf1, double *vf2,
        int num_boundary_surfaces, const double *boundary_vf1, double *boundary_vf2);

void fvMtx_A(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *boundary_cell_face, const double *internal_coeffs, const double *volume, const double *diag, 
        double *A_pEqn);

void fvMtx_H(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *volume,
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *internal_coffs, const double *boundary_coeffs, 
        const double *lower, const double *upper, const double *source, const double *psi, 
        double *H_pEqn, double *H_pEqn_perm);

void fvMtx_flux(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *lower, const double *upper,
        const double *psi, double *output, //end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *internal_coeffs, const double *boundary_coeffs, 
        const double *boundary_psi, double *boundary_output);

void solve_explicit_scalar(cudaStream_t stream, int num_cells, const double *diag, const double *source,
        double *psi);

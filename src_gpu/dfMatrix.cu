#include "dfMatrix.H"

void checkVectorEqual(int count, const double* basevec, double* vec, double max_relative_error) {
    for (size_t i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        if (abs_diff > 1e-16 && rel_diff > max_relative_error && !std::isinf(rel_diff))
            fprintf(stderr, "mismatch index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }
}

// kernel functions
__global__ void fvm_ddt_kernel(int num_cells, int num_faces, const double rdelta_t,
        const int* csr_row_index, const int* csr_diag_index,
        const double* rho_old, const double* rho_new, const double* volume, const double* velocity_old,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output, double* psi) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int diag_index = csr_diag_index[index];

    int csr_dim = num_cells + num_faces;
    int csr_index = row_index + diag_index;
    double ddt_diag = rdelta_t * rho_new[index] * volume[index];
    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + ddt_diag;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + ddt_diag;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + ddt_diag;

    double ddt_part_term = rdelta_t * rho_old[index] * volume[index];
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + ddt_part_term * velocity_old[index * 3 + 0];
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + ddt_part_term * velocity_old[index * 3 + 1];
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + ddt_part_term * velocity_old[index * 3 + 2];

    psi[num_cells * 0 + index] = velocity_old[index * 3 + 0];
    psi[num_cells * 1 + index] = velocity_old[index * 3 + 1];
    psi[num_cells * 2 + index] = velocity_old[index * 3 + 2];
}

__global__ void fvm_div_internal(int num_cells, int num_faces,
        const int* csr_row_index, const int* csr_diag_index,
        const double* weight, const double* phi,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;

    double div_diag = 0;
    for (int i = row_index; i < next_row_index; i++) {
      int inner_index = i - row_index;
      // lower
      if (inner_index < diag_index) {
        int neighbor_index = neighbor_offset + inner_index;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + (-w) * f;
        A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + (-w) * f;
        A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + (-w) * f;
        // lower neighbors contribute to sum of -1
        div_diag += (w - 1) * f;
      }
      // upper
      if (inner_index > diag_index) {
        // upper, index - 1, consider of diag
        int neighbor_index = neighbor_offset + inner_index - 1;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + (1 - w) * f;
        A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + (1 - w) * f;
        A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + (1 - w) * f;
        // upper neighbors contribute to sum of 1
        div_diag += w * f;
      }
    }
    A_csr_output[csr_dim * 0 + row_index + diag_index] = A_csr_input[csr_dim * 0 + row_index + diag_index] + div_diag; // diag
    A_csr_output[csr_dim * 1 + row_index + diag_index] = A_csr_input[csr_dim * 1 + row_index + diag_index] + div_diag; // diag
    A_csr_output[csr_dim * 2 + row_index + diag_index] = A_csr_input[csr_dim * 2 + row_index + diag_index] + div_diag; // diag
}

__global__ void fvm_div_boundary(int num_cells, int num_faces, int num_boundary_cells,
        const int* csr_row_index, const int* csr_diag_index,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* internal_coeffs, const double* boundary_coeffs,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int cell_index = boundary_cell_id[cell_offset];
    int loop_size = boundary_cell_offset[index + 1] - cell_offset;

    int row_index = csr_row_index[cell_index];
    int diag_index = csr_diag_index[cell_index];
    int csr_dim = num_cells + num_faces;
    int csr_index = row_index + diag_index;
    if (index == 24570)
    {
        printf("csr_index = %d\n", csr_index);
    }
    // construct internalCoeffs & boundaryCoeffs
    double internal_coeffs_x = 0;
    double internal_coeffs_y = 0;
    double internal_coeffs_z = 0;
    double boundary_coeffs_x = 0;
    double boundary_coeffs_y = 0;
    double boundary_coeffs_z = 0;
    for (int i = 0; i < loop_size; i++) {
        internal_coeffs_x += internal_coeffs[(cell_offset + i) * 3 + 0];
        internal_coeffs_y += internal_coeffs[(cell_offset + i) * 3 + 1];
        internal_coeffs_z += internal_coeffs[(cell_offset + i) * 3 + 2];
        boundary_coeffs_x += boundary_coeffs[(cell_offset + i) * 3 + 0];
        boundary_coeffs_y += boundary_coeffs[(cell_offset + i) * 3 + 1];
        boundary_coeffs_z += boundary_coeffs[(cell_offset + i) * 3 + 2];
    }
    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + internal_coeffs_x;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + internal_coeffs_y;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + internal_coeffs_z;
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + boundary_coeffs_x;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + boundary_coeffs_y;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + boundary_coeffs_z;
}

__global__ void fvc_grad_internal_face(int num_cells,
        const int* csr_row_index, const int* csr_col_index, const int* csr_diag_index,
        const double* face_vector, const double* weight, const double* pressure,
        const double* b_input, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_p = pressure[index];
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    double grad_bx_low = 0;
    double grad_bx_upp = 0;
    double grad_by_low = 0;
    double grad_by_upp = 0;
    double grad_bz_low = 0;
    double grad_bz_upp = 0;
    for (int i = row_index; i < next_row_index; i++) {
      int inner_index = i - row_index;
      // lower
      if (inner_index < diag_index) {
        int neighbor_index = neighbor_offset + inner_index;
        double w = weight[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        int neighbor_cell_id = csr_col_index[row_index + inner_index];
        double neighbor_cell_p = pressure[neighbor_cell_id];
        double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
        grad_bx_low -= face_p * sfx;
        grad_by_low -= face_p * sfy;
        grad_bz_low -= face_p * sfz;
      }
      // upper
      if (inner_index > diag_index) {
        int neighbor_index = neighbor_offset + inner_index - 1;
        double w = weight[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        int neighbor_cell_id = csr_col_index[row_index + inner_index];
        double neighbor_cell_p = pressure[neighbor_cell_id];
        double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
        grad_bx_upp += face_p * sfx;
        grad_by_upp += face_p * sfy;
        grad_bz_upp += face_p * sfz;
      }
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + grad_bx_low + grad_bx_upp;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + grad_by_low + grad_by_upp;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + grad_bz_low + grad_bz_upp;
}

__global__ void fvc_grad_boundary_face(int num_cells, int num_boundary_cells,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_face_vector, const double* boundary_pressure,
        const double* b_input, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // compute boundary gradient
    double grad_bx = 0; 
    double grad_by = 0; 
    double grad_bz = 0; 
    for (int i = cell_offset; i < next_cell_offset; i++) {
      double sfx = boundary_face_vector[i * 3 + 0];
      double sfy = boundary_face_vector[i * 3 + 1];
      double sfz = boundary_face_vector[i * 3 + 2];
      double face_p = boundary_pressure[i];
      grad_bx += face_p * sfx;
      grad_by += face_p * sfy;
      grad_bz += face_p * sfz;
    }

    //// correct the boundary gradient
    //double nx = boundary_face_vector[face_index * 3 + 0] / magSf[face_index];
    //double ny = boundary_face_vector[face_index * 3 + 1] / magSf[face_index];
    //double nz = boundary_face_vector[face_index * 3 + 2] / magSf[face_index];
    //double sn_grad = 0;
    //double grad_correction = sn_grad * volume[cell_index] - (nx * grad_bx + ny * grad_by + nz * grad_bz);
    //grad_bx += nx * grad_correction; 
    //grad_by += ny * grad_correction; 
    //grad_bz += nz * grad_correction; 

    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + grad_bx;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + grad_by;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + grad_bz;
}

__global__ void add_fvMatrix_kernel(int num_cells, int num_faces,
        const int* csr_row_index,
        const double* turbSrc_A, const double* turbSrc_b,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int csr_dim = num_cells + num_faces;
    double A_entry;

    for (int i = row_index; i < next_row_index; i++)
    {
      A_entry = turbSrc_A[i];
      A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + A_entry;
      A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + A_entry;
      A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + A_entry;
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + turbSrc_b[index * 3 + 0];
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + turbSrc_b[index * 3 + 1];
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + turbSrc_b[index * 3 + 2];
}

__global__ void offdiagPermutation(const int num_faces, const int *permedIndex,
        const double *d_phi_init, double *d_phi)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_faces) return;

    int p = permedIndex[index];
    d_phi[index] = d_phi_init[p];
}

__global__ void boundaryPermutation(const int num_boundary_faces, const int *bouPermedIndex,
        const double *ueqn_internalCoeffs_init, const double *ueqn_boundaryCoeffs_init, 
        const double *ueqn_laplac_internalCoeffs_init, const double *ueqn_laplac_boundaryCoeffs_init, 
        const double *boundary_pressure_init, const double *boundary_velocity_init,
        double *ueqn_internalCoeffs, double *ueqn_boundaryCoeffs, 
        double *ueqn_laplac_internalCoeffs, double *ueqn_laplac_boundaryCoeffs, 
        double *boundary_pressure, double *boundary_velocity,
        double *boundary_nuEff_init, double *boundary_nuEff,
        double *boundary_rho_init, double *boundary_rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces) return;

    int p = bouPermedIndex[index];
    ueqn_internalCoeffs[3*index] = ueqn_internalCoeffs_init[3*p];
    ueqn_internalCoeffs[3*index + 1] = ueqn_internalCoeffs_init[3*p + 1];
    ueqn_internalCoeffs[3*index + 2] = ueqn_internalCoeffs_init[3*p + 2];
    ueqn_boundaryCoeffs[3*index] = ueqn_boundaryCoeffs_init[3*p];
    ueqn_boundaryCoeffs[3*index + 1] = ueqn_boundaryCoeffs_init[3*p + 1];
    ueqn_boundaryCoeffs[3*index + 2] = ueqn_boundaryCoeffs_init[3*p + 2];

    ueqn_laplac_internalCoeffs[3*index] = ueqn_laplac_internalCoeffs_init[3*p];
    ueqn_laplac_internalCoeffs[3*index + 1] = ueqn_laplac_internalCoeffs_init[3*p + 1];
    ueqn_laplac_internalCoeffs[3*index + 2] = ueqn_laplac_internalCoeffs_init[3*p + 2];
    ueqn_laplac_boundaryCoeffs[3*index] = ueqn_laplac_boundaryCoeffs_init[3*p];
    ueqn_laplac_boundaryCoeffs[3*index + 1] = ueqn_laplac_boundaryCoeffs_init[3*p + 1];
    ueqn_laplac_boundaryCoeffs[3*index + 2] = ueqn_laplac_boundaryCoeffs_init[3*p + 2];

    boundary_velocity[3*index] = boundary_velocity_init[3*p];
    boundary_velocity[3*index + 1] = boundary_velocity_init[3*p + 1];
    boundary_velocity[3*index + 2] = boundary_velocity_init[3*p + 2];
    boundary_pressure[index] = boundary_pressure_init[p];
    boundary_rho[index] = boundary_rho_init[p];
    boundary_nuEff[index] = boundary_nuEff_init[p];
}

__global__ void csrPermutation(const int num_Nz, const int *csrPermedIndex, 
        const double *d_turbSrc_A_init, double *d_turbSrc_A)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_Nz) return;

    int p = csrPermedIndex[index];
    d_turbSrc_A[index] = d_turbSrc_A_init[p];
}

__global__ void fvc_grad_vector_internal(int num_cells,
        const int* csr_row_index, const int* csr_col_index, const int* csr_diag_index,
        const double* sf, const double* vf, const double* tlambdas, const double* volume,
        double* grad) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    if (index == 0)
    {
        printf("diag_index = %d\n", diag_index);
        printf("row_elements = %d\n", row_elements);
    }

    double own_vf_x = vf[index * 3 + 0];
    double own_vf_y = vf[index * 3 + 1];
    double own_vf_z = vf[index * 3 + 2];
    double grad_xx = 0;
    double grad_xy = 0;
    double grad_xz = 0;
    double grad_yx = 0;
    double grad_yy = 0;
    double grad_yz = 0;
    double grad_zx = 0;
    double grad_zy = 0;
    double grad_zz = 0;
    // lower
    for (int i = 0; i < diag_index; i++) {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_x = vf[neighbor_cell_id * 3 + 0];
        double neighbor_vf_y = vf[neighbor_cell_id * 3 + 1];
        double neighbor_vf_z = vf[neighbor_cell_id * 3 + 2];
        double face_x = (1 - w) * own_vf_x + w * neighbor_vf_x;
        double face_y = (1 - w) * own_vf_y + w * neighbor_vf_y;
        double face_z = (1 - w) * own_vf_z + w * neighbor_vf_z;
        grad_xx -= sf_x * face_x;
        grad_xy -= sf_x * face_y;
        grad_xz -= sf_x * face_z;
        grad_yx -= sf_y * face_x;
        grad_yy -= sf_y * face_y;
        grad_yz -= sf_y * face_z;
        grad_zx -= sf_z * face_x;
        grad_zy -= sf_z * face_y;
        grad_zz -= sf_z * face_z;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++) {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_x = vf[neighbor_cell_id * 3 + 0];
        double neighbor_vf_y = vf[neighbor_cell_id * 3 + 1];
        double neighbor_vf_z = vf[neighbor_cell_id * 3 + 2];
        double face_x = w * own_vf_x + (1 - w) * neighbor_vf_x;
        double face_y = w * own_vf_y + (1 - w) * neighbor_vf_y;
        double face_z = w * own_vf_z + (1 - w) * neighbor_vf_z;
        grad_xx += sf_x * face_x;
        grad_xy += sf_x * face_y;
        grad_xz += sf_x * face_z;
        grad_yx += sf_y * face_x;
        grad_yy += sf_y * face_y;
        grad_yz += sf_y * face_z;
        grad_zx += sf_z * face_x;
        grad_zy += sf_z * face_y;
        grad_zz += sf_z * face_z;
        // if (index == 0)
        // {
        //     printf("grad_xx = %.20lf\n", grad_xx);
        //     // printf("sf_x = %.20lf\n", sf_x);
        //     // printf("face_x = %.20lf\n", face_x);
        // }
        }
    double vol = volume[index];
    grad[index * 9 + 0] = grad_xx / vol;
    grad[index * 9 + 1] = grad_xy / vol;
    grad[index * 9 + 2] = grad_xz / vol;
    grad[index * 9 + 3] = grad_yx / vol;
    grad[index * 9 + 4] = grad_yy / vol;
    grad[index * 9 + 5] = grad_yz / vol;
    grad[index * 9 + 6] = grad_zx / vol;
    grad[index * 9 + 7] = grad_zy / vol;
    grad[index * 9 + 8] = grad_zz / vol;
}

__global__ void fvc_grad_vector_boundary(int num_cells, int num_boundary_cells,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_sf, const double* boundary_vf, const double* volume,
        double* grad, double* grad_boundary_init) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    double grad_xx = 0;
    double grad_xy = 0;
    double grad_xz = 0;
    double grad_yx = 0;
    double grad_yy = 0;
    double grad_yz = 0;
    double grad_zx = 0;
    double grad_zy = 0;
    double grad_zz = 0;
    for (int i = cell_offset; i < next_cell_offset; i++) {
      double sf_x = boundary_sf[i * 3 + 0];
      double sf_y = boundary_sf[i * 3 + 1];
      double sf_z = boundary_sf[i * 3 + 2];
      double vf_x = boundary_vf[i * 3 + 0];
      double vf_y = boundary_vf[i * 3 + 1];
      double vf_z = boundary_vf[i * 3 + 2];
      grad_xx += sf_x * vf_x;
      grad_xy += sf_x * vf_y;
      grad_xz += sf_x * vf_z;
      grad_yx += sf_y * vf_x;
      grad_yy += sf_y * vf_y;
      grad_yz += sf_y * vf_z;
      grad_zx += sf_z * vf_x;
      grad_zy += sf_z * vf_y;
      grad_zz += sf_z * vf_z;
    }

    double vol = volume[cell_index];
    
    grad[cell_index * 9 + 0] += grad_xx / vol;
    grad[cell_index * 9 + 1] += grad_xy / vol;
    grad[cell_index * 9 + 2] += grad_xz / vol;
    grad[cell_index * 9 + 3] += grad_yx / vol;
    grad[cell_index * 9 + 4] += grad_yy / vol;
    grad[cell_index * 9 + 5] += grad_yz / vol;
    grad[cell_index * 9 + 6] += grad_zx / vol;
    grad[cell_index * 9 + 7] += grad_zy / vol;
    grad[cell_index * 9 + 8] += grad_zz / vol;

    grad_boundary_init[index * 9 + 0] = grad[cell_index * 9 + 0];
    grad_boundary_init[index * 9 + 1] = grad[cell_index * 9 + 1];
    grad_boundary_init[index * 9 + 2] = grad[cell_index * 9 + 2];
    grad_boundary_init[index * 9 + 3] = grad[cell_index * 9 + 3];
    grad_boundary_init[index * 9 + 4] = grad[cell_index * 9 + 4];
    grad_boundary_init[index * 9 + 5] = grad[cell_index * 9 + 5];
    grad_boundary_init[index * 9 + 6] = grad[cell_index * 9 + 6];
    grad_boundary_init[index * 9 + 7] = grad[cell_index * 9 + 7];
    grad_boundary_init[index * 9 + 8] = grad[cell_index * 9 + 8];
    
}

__global__ void correct_boundary_conditions(int num_boundary_cells,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_sf, const double* mag_sf,
        double* boundary_grad_init, double* boundary_grad) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];

    // initialize boundary_grad
    double grad_xx = boundary_grad_init[index * 9 + 0];
    double grad_xy = boundary_grad_init[index * 9 + 1];
    double grad_xz = boundary_grad_init[index * 9 + 2];
    double grad_yx = boundary_grad_init[index * 9 + 3];
    double grad_yy = boundary_grad_init[index * 9 + 4];
    double grad_yz = boundary_grad_init[index * 9 + 5];
    double grad_zx = boundary_grad_init[index * 9 + 6];
    double grad_zy = boundary_grad_init[index * 9 + 7];
    double grad_zz = boundary_grad_init[index * 9 + 8];

    for (int i = cell_offset; i < next_cell_offset; i++) {
        // OpenFoam code
        // const vectorField n
        //     (
        //      vsf.mesh().Sf().boundaryField()[patchi]
        //      / vsf.mesh().magSf().boundaryField()[patchi]
        //     );
        // gGradbf[patchi] += n *
        //     (
        //      vsf.boundaryField()[patchi].snGrad()
        //      - (n & gGradbf[patchi])
        //     );

        double n_x = boundary_sf[i * 3 + 0] / mag_sf[i];
        double n_y = boundary_sf[i * 3 + 1] / mag_sf[i];
        double n_z = boundary_sf[i * 3 + 2] / mag_sf[i];
        double sn_grad_x = 0;
        double sn_grad_y = 0;
        double sn_grad_z = 0;
        double grad_correction_x = sn_grad_x - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx);
        double grad_correction_y = sn_grad_y - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
        double grad_correction_z = sn_grad_z - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);
        boundary_grad[i * 9 + 0] = grad_xx + n_x * grad_correction_x;
        boundary_grad[i * 9 + 1] = grad_xy + n_x * grad_correction_y;
        boundary_grad[i * 9 + 2] = grad_xz + n_x * grad_correction_z;
        boundary_grad[i * 9 + 3] = grad_yx + n_y * grad_correction_x;
        boundary_grad[i * 9 + 4] = grad_yy + n_y * grad_correction_y;
        boundary_grad[i * 9 + 5] = grad_yz + n_y * grad_correction_z;
        boundary_grad[i * 9 + 6] = grad_zx + n_z * grad_correction_x;
        boundary_grad[i * 9 + 7] = grad_zy + n_z * grad_correction_y;
        boundary_grad[i * 9 + 8] = grad_zz + n_z * grad_correction_z;
    } 
}

__global__ void dev2_t_tensor(int num, double* tensor) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num) return;

    double t_xx = tensor[index * 9 + 0];
    double t_xy = tensor[index * 9 + 1];
    double t_xz = tensor[index * 9 + 2];
    double t_yx = tensor[index * 9 + 3];
    double t_yy = tensor[index * 9 + 4];
    double t_yz = tensor[index * 9 + 5];
    double t_zx = tensor[index * 9 + 6];
    double t_zy = tensor[index * 9 + 7];
    double t_zz = tensor[index * 9 + 8];
    double trace_coeff = (2. / 3.) * (t_xx + t_yy + t_zz);
    tensor[index * 9 + 0] = t_xx - trace_coeff;
    tensor[index * 9 + 1] = t_yx;
    tensor[index * 9 + 2] = t_zx;
    tensor[index * 9 + 3] = t_xy;
    tensor[index * 9 + 4] = t_yy - trace_coeff;
    tensor[index * 9 + 5] = t_zy;
    tensor[index * 9 + 6] = t_xz;
    tensor[index * 9 + 7] = t_yz;
    tensor[index * 9 + 8] = t_zz - trace_coeff;
    
}

__global__ void fvc_div_tensor_internal(int num_cells,
        const int* csr_row_index, const int* csr_col_index, const int* csr_diag_index,
        const double* scalar0, const double* scalar1,
        const double* sf, const double* vf, const double* tlambdas, const double* volume,
        const double sign, const double* b_input, double* b_output, double* grad_test) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_vf_xx = vf[index * 9 + 0];
    double own_vf_xy = vf[index * 9 + 1];
    double own_vf_xz = vf[index * 9 + 2];
    double own_vf_yx = vf[index * 9 + 3];
    double own_vf_yy = vf[index * 9 + 4];
    double own_vf_yz = vf[index * 9 + 5];
    double own_vf_zx = vf[index * 9 + 6];
    double own_vf_zy = vf[index * 9 + 7];
    double own_vf_zz = vf[index * 9 + 8];
    double sum_x = 0;
    double sum_y = 0;
    double sum_z = 0;
    
    // lower
    for (int i = 0; i < diag_index; i++) {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        // double w = 0.5;
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_xx = vf[neighbor_cell_id * 9 + 0];
        double neighbor_vf_xy = vf[neighbor_cell_id * 9 + 1];
        double neighbor_vf_xz = vf[neighbor_cell_id * 9 + 2];
        double neighbor_vf_yx = vf[neighbor_cell_id * 9 + 3];
        double neighbor_vf_yy = vf[neighbor_cell_id * 9 + 4];
        double neighbor_vf_yz = vf[neighbor_cell_id * 9 + 5];
        double neighbor_vf_zx = vf[neighbor_cell_id * 9 + 6];
        double neighbor_vf_zy = vf[neighbor_cell_id * 9 + 7];
        double neighbor_vf_zz = vf[neighbor_cell_id * 9 + 8];
        double face_xx = (1 - w) * own_vf_xx + w * neighbor_vf_xx;
        double face_xy = (1 - w) * own_vf_xy + w * neighbor_vf_xy;
        double face_xz = (1 - w) * own_vf_xz + w * neighbor_vf_xz;
        double face_yx = (1 - w) * own_vf_yx + w * neighbor_vf_yx;
        double face_yy = (1 - w) * own_vf_yy + w * neighbor_vf_yy;
        double face_yz = (1 - w) * own_vf_yz + w * neighbor_vf_yz;
        double face_zx = (1 - w) * own_vf_zx + w * neighbor_vf_zx;
        double face_zy = (1 - w) * own_vf_zy + w * neighbor_vf_zy;
        double face_zz = (1 - w) * own_vf_zz + w * neighbor_vf_zz;
        sum_x -= sf_x * face_xx + sf_y * face_yx + sf_z * face_zx;
        sum_y -= sf_x * face_xy + sf_y * face_yy + sf_z * face_zy;
        sum_z -= sf_x * face_xz + sf_y * face_yz + sf_z * face_zz;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++) {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        // double w = 0.5;
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_xx = vf[neighbor_cell_id * 9 + 0];
        double neighbor_vf_xy = vf[neighbor_cell_id * 9 + 1];
        double neighbor_vf_xz = vf[neighbor_cell_id * 9 + 2];
        double neighbor_vf_yx = vf[neighbor_cell_id * 9 + 3];
        double neighbor_vf_yy = vf[neighbor_cell_id * 9 + 4];
        double neighbor_vf_yz = vf[neighbor_cell_id * 9 + 5];
        double neighbor_vf_zx = vf[neighbor_cell_id * 9 + 6];
        double neighbor_vf_zy = vf[neighbor_cell_id * 9 + 7];
        double neighbor_vf_zz = vf[neighbor_cell_id * 9 + 8];
        double face_xx = w * own_vf_xx + (1 - w) * neighbor_vf_xx;
        double face_xy = w * own_vf_xy + (1 - w) * neighbor_vf_xy;
        double face_xz = w * own_vf_xz + (1 - w) * neighbor_vf_xz;
        double face_yx = w * own_vf_yx + (1 - w) * neighbor_vf_yx;
        double face_yy = w * own_vf_yy + (1 - w) * neighbor_vf_yy;
        double face_yz = w * own_vf_yz + (1 - w) * neighbor_vf_yz;
        double face_zx = w * own_vf_zx + (1 - w) * neighbor_vf_zx;
        double face_zy = w * own_vf_zy + (1 - w) * neighbor_vf_zy;
        double face_zz = w * own_vf_zz + (1 - w) * neighbor_vf_zz;
        sum_x += sf_x * face_xx + sf_y * face_yx + sf_z * face_zx;
        sum_y += sf_x * face_xy + sf_y * face_yy + sf_z * face_zy;
        sum_z += sf_x * face_xz + sf_y * face_yz + sf_z * face_zz;
    }
    double coeff = scalar0[index] * scalar1[index];
    double vol = volume[index];
    
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + sum_x * coeff * sign / vol;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + sum_y * coeff * sign / vol;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + sum_z * coeff * sign / vol;

    // test
    grad_test[index * 9 + 0] = own_vf_xx * coeff;
    grad_test[index * 9 + 1] = own_vf_xy * coeff;
    grad_test[index * 9 + 2] = own_vf_xz * coeff;
    grad_test[index * 9 + 3] = own_vf_yx * coeff;
    grad_test[index * 9 + 4] = own_vf_yy * coeff;
    grad_test[index * 9 + 5] = own_vf_yz * coeff;
    grad_test[index * 9 + 6] = own_vf_zx * coeff;
    grad_test[index * 9 + 7] = own_vf_zy * coeff;
    grad_test[index * 9 + 8] = own_vf_zz * coeff;
}

__global__ void fvc_div_tensor_boundary(int num_cells, int num_boundary_cells,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_scalar0, const double* boundary_scalar1,
        const double* boundary_sf, const double* boundary_vf, const double* volume,
        const double sign, const double* b_input, double* b_output, double* grad_boundary_test) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // OpenFoam code
    // Foam::surfaceInterpolationScheme<Type>::dotInterpolate
    // if (vf.boundaryField()[pi].coupled())
    // {
    //     psf =
    //         pSf
    //         & (
    //                 pLambda*vf.boundaryField()[pi].patchInternalField()
    //                 + (1.0 - pLambda)*vf.boundaryField()[pi].patchNeighbourField()
    //           );
    // }
    // else
    // {
    //     psf = pSf & vf.boundaryField()[pi];
    // }
    // tmp<GeometricField<Type, fvPatchField, volMesh>> surfaceIntegrate
    // forAll(mesh.boundary()[patchi], facei)
    // {
    //     ivf[pFaceCells[facei]] += pssf[facei];
    // }
    double sum_x = 0; 
    double sum_y = 0; 
    double sum_z = 0; 
    for (int i = cell_offset; i < next_cell_offset; i++) {
      double sf_x = boundary_sf[i * 3 + 0];
      double sf_y = boundary_sf[i * 3 + 1];
      double sf_z = boundary_sf[i * 3 + 2];
      double face_xx = boundary_vf[i * 9 + 0];
      double face_xy = boundary_vf[i * 9 + 1];
      double face_xz = boundary_vf[i * 9 + 2];
      double face_yx = boundary_vf[i * 9 + 3];
      double face_yy = boundary_vf[i * 9 + 4];
      double face_yz = boundary_vf[i * 9 + 5];
      double face_zx = boundary_vf[i * 9 + 6];
      double face_zy = boundary_vf[i * 9 + 7];
      double face_zz = boundary_vf[i * 9 + 8];

      // if not coupled
      double coeff = boundary_scalar0[i] * boundary_scalar1[i];
      sum_x += (sf_x * face_xx + sf_y * face_yx + sf_z * face_zx) * coeff;
      sum_y += (sf_x * face_xy + sf_y * face_yy + sf_z * face_zy) * coeff;
      sum_z += (sf_x * face_xz + sf_y * face_yz + sf_z * face_zz) * coeff;

    grad_boundary_test[i * 9 + 0] = face_xx * coeff;
    grad_boundary_test[i * 9 + 1] = face_xy * coeff;
    grad_boundary_test[i * 9 + 2] = face_xz * coeff;
    grad_boundary_test[i * 9 + 3] = face_yx * coeff;
    grad_boundary_test[i * 9 + 4] = face_yy * coeff;
    grad_boundary_test[i * 9 + 5] = face_yz * coeff;
    grad_boundary_test[i * 9 + 6] = face_zx * coeff;
    grad_boundary_test[i * 9 + 7] = face_zy * coeff;
    grad_boundary_test[i * 9 + 8] = face_zz * coeff;
      
    }
    double vol = volume[cell_index];
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + sum_x * sign / vol;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + sum_y * sign / vol;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + sum_z * sign / vol;

    // test
    // grad_boundary_test[index * 3 + 0] = sum_x;
    // grad_boundary_test[index * 3 + 1] = sum_y;
    // grad_boundary_test[index * 3 + 2] = sum_z;
}

__global__ void fvm_laplacian_uncorrected_vector_internal(int num_cells, int num_faces,
        const int* csr_row_index, const int* csr_col_index, const int* csr_diag_index,
        const double* scalar0, const double* scalar1, const double* weight,
        const double* magsf, const double* distance,
        const double sign, const double* A_csr_input, double* A_csr_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;

    double own_scalar0 = scalar0[index];
    double own_scalar1 = scalar1[index];

    // fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm.negSumDiag();
    double sum_diag = 0;
    // lower
    for (int i = 0; i < diag_index; i++) {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_scalar0 = scalar0[neighbor_cell_id];
        double nei_scalar1 = scalar1[neighbor_cell_id];
        double face_scalar0 = w * own_scalar0 + (1 - w) * nei_scalar0;
        double face_scalar1 = w * own_scalar1 + (1 - w) * nei_scalar1;
        double gamma = face_scalar0 * face_scalar1;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[csr_dim * 0 + row_index + i] = A_csr_input[csr_dim * 0 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 1 + row_index + i] = A_csr_input[csr_dim * 1 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 2 + row_index + i] = A_csr_input[csr_dim * 2 + row_index + i] + coeff * sign;
        // if (index == 0)
        // {
        //     printf("gamma_magsf = %lf\n", gamma_magsf);
        //     printf("gamma_magsf = %lf\n", gamma_magsf);
        // }
        
        sum_diag += (-coeff);
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++) {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_scalar0 = scalar0[neighbor_cell_id];
        double nei_scalar1 = scalar1[neighbor_cell_id];
        double face_scalar0 = w * own_scalar0 + (1 - w) * nei_scalar0;
        double face_scalar1 = w * own_scalar1 + (1 - w) * nei_scalar1;
        double gamma = face_scalar0 * face_scalar1;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[csr_dim * 0 + row_index + i] = A_csr_input[csr_dim * 0 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 1 + row_index + i] = A_csr_input[csr_dim * 1 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 2 + row_index + i] = A_csr_input[csr_dim * 2 + row_index + i] + coeff * sign;
        sum_diag += (-coeff);
    }
    A_csr_output[csr_dim * 0 + row_index + diag_index] = A_csr_input[csr_dim * 0 + row_index + diag_index] + sum_diag * sign; // diag
    A_csr_output[csr_dim * 1 + row_index + diag_index] = A_csr_input[csr_dim * 1 + row_index + diag_index] + sum_diag * sign; // diag
    A_csr_output[csr_dim * 2 + row_index + diag_index] = A_csr_input[csr_dim * 2 + row_index + diag_index] + sum_diag * sign; // diag
}

__global__ void fvm_laplacian_uncorrected_vector_boundary(int num_cells, int num_faces, int num_boundary_cells,
        const int* csr_row_index, const int* csr_diag_index,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_scalar0, const double* boundary_scalar1,
        const double* boundary_magsf, const double* gradient_internal_coeffs, const double* gradient_boundary_coeffs,
        const double sign, const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    int row_index = csr_row_index[cell_index];
    int diag_index = csr_diag_index[cell_index];
    int csr_dim = num_cells + num_faces;
    int csr_index = row_index + diag_index;

    // OpenFoam code
    // if (pvf.coupled())
    // {
    //     fvm.internalCoeffs()[patchi] =
    //         pGamma*pvf.gradientInternalCoeffs(pDeltaCoeffs);
    //     fvm.boundaryCoeffs()[patchi] =
    //         -pGamma*pvf.gradientBoundaryCoeffs(pDeltaCoeffs);
    // }
    // else
    // {
    //     fvm.internalCoeffs()[patchi] = pGamma*pvf.gradientInternalCoeffs();
    //     fvm.boundaryCoeffs()[patchi] = -
    //         pGamma*pvf.gradientBoundaryCoeffs();
    // }
    double internal_coeffs_x = 0;
    double internal_coeffs_y = 0;
    double internal_coeffs_z = 0;
    double boundary_coeffs_x = 0;
    double boundary_coeffs_y = 0;
    double boundary_coeffs_z = 0;
    for (int i = cell_offset; i < next_cell_offset; i++) {
        double gamma = boundary_scalar0[i] * boundary_scalar1[i];
        double gamma_magsf = gamma * boundary_magsf[i];
        internal_coeffs_x += gamma_magsf * gradient_internal_coeffs[i * 3 + 0];
        internal_coeffs_y += gamma_magsf * gradient_internal_coeffs[i * 3 + 1];
        internal_coeffs_z += gamma_magsf * gradient_internal_coeffs[i * 3 + 2];
        boundary_coeffs_x += gamma_magsf * gradient_boundary_coeffs[i * 3 + 0];
        boundary_coeffs_y += gamma_magsf * gradient_boundary_coeffs[i * 3 + 1];
        boundary_coeffs_z += gamma_magsf * gradient_boundary_coeffs[i * 3 + 2];
    }

    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + internal_coeffs_x * sign;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + internal_coeffs_y * sign;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + internal_coeffs_z * sign;
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + boundary_coeffs_x * sign;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + boundary_coeffs_y * sign;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + boundary_coeffs_z * sign;
}

// constructor (construct mesh variable)
dfMatrix::dfMatrix(){}
dfMatrix::dfMatrix(int num_surfaces, int num_cells, int num_boundary_faces, int & num_boundary_cells_output,
    const int *neighbour, const int *owner, const double* volume, const double* weight, const double* face_vector, const double* face, const double* deltaCoeffs,
    std::vector<double> boundary_face_vector_init, std::vector<double> boundary_face_init, std::vector<double> boundary_deltaCoeffs_init,
    std::vector<int> boundary_cell_id_init, const std::string &modeStr, const std::string &cfgFile)
: num_cells(num_cells), num_faces(num_surfaces*2), num_surfaces(num_surfaces), num_iteration(0),
  num_boundary_faces(num_boundary_faces), h_volume(volume), time_monitor_CPU(0.), time_monitor_GPU(0.)
{   
    // 
    unsigned int flags = 0;
    checkCudaErrors(cudaGetDeviceFlags(&flags));
    flags |= cudaDeviceScheduleYield;
    checkCudaErrors(cudaSetDeviceFlags(flags));
    
    // initialize vectors
    // allocate field vector in pin memory
    cudaMallocHost(&h_phi_init, num_faces * sizeof(double));
    // h_phi_vec_init.assign(h_phi_init, h_phi_init + num_faces);

    h_weight_vec_init.resize(num_faces);
    h_weight_vec.resize(num_faces);
    h_face_vector_vec_init.resize(num_faces*3);
    h_face_vector_vec.resize(num_faces*3);
    h_face_vec_init.resize(num_faces);
    h_face_vec.resize(num_faces);
    h_deltaCoeffs_vec_init.resize(num_faces);
    h_deltaCoeffs_vec.resize(num_faces);
    h_turbSrc_init_mtx_vec.resize(num_faces + num_cells);
    h_turbSrc_init_1mtx.resize(num_faces + num_cells);
    h_turbSrc_init_src_vec.resize(3*num_cells);
    h_turbSrc_src_vec.resize(3*num_cells);

    // byte sizes
    cell_bytes = num_cells * sizeof(double);
    cell_vec_bytes = num_cells * 3 * sizeof(double);
    cell_index_bytes = num_cells * sizeof(int);

    face_bytes = num_faces * sizeof(double);
    face_vec_bytes = num_faces * 3 * sizeof(double);
    face_index_bytes = num_faces * sizeof(int);

    // A_csr has one more element in each row: itself
    csr_row_index_bytes = (num_cells + 1) * sizeof(int);
    csr_col_index_bytes = (num_cells + num_faces) * sizeof(int);
    csr_value_bytes = (num_cells + num_faces) * sizeof(double);
    csr_value_vec_bytes = (num_cells + num_faces) * 3 * sizeof(double);

    h_A_csr = new double[(num_cells + num_faces) * 3];
    // h_b = new double[num_cells * 3];
    // h_psi = new double[num_cells * 3];
    cudaMallocHost(&h_b, cell_vec_bytes);
    cudaMallocHost(&h_psi, cell_vec_bytes);

    /************************construct mesh variables****************************/
    /**
     * 1. h_csr_row_index & h_csr_diag_index
    */
    std::vector<int> h_mtxEntry_perRow_vec(num_cells);
    std::vector<int> h_csr_diag_index_vec(num_cells);
    std::vector<int> h_csr_row_index_vec(num_cells + 1, 0);

    for (int faceI = 0; faceI < num_surfaces; faceI++)
    {
        h_csr_diag_index_vec[neighbour[faceI]]++;
        h_mtxEntry_perRow_vec[neighbour[faceI]]++;
        h_mtxEntry_perRow_vec[owner[faceI]]++;
    }

    // - consider diagnal element in each row
    std::transform(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_mtxEntry_perRow_vec.begin(), [](int n)
        {return n + 1;});
    // - construct h_csr_row_index & h_csr_diag_index
    std::partial_sum(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_csr_row_index_vec.begin()+1);
    // - assign h_csr_row_index & h_csr_diag_index
    h_A_csr_row_index = h_csr_row_index_vec.data();
    h_A_csr_diag_index = h_csr_diag_index_vec.data();

    /**
     * 2. h_csr_col_index
    */
    std::vector<int> rowIndex(num_faces + num_cells), colIndex(num_faces + num_cells), diagIndex(num_cells);
    std::iota(diagIndex.begin(), diagIndex.end(), 0);

    // initialize the RowIndex (rowIndex of lower + upper + diagnal)
    std::copy(neighbour, neighbour + num_surfaces, rowIndex.begin());
    std::copy(owner, owner + num_surfaces, rowIndex.begin() + num_surfaces);
    std::copy(diagIndex.begin(), diagIndex.end(), rowIndex.begin() + num_faces);
    // initialize the ColIndex (colIndex of lower + upper + diagnal)
    std::copy(owner, owner + num_surfaces, colIndex.begin());
    std::copy(neighbour, neighbour + num_surfaces, colIndex.begin() + num_surfaces);
    std::copy(diagIndex.begin(), diagIndex.end(), colIndex.begin() + num_faces);

    // - construct hashTable for sorting
    std::multimap<int,int> rowColPair;
    for (int i = 0; i < 2*num_surfaces+num_cells; i++)
    {
        rowColPair.insert(std::make_pair(rowIndex[i], colIndex[i]));
    }
    // - sort
    std::vector<std::pair<int, int>> globalPerm(rowColPair.begin(), rowColPair.end());
    std::sort(globalPerm.begin(), globalPerm.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
    if (pair1.first != pair2.first) {
        return pair1.first < pair2.first;
    } else {
        return pair1.second < pair2.second;
    }
    });

    std::vector<int> h_csr_col_index_vec;
    std::transform(globalPerm.begin(), globalPerm.end(), std::back_inserter(h_csr_col_index_vec), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });
    h_A_csr_col_index = h_csr_col_index_vec.data();

    // construct a tmp permutated List for add fvMatrix
    std::vector<int> tmp_permutation(2*num_surfaces + num_cells);
    std::vector<int> tmp_rowIndex(2*num_surfaces + num_cells);
    std::iota(tmp_permutation.begin(), tmp_permutation.end(), 0);
    std::copy(neighbour, neighbour + num_surfaces, tmp_rowIndex.begin());
    std::copy(diagIndex.begin(), diagIndex.end(), tmp_rowIndex.begin() + num_surfaces);
    std::copy(owner, owner + num_surfaces, tmp_rowIndex.begin() + num_surfaces + num_cells);
    std::multimap<int,int> tmpPair;
    for (int i = 0; i < 2*num_surfaces+num_cells; i++)
    {
        tmpPair.insert(std::make_pair(tmp_rowIndex[i], tmp_permutation[i]));
    }
    std::vector<std::pair<int, int>> tmpPerm(tmpPair.begin(), tmpPair.end());
    std::sort(tmpPerm.begin(), tmpPerm.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
    if (pair1.first != pair2.first) {
        return pair1.first < pair2.first;
    } else {
        return pair1.second < pair2.second;
    }
    });
    std::transform(tmpPerm.begin(), tmpPerm.end(), std::back_inserter(tmpPermutatedList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });
    

    /**
     * 3. boundary imformations
    */
    // get boundPermutation and offset lists
    std::vector<int> boundPermutationListInit(num_boundary_faces);
    std::vector<int> boundOffsetList;
    std::iota(boundPermutationListInit.begin(), boundPermutationListInit.end(), 0);

    // - construct hashTable for sorting
    std::multimap<int,int> boundPermutation;
    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundPermutation.insert(std::make_pair(boundary_cell_id_init[i], boundPermutationListInit[i]));
    }

    // - sort 
    std::vector<std::pair<int, int>> boundPermPair(boundPermutation.begin(), boundPermutation.end());
    std::sort(boundPermPair.begin(), boundPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });

    // - construct boundPermedIndex and boundary_cell_id
    std::vector<int> boundary_cell_id;
    boundPermutationList.clear();
    std::transform(boundPermPair.begin(), boundPermPair.end(), std::back_inserter(boundary_cell_id), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    std::transform(boundPermPair.begin(), boundPermPair.end(), std::back_inserter(boundPermutationList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // construct boundary_cell_offset
    std::map<int, int> countMap;
    std::vector<int> boundaryCellcount;
    for (const auto& cellIndex : boundary_cell_id)
        ++ countMap[cellIndex];
    for (const auto& [cellIndex, count] : countMap)
        boundaryCellcount.push_back(count);

    num_boundary_cells = boundaryCellcount.size();
    num_boundary_cells_output = num_boundary_cells;

    std::vector<int> boundary_cell_offset(boundaryCellcount.size() + 1, 0);
    std::partial_sum(boundaryCellcount.begin(), boundaryCellcount.end(), boundary_cell_offset.begin()+1);
    
    // assign h_boundary_cell_offset & h_boundary_cell_id
    h_boundary_cell_offset = boundary_cell_offset.data();
    h_boundary_cell_id = boundary_cell_id.data();

    // 
    boundary_cell_bytes = num_boundary_cells * sizeof(double);
    boundary_cell_vec_bytes = num_boundary_cells * 3 * sizeof(double);
    boundary_cell_index_bytes = num_boundary_cells * sizeof(int);

    boundary_face_bytes = num_boundary_faces * sizeof(double);
    boundary_face_vec_bytes = num_boundary_faces * 3 * sizeof(double);
    boundary_face_index_bytes = num_boundary_faces * sizeof(int);

    ueqn_internalCoeffs.resize(3*num_boundary_faces);
    ueqn_boundaryCoeffs.resize(3*num_boundary_faces);

    boundary_face_vector.resize(3*num_boundary_faces);
    boundary_pressure.resize(num_boundary_faces);
    boundary_face.resize(num_boundary_faces);
    boundary_deltaCoeffs.resize(num_boundary_faces);


    /**
     * 4. permutation list for field variables
    */
    std::vector<int> offdiagRowIndex(2*num_surfaces), permIndex(2*num_surfaces);
    // - initialize the offdiagRowIndex (rowIndex of lower + rowIndex of upper)
    std::copy(neighbour, neighbour + num_surfaces, offdiagRowIndex.begin());
    std::copy(owner, owner + num_surfaces, offdiagRowIndex.begin() + num_surfaces);

    // - initialize the permIndex (0, 1, ..., 2*num_surfaces)
    std::iota(permIndex.begin(), permIndex.end(), 0);

    // - construct hashTable for sorting
    std::multimap<int,int> permutation;
    for (int i = 0; i < 2*num_surfaces; i++)
    {
        permutation.insert(std::make_pair(offdiagRowIndex[i], permIndex[i]));
    }
    // - sort 
    std::vector<std::pair<int, int>> permPair(permutation.begin(), permutation.end());
    std::sort(permPair.begin(), permPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    // - form permedIndex list
    std::transform(permPair.begin(), permPair.end(), std::back_inserter(permedIndex), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // copy and permutate cell variables
    std::copy(weight, weight + num_surfaces, h_weight_vec_init.begin());
    std::copy(weight, weight + num_surfaces, h_weight_vec_init.begin() + num_surfaces);
    std::copy(face_vector, face_vector + 3*num_surfaces, h_face_vector_vec_init.begin());
    std::copy(face_vector, face_vector + 3*num_surfaces, h_face_vector_vec_init.begin() + 3*num_surfaces);
    std::copy(face, face + num_surfaces, h_face_vec_init.begin());
    std::copy(face, face + num_surfaces, h_face_vec_init.begin() + num_surfaces);
    std::copy(deltaCoeffs, deltaCoeffs + num_surfaces, h_deltaCoeffs_vec_init.begin());
    std::copy(deltaCoeffs, deltaCoeffs + num_surfaces, h_deltaCoeffs_vec_init.begin() + num_surfaces);
    for (int i = 0; i < num_faces; i++)
    {
        h_weight_vec[i] = h_weight_vec_init[permedIndex[i]];
        h_face_vec[i] = h_face_vec_init[permedIndex[i]];
        h_deltaCoeffs_vec[i] = h_deltaCoeffs_vec_init[permedIndex[i]];
        h_face_vector_vec[i*3] = h_face_vector_vec_init[3*permedIndex[i]];
        h_face_vector_vec[i*3+1] = h_face_vector_vec_init[3*permedIndex[i]+1];
        h_face_vector_vec[i*3+2] = h_face_vector_vec_init[3*permedIndex[i]+2];
    }
    h_weight = h_weight_vec.data();
    h_face_vector = h_face_vector_vec.data();
    h_face = h_face_vec.data();
    h_deltaCoeffs = h_deltaCoeffs_vec.data();

    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundary_face_vector[3*i] = boundary_face_vector_init[3*boundPermutationList[i]];
        boundary_face_vector[3*i+1] = boundary_face_vector_init[3*boundPermutationList[i]+1];
        boundary_face_vector[3*i+2] = boundary_face_vector_init[3*boundPermutationList[i]+2];
        boundary_face[i] = boundary_face_init[boundPermutationList[i]];
        boundary_deltaCoeffs[i] = boundary_deltaCoeffs_init[boundPermutationList[i]];
    }
    h_boundary_face_vector = boundary_face_vector.data();
    h_boundary_face = boundary_face.data();
    h_boundary_deltaCoeffs = boundary_deltaCoeffs.data();

    /************************allocate memory on device****************************/
    int total_bytes = 0;

    checkCudaErrors(cudaMalloc((void**)&d_A_csr_row_index, csr_row_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A_csr_col_index, csr_col_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A_csr_diag_index, cell_index_bytes));
    total_bytes += (csr_row_index_bytes + csr_col_index_bytes + cell_index_bytes);

    checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_rho_new, cell_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_volume, cell_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_pressure, cell_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_velocity_old, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_weight, face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_face, face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_deltaCoeffs, face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi, face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_init, face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_face_vector, face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_nuEff, cell_bytes));
    total_bytes += (cell_bytes * 5 + face_bytes * 5 + cell_vec_bytes + face_vec_bytes);

    checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_id, boundary_face_index_bytes));
    total_bytes += (boundary_face_index_bytes + (num_boundary_cells+1) * sizeof(int));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_pressure_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_pressure, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_velocity_init, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_velocity, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_face_vector, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_face, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_deltaCoeffs, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs_init, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs_init, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_laplac_internal_coeffs_init, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_laplac_boundary_coeffs_init, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_laplac_internal_coeffs, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_laplac_boundary_coeffs, boundary_face_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_nuEff, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_nuEff_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_init, boundary_face_bytes));
    total_bytes += (boundary_face_bytes*8 + boundary_face_vec_bytes * 11);

    checkCudaErrors(cudaMalloc((void**)&d_A_csr, csr_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_b, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_psi, cell_vec_bytes));
    total_bytes += (boundary_face_bytes + boundary_face_vec_bytes * 3);

    checkCudaErrors(cudaMalloc((void**)&d_turbSrc_A, csr_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_turbSrc_A_init, csr_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_turbSrc_b, cell_vec_bytes));
    total_bytes += (2*csr_value_bytes + cell_vec_bytes);

    checkCudaErrors(cudaMalloc((void**)&d_permedIndex, face_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_bouPermedIndex, boundary_face_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_tmpPermutatedList, csr_col_index_bytes));
    total_bytes += (face_index_bytes + boundary_face_index_bytes + csr_col_index_bytes);

    checkCudaErrors(cudaMalloc((void**)&d_grad, num_cells * 9 * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_grad_boundary, boundary_face_bytes * 9));
    checkCudaErrors(cudaMalloc((void**)&d_grad_boundary_init, boundary_cell_bytes * 9));
    total_bytes += (num_cells * 9 * sizeof(double) + boundary_face_bytes * 9 + boundary_cell_bytes * 9); // FIXME: rename


    fprintf(stderr, "Total bytes malloc on GPU: %.2fMB\n", total_bytes * 1.0 / 1024 / 1024);

    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));

    checkCudaErrors(cudaMemcpyAsync(d_A_csr_row_index, h_A_csr_row_index, csr_row_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_csr_col_index, h_A_csr_col_index, csr_col_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_A_csr_diag_index, h_A_csr_diag_index, cell_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_cell_offset, h_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_cell_id, h_boundary_cell_id, boundary_face_index_bytes, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_volume, h_volume, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_weight, h_weight, face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_face, h_face, face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_deltaCoeffs, h_deltaCoeffs, face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_face_vector, h_face_vector, face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_face_vector, h_boundary_face_vector, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_face, h_boundary_face, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_deltaCoeffs, h_boundary_deltaCoeffs, boundary_face_bytes, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_permedIndex, permedIndex.data(), face_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_bouPermedIndex, boundPermutationList.data(), boundary_face_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_tmpPermutatedList, tmpPermutatedList.data(), csr_col_index_bytes, cudaMemcpyHostToDevice, stream));

    // construct AmgXSolver
    UxSolver = new AmgXSolver(modeStr, cfgFile);
    UySolver = new AmgXSolver(modeStr, cfgFile);
    UzSolver = new AmgXSolver(modeStr, cfgFile);
}

dfMatrix::~dfMatrix()
{
}

void dfMatrix::fvm_ddt(double *rho_old, double *rho_new, double* vector_old)
{
    clock_t start = std::clock();
    // copy cell variables directly
    h_rho_new = rho_new;
    h_rho_old = rho_old;
    h_velocity_old = vector_old;
    
    // Copy the host input array in host memory to the device input array in device memory
    checkCudaErrors(cudaMemcpyAsync(d_rho_old, h_rho_old, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_rho_new, h_rho_new, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_velocity_old, h_velocity_old, cell_vec_bytes, cudaMemcpyHostToDevice, stream));

    // launch cuda kernel
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    printf("CUDA kernel fvm_ddt launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);
    fvm_ddt_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, rdelta_t,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_rho_old, d_rho_new, d_volume, d_velocity_old, d_A_csr, d_b, d_A_csr, d_b, d_psi);
    
    clock_t end = std::clock();
    time_monitor_GPU += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfMatrix::fvm_div(double* phi, double* ueqn_internalCoeffs_init, double* ueqn_boundaryCoeffs_init, 
    double* ueqn_laplac_internalCoeffs_init, double* ueqn_laplac_boundaryCoeffs_init,
    double* boundary_pressure_init, double* boundary_velocity_init,
    double *boundary_nuEff_init, double *boundary_rho_init)
{
    // copy and permutate face variables
    clock_t start = std::clock();
    std::copy(phi, phi + num_surfaces, h_phi_init);
    std::copy(phi, phi + num_surfaces, h_phi_init + num_surfaces);
    
    clock_t end = std::clock();
    time_monitor_CPU += double(end - start) / double(CLOCKS_PER_SEC);

    start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(d_phi_init, h_phi_init, face_bytes, cudaMemcpyHostToDevice, stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_faces + threads_per_block - 1) / threads_per_block;
    offdiagPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_faces, d_permedIndex, d_phi_init, d_phi);

    // copy and permutate boundary variable
    checkCudaErrors(cudaMemcpyAsync(d_internal_coeffs_init, ueqn_internalCoeffs_init, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_coeffs_init, ueqn_boundaryCoeffs_init, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_laplac_internal_coeffs_init, ueqn_laplac_internalCoeffs_init, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_laplac_boundary_coeffs_init, ueqn_laplac_boundaryCoeffs_init, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_velocity_init, boundary_velocity_init, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_pressure_init, boundary_pressure_init, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_nuEff_init, boundary_nuEff_init, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_rho_init, boundary_rho_init, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    
    
    blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
    boundaryPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces, d_bouPermedIndex, d_internal_coeffs_init, d_boundary_coeffs_init,
          d_laplac_internal_coeffs_init, d_laplac_boundary_coeffs_init, d_boundary_pressure_init, d_boundary_velocity_init, d_internal_coeffs, d_boundary_coeffs, 
          d_laplac_internal_coeffs, d_laplac_boundary_coeffs, d_boundary_pressure, d_boundary_velocity, d_boundary_nuEff_init, d_boundary_nuEff, d_boundary_rho_init, d_boundary_rho);

    // launch cuda kernel
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_div_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_weight, d_phi, d_A_csr, d_b, d_A_csr, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvm_div_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_boundary_cell_offset, d_boundary_cell_id,
          d_internal_coeffs, d_boundary_coeffs, d_A_csr, d_b, d_A_csr, d_b);
    end = std::clock();
    time_monitor_GPU += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfMatrix::fvc_grad(double* pressure)
{
    // copy cell variables directly
    clock_t start = std::clock();
    h_pressure = pressure;
    clock_t end = std::clock();
    time_monitor_CPU += double(end - start) / double(CLOCKS_PER_SEC);

    // Copy the host input array in host memory to the device input array in device memory
    start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(d_pressure, h_pressure, cell_bytes, cudaMemcpyHostToDevice, stream));
        
    // launch cuda kernel
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_internal_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
          d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
          d_face_vector, d_weight, d_pressure, d_b, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_boundary_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
          d_boundary_cell_offset, d_boundary_cell_id,
          d_boundary_face_vector, d_boundary_pressure, d_b, d_b);
    end = std::clock();
    time_monitor_GPU += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfMatrix::add_fvMatrix(double* turbSrc_low, double* turbSrc_diag, double* turbSrc_upp, double* turbSrc_source)
{
    // copy and permutate matrix variables
    clock_t start = std::clock();
    std::copy(turbSrc_low, turbSrc_low + num_surfaces, h_turbSrc_init_mtx_vec.begin());
    std::copy(turbSrc_diag, turbSrc_diag + num_cells, h_turbSrc_init_mtx_vec.begin() + num_surfaces);
    std::copy(turbSrc_upp, turbSrc_upp + num_surfaces, h_turbSrc_init_mtx_vec.begin() + num_surfaces + num_cells);
    std::copy(turbSrc_source, turbSrc_source + 3*num_cells, h_turbSrc_init_src_vec.begin());
    clock_t end = std::clock();
    time_monitor_CPU += double(end - start) / double(CLOCKS_PER_SEC);

    // permutate
    start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(d_turbSrc_A_init, h_turbSrc_init_mtx_vec.data(), csr_value_bytes, cudaMemcpyHostToDevice, stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + num_faces + threads_per_block - 1) / threads_per_block;
    // for (int i = 0; i < num_cells+2*num_surfaces; i++)
    //     h_turbSrc_init_1mtx[i] = h_turbSrc_init_mtx_vec[tmpPermutatedList[i]];
    csrPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells + num_faces, d_tmpPermutatedList, d_turbSrc_A_init, d_turbSrc_A);

    // Copy the host input array in host memory to the device input array in device memory
    checkCudaErrors(cudaMemcpyAsync(d_turbSrc_b, h_turbSrc_init_src_vec.data(), cell_vec_bytes, cudaMemcpyHostToDevice, stream));
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    add_fvMatrix_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
          d_A_csr_row_index, d_turbSrc_A, d_turbSrc_b, d_A_csr, d_b, d_A_csr, d_b);
    end = std::clock();
    time_monitor_GPU += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfMatrix::checkValue(bool print)
{
    checkCudaErrors(cudaMemcpyAsync(h_A_csr, d_A_csr, csr_value_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_b, d_b, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize stream
    checkCudaErrors(cudaStreamSynchronize(stream));
    if (print)
    {
        for (int i = 0; i < (2*num_surfaces + num_cells); i++)
            fprintf(stderr, "h_A_csr[%d]: %.15lf\n", i, h_A_csr[i]);
        for (int i = 0; i < num_cells * 3; i++)
            fprintf(stderr, "h_b[%d]: %.15lf\n", i, h_b[i]);
    }

    char *input_file = "of_output.txt";
    FILE *fp = fopen(input_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", input_file);
    }
    int readfile = 0;
    double *of_b = new double[3*num_cells];
    double *of_A = new double[2*num_surfaces + num_cells];
    readfile = fread(of_b, num_cells * 3 * sizeof(double), 1, fp);
    readfile = fread(of_A, (2*num_surfaces + num_cells) * sizeof(double), 1, fp);

    std::vector<double> h_A_of_init_vec(num_cells+2*num_surfaces);
    std::copy(of_A, of_A + num_cells+2*num_surfaces, h_A_of_init_vec.begin());

    std::vector<double> h_A_of_vec_1mtx(2*num_surfaces + num_cells, 0);
    for (int i = 0; i < 2*num_surfaces + num_cells; i++)
    {
        h_A_of_vec_1mtx[i] = h_A_of_init_vec[tmpPermutatedList[i]];
    }

    std::vector<double> h_A_of_vec((2*num_surfaces + num_cells)*3);
    for (int i =0; i < 3; i ++)
    {
        std::copy(h_A_of_vec_1mtx.begin(), h_A_of_vec_1mtx.end(), h_A_of_vec.begin()+i*(2*num_surfaces + num_cells));
    }

    // b
    std::vector<double> h_b_of_init_vec(3*num_cells);
    std::copy(of_b, of_b + 3*num_cells, h_b_of_init_vec.begin());
    std::vector<double> h_b_of_vec;
    for (int i = 0; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_y
    for (int i = 1; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_z
    for (int i = 2; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }

    if (print)
    {
        for (int i = 0; i < (2*num_surfaces + num_cells); i++)
            fprintf(stderr, "h_A_of_vec_1mtx[%d]: %.15lf\n", i, h_A_of_vec_1mtx[i]);
        for (int i = 0; i < 3*num_cells; i++)
            fprintf(stderr, "h_b_of_vec[%d]: %.15lf\n", i, h_b_of_vec[i]);
    }

    // check
    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual(2*num_surfaces + num_cells, h_A_of_vec_1mtx.data(), h_A_csr, 1e-5);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(3*num_cells, h_b_of_vec.data(), h_b, 1e-5);
}

void dfMatrix::solve()
{
    // for (size_t i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_velocity_old[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_velocity_old[3*i], 
    //     h_velocity_old[3*i + 1], h_velocity_old[3*i + 2]);
    // constructor AmgXSolver at first interation
    // Synchronize stream
    // checkCudaErrors(cudaMemcpyAsync(h_b, d_b, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaMemcpyAsync(h_psi, d_psi, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
    printf("CPU Time (copy&permutate)  = %.6lf s\n", time_monitor_CPU);
    printf("GPU Time                   = %.6lf s\n", time_monitor_GPU);
    time_monitor_CPU = 0;
    time_monitor_GPU = 0;

    // nvtxRangePush("solve");

    int nNz = num_cells + num_faces; // matrix entries
    if (num_iteration == 0) // first interation
    {
        printf("Initializing AmgX Linear Solver\n");
        UxSolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr);
        UySolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr + nNz);
        UzSolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr + 2*nNz);
    }
    else
    {
        UxSolver->updateOperator(num_cells, nNz, d_A_csr);
        UySolver->updateOperator(num_cells, nNz, d_A_csr + nNz);
        UzSolver->updateOperator(num_cells, nNz, d_A_csr + 2*nNz);
    }
    UxSolver->solve(num_cells, d_psi, d_b);
    UySolver->solve(num_cells, d_psi + num_cells, d_b + num_cells);
    UzSolver->solve(num_cells, d_psi + 2*num_cells, d_b + 2*num_cells);
    num_iteration ++;

    checkCudaErrors(cudaMemcpyAsync(h_psi, d_psi, cell_vec_bytes, cudaMemcpyDeviceToHost));
    // for (size_t i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_velocity_after[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_psi[i], 
    //     h_psi[num_cells + i], h_psi[num_cells*2 + i]);
}
void dfMatrix::updatePsi(double* Psi)
{
    for (size_t i = 0; i < num_cells; i++)
    {
        Psi[i*3] = h_psi[i];
        Psi[i*3 + 1] = h_psi[num_cells + i];
        Psi[i*3 + 2] = h_psi[num_cells*2 + i];
    }
}
void dfMatrix::fvc_grad_vector(std::vector<double> tmp_boundary_gradU, const double *ref_gradU)
{
    double* h_grad = new double[num_cells * 9];
    double* h_grad_boundary = new double[num_boundary_faces * 9];
    double* h_grad_boundary_init = new double[num_boundary_cells * 9];

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 102400);
    fvc_grad_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
          d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
          d_face_vector, d_velocity_old, d_weight, d_volume, d_grad);
    checkCudaErrors(cudaStreamSynchronize(stream));

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
          d_boundary_cell_offset, d_boundary_cell_id, d_boundary_face_vector, d_boundary_velocity, d_volume, 
          d_grad, d_grad_boundary_init);
    
    double* h_velocity_test = new double[8*3], *h_face_vector_test = new double[8*3];
    cudaMemcpy(h_velocity_test, d_boundary_velocity, 8*3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_face_vector_test, d_boundary_face_vector, 8*3*sizeof(double), cudaMemcpyDeviceToHost);

    correct_boundary_conditions<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
          d_boundary_cell_offset, d_boundary_cell_id, d_boundary_face_vector, d_boundary_face, 
          d_grad_boundary_init, d_grad_boundary);

    // checkBoundaryValue
    std::vector<double> boundary_gradU(num_boundary_faces*9);
    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundary_gradU[9*i] = tmp_boundary_gradU[9*boundPermutationList[i]];
        boundary_gradU[9*i+1] = tmp_boundary_gradU[9*boundPermutationList[i]+1];
        boundary_gradU[9*i+2] = tmp_boundary_gradU[9*boundPermutationList[i]+2];
        boundary_gradU[9*i+3] = tmp_boundary_gradU[9*boundPermutationList[i]+3];
        boundary_gradU[9*i+4] = tmp_boundary_gradU[9*boundPermutationList[i]+4];
        boundary_gradU[9*i+5] = tmp_boundary_gradU[9*boundPermutationList[i]+5];
        boundary_gradU[9*i+6] = tmp_boundary_gradU[9*boundPermutationList[i]+6];
        boundary_gradU[9*i+7] = tmp_boundary_gradU[9*boundPermutationList[i]+7];
        boundary_gradU[9*i+8] = tmp_boundary_gradU[9*boundPermutationList[i]+8];
    }
    // for (int i = 0; i < num_boundary_faces; i++)
    //     fprintf(stderr, "of_boundary_gradU[%d]: (%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf)\n", i, 
    //     boundary_gradU[i * 9 + 0], boundary_gradU[i * 9 + 1], boundary_gradU[i * 9 + 2], 
    //     boundary_gradU[i * 9 + 3], boundary_gradU[i * 9 + 4], boundary_gradU[i * 9 + 5],
    //     boundary_gradU[i * 9 + 6], boundary_gradU[i * 9 + 7], boundary_gradU[i * 9 + 8]);
    
    checkCudaErrors(cudaMemcpy(h_grad_boundary, d_grad_boundary, num_boundary_faces * 9 * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_boundary_faces; i++)
    //     fprintf(stderr, "df_boundary_gradU[%d]: (%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf)\n", i, 
    //     h_grad_boundary[i * 9 + 0], h_grad_boundary[i * 9 + 1], h_grad_boundary[i * 9 + 2], 
    //     h_grad_boundary[i * 9 + 3], h_grad_boundary[i * 9 + 4], h_grad_boundary[i * 9 + 5],
    //     h_grad_boundary[i * 9 + 6], h_grad_boundary[i * 9 + 7], h_grad_boundary[i * 9 + 8]);
    
    checkCudaErrors(cudaMemcpy(h_grad_boundary_init, d_grad_boundary_init, num_boundary_cells * 9 * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_boundary_faces; i++)
    //     fprintf(stderr, "df_boundary_gradU_init[%d]: (%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf)\n", i, 
    //     h_grad_boundary_init[i * 9 + 0], h_grad_boundary_init[i * 9 + 1], h_grad_boundary_init[i * 9 + 2], 
    //     h_grad_boundary_init[i * 9 + 3], h_grad_boundary_init[i * 9 + 4], h_grad_boundary_init[i * 9 + 5],
    //     h_grad_boundary_init[i * 9 + 6], h_grad_boundary_init[i * 9 + 7], h_grad_boundary_init[i * 9 + 8]);
    fprintf(stderr, "error of fvc::grad(U)'s boundary\n");
    checkVectorEqual(num_boundary_cells * 9, boundary_gradU.data(), h_grad_boundary, 1e-5);


    checkCudaErrors(cudaMemcpy(h_grad, d_grad, num_cells * 9 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_cells; i++)
        fprintf(stderr, "fvc_grad_tensor[%d]: (%.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf, %.15lf)\n", i, 
        h_grad[i * 9 + 0], h_grad[i * 9 + 1], h_grad[i * 9 + 2], 
        h_grad[i * 9 + 3], h_grad[i * 9 + 4], h_grad[i * 9 + 5],
        h_grad[i * 9 + 6], h_grad[i * 9 + 7], h_grad[i * 9 + 8]);
    
    // check grad value
    fprintf(stderr, "error of fvc::grad(U)\n");
    checkVectorEqual(num_cells * 9, ref_gradU, h_grad, 1e-10);

}
void dfMatrix::divDevRhoReff(const double *ref_grad_T)
{
    // dev2(T(fvc::grad(U)))
    double* h_grad = new double[num_cells * 9];
    double* h_grad_boundary = new double[num_boundary_faces * 9];

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    dev2_t_tensor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, d_grad);
     
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    dev2_t_tensor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces, d_grad_boundary);

    checkCudaErrors(cudaMemcpy(h_grad, d_grad, num_cells * 9 * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_cells; i++)
    //     fprintf(stderr, "fvc_grad_tensor_T[%d]: (%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf)\n", i, 
    //     h_grad[i * 9 + 0], h_grad[i * 9 + 1], h_grad[i * 9 + 2], 
    //     h_grad[i * 9 + 3], h_grad[i * 9 + 4], h_grad[i * 9 + 5],
    //     h_grad[i * 9 + 6], h_grad[i * 9 + 7], h_grad[i * 9 + 8]);
    fprintf(stderr, "error of dev2(T(fvc::grad(U)))\n");
    checkVectorEqual(num_cells * 9, ref_grad_T, h_grad, 1);
    
    checkCudaErrors(cudaMemcpy(h_grad_boundary, d_grad_boundary, num_boundary_faces * 9 * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < num_boundary_faces; i++)
    //     fprintf(stderr, "df_boundary_gradU_T[%d]: (%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf)\n", i, 
    //     h_grad_boundary[i * 9 + 0], h_grad_boundary[i * 9 + 1], h_grad_boundary[i * 9 + 2], 
    //     h_grad_boundary[i * 9 + 3], h_grad_boundary[i * 9 + 4], h_grad_boundary[i * 9 + 5],
    //     h_grad_boundary[i * 9 + 6], h_grad_boundary[i * 9 + 7], h_grad_boundary[i * 9 + 8]);
}

void dfMatrix::fvc_div_tensor(const double *nuEff, double* of_ref_boundary, double *of_ref_grad, double *of_ref_div)
{
    double* d_b_test = nullptr, *h_b_test = new double[3*num_cells];
    double* d_grad_test = nullptr, *h_grad_test = new double[9*num_cells];
    double* d_grad_boundary_test = nullptr,*h_grad_boundary_test = new double[num_boundary_faces * 9];
    double* h_weight_test = new double[num_faces];

    checkCudaErrors(cudaMalloc((void**)&d_b_test, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_grad_test, 9*num_cells*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_grad_boundary_test, 9*num_boundary_faces*sizeof(double)));

    checkCudaErrors(cudaMemcpyAsync(d_nuEff, nuEff, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemsetAsync(d_b_test, 0, cell_vec_bytes, stream));

    // launch cuda kernel
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_tensor_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
          d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
          d_nuEff, d_rho_new, d_face_vector, d_grad, d_weight, d_volume, 1., d_b_test, d_b_test, d_grad_test);
    checkCudaErrors(cudaStreamSynchronize(stream));

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_tensor_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
          d_boundary_cell_offset, d_boundary_cell_id,
          d_boundary_nuEff, d_boundary_rho, d_boundary_face_vector, d_grad_boundary, d_volume, 1., d_b_test, d_b_test, d_grad_boundary_test);

    checkCudaErrors(cudaMemcpyAsync(h_b_test, d_b_test, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_grad_test, d_grad_test, 9*num_cells*sizeof(double), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_grad_boundary_test, d_grad_boundary_test, 9*num_boundary_faces*sizeof(double), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_weight_test, d_weight, num_faces*sizeof(double), cudaMemcpyDeviceToHost, stream));
    
    for (int i = 0; i < num_cells; i++)
          fprintf(stderr, "h_b[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_b_test[i], h_b_test[num_cells + i],
          h_b_test[2*num_cells + i]);
    
    fprintf(stderr,"error of fvc_div\n");
    std::vector<double> of_ref_div_init_vec(3*num_cells);
    std::copy(of_ref_div, of_ref_div + 3*num_cells, of_ref_div_init_vec.begin());
    std::vector<double> of_ref_div_vec;
    for (int i = 0; i < 3*num_cells; i+=3)
    {
        of_ref_div_vec.push_back(of_ref_div_init_vec[i]);
    }
    // fill RHS_y
    for (int i = 1; i < 3*num_cells; i+=3)
    {
        of_ref_div_vec.push_back(of_ref_div_init_vec[i]);
    }
    // fill RHS_z
    for (int i = 2; i < 3*num_cells; i+=3)
    {
        of_ref_div_vec.push_back(of_ref_div_init_vec[i]);
    }
    checkVectorEqual(3*num_cells, of_ref_div_vec.data(), h_b_test, 0.1);

    for (int i = 0; i < num_cells; i++)
        fprintf(stderr, "fvc_grad_test[%d]: (%.7lf, %.7lf, %.7lf, %.7lf, %.7lf, %.7lf, %.7lf, %.7lf, %.7lf)\n", i, 
        h_grad_test[i * 9 + 0], h_grad_test[i * 9 + 1], h_grad_test[i * 9 + 2], 
        h_grad_test[i * 9 + 3], h_grad_test[i * 9 + 4], h_grad_test[i * 9 + 5],
        h_grad_test[i * 9 + 6], h_grad_test[i * 9 + 7], h_grad_test[i * 9 + 8]);
    fprintf(stderr,"error of 'this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))'\n");
    checkVectorEqual(9*num_cells, of_ref_grad, h_grad_test, 0.1);

    // for (int i = 0; i < num_boundary_faces; i++)
    //     fprintf(stderr, "fvc_grad_boundary_test[%d]: (%.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf)\n", i, 
    //     h_grad_boundary_test[i * 9 + 0], h_grad_boundary_test[i * 9 + 1], h_grad_boundary_test[i * 9 + 2],
    //     h_grad_boundary_test[i * 9 + 3], h_grad_boundary_test[i * 9 + 4], h_grad_boundary_test[i * 9 + 5],
    //     h_grad_boundary_test[i * 9 + 6], h_grad_boundary_test[i * 9 + 7], h_grad_boundary_test[i * 9 + 8]);

    std::vector<double> boundary_coeff_gradU(num_boundary_faces*9);
    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundary_coeff_gradU[9*i] = of_ref_boundary[9*boundPermutationList[i]];
        boundary_coeff_gradU[9*i+1] = of_ref_boundary[9*boundPermutationList[i]+1];
        boundary_coeff_gradU[9*i+2] = of_ref_boundary[9*boundPermutationList[i]+2];
        boundary_coeff_gradU[9*i+3] = of_ref_boundary[9*boundPermutationList[i]+3];
        boundary_coeff_gradU[9*i+4] = of_ref_boundary[9*boundPermutationList[i]+4];
        boundary_coeff_gradU[9*i+5] = of_ref_boundary[9*boundPermutationList[i]+5];
        boundary_coeff_gradU[9*i+6] = of_ref_boundary[9*boundPermutationList[i]+6];
        boundary_coeff_gradU[9*i+7] = of_ref_boundary[9*boundPermutationList[i]+7];
        boundary_coeff_gradU[9*i+8] = of_ref_boundary[9*boundPermutationList[i]+8];
    }
    for (int i = 0; i < num_boundary_faces; i++)
        printf("of_boundary_gradU[%d]: (%.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf, %.9lf)\n", i, 
        boundary_coeff_gradU[i * 9 + 0], boundary_coeff_gradU[i * 9 + 1], boundary_coeff_gradU[i * 9 + 2], 
        boundary_coeff_gradU[i * 9 + 3], boundary_coeff_gradU[i * 9 + 4], boundary_coeff_gradU[i * 9 + 5],
        boundary_coeff_gradU[i * 9 + 6], boundary_coeff_gradU[i * 9 + 7], boundary_coeff_gradU[i * 9 + 8]);
    
    fprintf(stderr,"check boundary value\n");
    checkVectorEqual(9*num_boundary_faces, boundary_coeff_gradU.data(), h_grad_boundary_test, 1e-5);

}

void dfMatrix::fvm_laplacian()
{
    double* d_b_test = nullptr, *h_b_test = new double[3*num_cells];
    double* d_A_csr_test = nullptr, *h_A_csr_test = new double[(2*num_surfaces + num_cells)*3];

    checkCudaErrors(cudaMalloc((void**)&d_b_test, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A_csr_test, csr_value_vec_bytes));

    checkCudaErrors(cudaMemsetAsync(d_A_csr_test, 0, csr_value_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_b_test, 0, cell_vec_bytes, stream));
    
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_uncorrected_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
          d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index, d_rho_new, d_nuEff, d_weight, d_face, d_deltaCoeffs, -1., d_A_csr, d_A_csr);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_uncorrected_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
          d_A_csr_row_index, d_A_csr_diag_index, d_boundary_cell_offset, d_boundary_cell_id, 
          d_boundary_nuEff, d_boundary_rho, d_boundary_face, d_laplac_internal_coeffs, d_laplac_boundary_coeffs,
          -1., d_A_csr, d_b, d_A_csr, d_b);
    
    checkCudaErrors(cudaMemcpy(h_A_csr_test, d_A_csr_test, csr_value_vec_bytes, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < (2*num_surfaces + num_cells)*3; i++)
    //         fprintf(stderr, "h_A_csr_test[%d]: %.15lf\n", i, h_A_csr_test[i]);
    // for (int i = 0; i < num_cells*3; i++)
    //         fprintf(stderr, "h_b_test[%d]: %.15lf\n", i, h_b_test[i]);
}

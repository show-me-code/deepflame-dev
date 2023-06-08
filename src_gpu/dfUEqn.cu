#include "dfUEqn.H"

// kernel functions
__global__ void fvm_ddt_kernel(int num_cells, int num_faces, const double rdelta_t,
                               const int *csr_row_index, const int *csr_diag_index,
                               const double *rho_old, const double *rho_new, const double *volume, const double *velocity_old,
                               const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output, double *psi)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

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
                                 const int *csr_row_index, const int *csr_diag_index,
                                 const double *weight, const double *phi,
                                 const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;

    double div_diag = 0;
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
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
        if (inner_index > diag_index)
        {
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
                                 const int *csr_row_index, const int *csr_diag_index,
                                 const int *boundary_cell_offset, const int *boundary_cell_id,
                                 const double *internal_coeffs, const double *boundary_coeffs,
                                 const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output,
                                 double *ueqn_internal_coeffs, double *ueqn_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

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
    for (int i = 0; i < loop_size; i++)
    {
        internal_coeffs_x += internal_coeffs[(cell_offset + i) * 3 + 0];
        internal_coeffs_y += internal_coeffs[(cell_offset + i) * 3 + 1];
        internal_coeffs_z += internal_coeffs[(cell_offset + i) * 3 + 2];
        boundary_coeffs_x += boundary_coeffs[(cell_offset + i) * 3 + 0];
        boundary_coeffs_y += boundary_coeffs[(cell_offset + i) * 3 + 1];
        boundary_coeffs_z += boundary_coeffs[(cell_offset + i) * 3 + 2];
    }
    ueqn_internal_coeffs[cell_index * 3 + 0] = internal_coeffs_x;
    ueqn_internal_coeffs[cell_index * 3 + 1] = internal_coeffs_y;
    ueqn_internal_coeffs[cell_index * 3 + 2] = internal_coeffs_z;
    ueqn_boundary_coeffs[cell_index * 3 + 0] = boundary_coeffs_x;
    ueqn_boundary_coeffs[cell_index * 3 + 1] = boundary_coeffs_y;
    ueqn_boundary_coeffs[cell_index * 3 + 2] = boundary_coeffs_z;

    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + internal_coeffs_x;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + internal_coeffs_y;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + internal_coeffs_z;
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + boundary_coeffs_x;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + boundary_coeffs_y;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + boundary_coeffs_z;
}

__global__ void fvc_grad_internal_face(int num_cells,
                                       const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                       const double *face_vector, const double *weight, const double *pressure,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

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
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
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
        if (inner_index > diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = weight[neighbor_index];
            double sfx = face_vector[neighbor_index * 3 + 0];
            double sfy = face_vector[neighbor_index * 3 + 1];
            double sfz = face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = w * own_cell_p + (1 - w) * neighbor_cell_p;
            grad_bx_upp += face_p * sfx;
            grad_by_upp += face_p * sfy;
            grad_bz_upp += face_p * sfz;
        }
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] - grad_bx_low - grad_bx_upp;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] - grad_by_low - grad_by_upp;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] - grad_bz_low - grad_bz_upp;
}

__global__ void fvc_grad_boundary_face(int num_cells, int num_boundary_cells,
                                       const int *boundary_cell_offset, const int *boundary_cell_id,
                                       const double *boundary_face_vector, const double *boundary_pressure,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // compute boundary gradient
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double sfx = boundary_face_vector[i * 3 + 0];
        double sfy = boundary_face_vector[i * 3 + 1];
        double sfz = boundary_face_vector[i * 3 + 2];
        double face_p = boundary_pressure[i];
        grad_bx += face_p * sfx;
        grad_by += face_p * sfy;
        grad_bz += face_p * sfz;
    }

    //// correct the boundary gradient
    // double nx = boundary_face_vector[face_index * 3 + 0] / magSf[face_index];
    // double ny = boundary_face_vector[face_index * 3 + 1] / magSf[face_index];
    // double nz = boundary_face_vector[face_index * 3 + 2] / magSf[face_index];
    // double sn_grad = 0;
    // double grad_correction = sn_grad * volume[cell_index] - (nx * grad_bx + ny * grad_by + nz * grad_bz);
    // grad_bx += nx * grad_correction;
    // grad_by += ny * grad_correction;
    // grad_bz += nz * grad_correction;

    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] - grad_bx;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] - grad_by;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] - grad_bz;
}

__global__ void add_fvMatrix_kernel(int num_cells, int num_faces,
                                    const int *csr_row_index,
                                    const double *turbSrc_A, const double *turbSrc_b,
                                    const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

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
    if (index >= num_faces)
        return;

    int p = permedIndex[index];
    d_phi[index] = d_phi_init[p];
}

__global__ void boundaryPermutation(const int num_boundary_faces, const int *bouPermedIndex,
                                    const double *boundary_pressure_init, const double *boundary_velocity_init,
                                    double *boundary_pressure, double *boundary_velocity,
                                    double *boundary_nuEff_init, double *boundary_nuEff,
                                    double *boundary_rho_init, double *boundary_rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    int p = bouPermedIndex[index];
    boundary_velocity[3 * index] = boundary_velocity_init[3 * p];
    boundary_velocity[3 * index + 1] = boundary_velocity_init[3 * p + 1];
    boundary_velocity[3 * index + 2] = boundary_velocity_init[3 * p + 2];
    boundary_pressure[index] = boundary_pressure_init[p];
    boundary_rho[index] = boundary_rho_init[p];
    boundary_nuEff[index] = boundary_nuEff_init[p];
}

__global__ void fvc_grad_vector_internal(int num_cells,
                                         const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                         const double *sf, const double *vf, const double *tlambdas, const double *volume,
                                         double *grad)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

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
    for (int i = 0; i < diag_index; i++)
    {
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
    for (int i = diag_index + 1; i < row_elements; i++)
    {
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
                                         const int *boundary_cell_offset, const int *boundary_cell_id,
                                         const double *boundary_sf, const double *boundary_vf, const double *volume,
                                         double *grad, double *grad_boundary_init)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

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
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
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
                                            const int *boundary_cell_offset, const int *boundary_cell_id,
                                            const double *boundary_sf, const double *mag_sf,
                                            double *boundary_grad_init, double *boundary_grad)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

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

    for (int i = cell_offset; i < next_cell_offset; i++)
    {
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

__global__ void dev2_t_tensor(int num, double *tensor)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

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
                                        const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                        const double *scalar0, const double *scalar1,
                                        const double *sf, const double *vf, const double *tlambdas, const double *volume,
                                        const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double coeff_own = scalar0[index] * scalar1[index];

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
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double coeff_nei = scalar0[neighbor_cell_id] * scalar1[neighbor_cell_id];
        double w = tlambdas[neighbor_index];
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
        double face_xx = (1 - w) * own_vf_xx * coeff_own + w * neighbor_vf_xx * coeff_nei;
        double face_xy = (1 - w) * own_vf_xy * coeff_own + w * neighbor_vf_xy * coeff_nei;
        double face_xz = (1 - w) * own_vf_xz * coeff_own + w * neighbor_vf_xz * coeff_nei;
        double face_yx = (1 - w) * own_vf_yx * coeff_own + w * neighbor_vf_yx * coeff_nei;
        double face_yy = (1 - w) * own_vf_yy * coeff_own + w * neighbor_vf_yy * coeff_nei;
        double face_yz = (1 - w) * own_vf_yz * coeff_own + w * neighbor_vf_yz * coeff_nei;
        double face_zx = (1 - w) * own_vf_zx * coeff_own + w * neighbor_vf_zx * coeff_nei;
        double face_zy = (1 - w) * own_vf_zy * coeff_own + w * neighbor_vf_zy * coeff_nei;
        double face_zz = (1 - w) * own_vf_zz * coeff_own + w * neighbor_vf_zz * coeff_nei;
        sum_x -= sf_x * face_xx + sf_y * face_yx + sf_z * face_zx;
        sum_y -= sf_x * face_xy + sf_y * face_yy + sf_z * face_zy;
        sum_z -= sf_x * face_xz + sf_y * face_yz + sf_z * face_zz;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double coeff_nei = scalar0[neighbor_cell_id] * scalar1[neighbor_cell_id];
        double w = tlambdas[neighbor_index];
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
        double face_xx = w * own_vf_xx * coeff_own + (1 - w) * neighbor_vf_xx * coeff_nei;
        double face_xy = w * own_vf_xy * coeff_own + (1 - w) * neighbor_vf_xy * coeff_nei;
        double face_xz = w * own_vf_xz * coeff_own + (1 - w) * neighbor_vf_xz * coeff_nei;
        double face_yx = w * own_vf_yx * coeff_own + (1 - w) * neighbor_vf_yx * coeff_nei;
        double face_yy = w * own_vf_yy * coeff_own + (1 - w) * neighbor_vf_yy * coeff_nei;
        double face_yz = w * own_vf_yz * coeff_own + (1 - w) * neighbor_vf_yz * coeff_nei;
        double face_zx = w * own_vf_zx * coeff_own + (1 - w) * neighbor_vf_zx * coeff_nei;
        double face_zy = w * own_vf_zy * coeff_own + (1 - w) * neighbor_vf_zy * coeff_nei;
        double face_zz = w * own_vf_zz * coeff_own + (1 - w) * neighbor_vf_zz * coeff_nei;
        sum_x += sf_x * face_xx + sf_y * face_yx + sf_z * face_zx;
        sum_y += sf_x * face_xy + sf_y * face_yy + sf_z * face_zy;
        sum_z += sf_x * face_xz + sf_y * face_yz + sf_z * face_zz;
    }
    double vol = volume[index];

    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + sum_x * sign;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + sum_y * sign;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + sum_z * sign;
}

__global__ void fvc_div_tensor_boundary(int num_cells, int num_boundary_cells,
                                        const int *boundary_cell_offset, const int *boundary_cell_id,
                                        const double *boundary_scalar0, const double *boundary_scalar1,
                                        const double *boundary_sf, const double *boundary_vf, const double *volume,
                                        const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

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
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
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
    }
    double vol = volume[cell_index];
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + sum_x * sign;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + sum_y * sign;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + sum_z * sign;
}

__global__ void fvm_laplacian_uncorrected_vector_internal(int num_cells, int num_faces,
                                                          const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                                          const double *scalar0, const double *scalar1, const double *weight,
                                                          const double *magsf, const double *distance,
                                                          const double sign, const double *A_csr_input, double *A_csr_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;

    double own_scalar0 = scalar0[index];
    double own_scalar1 = scalar1[index];
    double own_coeff = own_scalar0 * own_scalar1;

    // fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm.negSumDiag();
    double sum_diag = 0;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_scalar0 = scalar0[neighbor_cell_id];
        double nei_scalar1 = scalar1[neighbor_cell_id];
        double nei_coeff = nei_scalar0 * nei_scalar1;
        double gamma = w * (nei_coeff - own_coeff) + own_coeff;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[csr_dim * 0 + row_index + i] = A_csr_input[csr_dim * 0 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 1 + row_index + i] = A_csr_input[csr_dim * 1 + row_index + i] + coeff * sign;
        A_csr_output[csr_dim * 2 + row_index + i] = A_csr_input[csr_dim * 2 + row_index + i] + coeff * sign;

        sum_diag += (-coeff);
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_scalar0 = scalar0[neighbor_cell_id];
        double nei_scalar1 = scalar1[neighbor_cell_id];
        double nei_coeff = nei_scalar0 * nei_scalar1;
        double gamma = w * (own_coeff - nei_coeff) + nei_coeff;
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
                                                          const int *csr_row_index, const int *csr_diag_index,
                                                          const int *boundary_cell_offset, const int *boundary_cell_id,
                                                          const double *boundary_scalar0, const double *boundary_scalar1,
                                                          const double *boundary_magsf, const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
                                                          const double sign, const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output,
                                                          double *ueqn_internal_coeffs, double *ueqn_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

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
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double gamma = boundary_scalar0[i] * boundary_scalar1[i];
        double gamma_magsf = gamma * boundary_magsf[i];
        internal_coeffs_x += gamma_magsf * gradient_internal_coeffs[i * 3 + 0];
        internal_coeffs_y += gamma_magsf * gradient_internal_coeffs[i * 3 + 1];
        internal_coeffs_z += gamma_magsf * gradient_internal_coeffs[i * 3 + 2];
        boundary_coeffs_x += gamma_magsf * gradient_boundary_coeffs[i * 3 + 0];
        boundary_coeffs_y += gamma_magsf * gradient_boundary_coeffs[i * 3 + 1];
        boundary_coeffs_z += gamma_magsf * gradient_boundary_coeffs[i * 3 + 2];
    }

    ueqn_internal_coeffs[cell_index * 3 + 0] += internal_coeffs_x * sign;
    ueqn_internal_coeffs[cell_index * 3 + 1] += internal_coeffs_y * sign;
    ueqn_internal_coeffs[cell_index * 3 + 2] += internal_coeffs_z * sign;
    ueqn_boundary_coeffs[cell_index * 3 + 0] += boundary_coeffs_x * sign;
    ueqn_boundary_coeffs[cell_index * 3 + 1] += boundary_coeffs_y * sign;
    ueqn_boundary_coeffs[cell_index * 3 + 2] += boundary_coeffs_z * sign;

    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + internal_coeffs_x * sign;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + internal_coeffs_y * sign;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + internal_coeffs_z * sign;
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + boundary_coeffs_x * sign;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + boundary_coeffs_y * sign;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + boundary_coeffs_z * sign;
}

__global__ void addBoundaryDiag(int num_cells, int num_boundary_cells,
                                const int *csr_row_index, const int *csr_diag_index,
                                const int *boundary_cell_offset, const int *boundary_cell_id,
                                const double *ueqn_internal_coeffs, const double *ueqn_boundary_coeffs,
                                const double *psi, double *H)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // addBoundaryDiag(boundaryDiagCmpt, cmpt); // add internal coeffs
    // boundaryDiagCmpt.negate();
    double internal_x = ueqn_internal_coeffs[cell_index * 3 + 0];
    double internal_y = ueqn_internal_coeffs[cell_index * 3 + 1];
    double internal_z = ueqn_internal_coeffs[cell_index * 3 + 2];

    // addCmptAvBoundaryDiag(boundaryDiagCmpt);
    double ave_internal = (internal_x + internal_y + internal_z) / 3;

    H[num_cells * 0 + cell_index] = (-internal_x + ave_internal) * psi[num_cells * 0 + cell_index];
    H[num_cells * 1 + cell_index] = (-internal_y + ave_internal) * psi[num_cells * 1 + cell_index];
    H[num_cells * 2 + cell_index] = (-internal_z + ave_internal) * psi[num_cells * 2 + cell_index];
}

__global__ void lduMatrix_H(int num_cells,
                            const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                            const double *volume, const double *psi, const double *A_csr, const double *b,
                            const double *ueqn_boundary_coeffs, double *H)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double APsi_x = 0.;
    double APsi_y = 0.;
    double APsi_z = 0.;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_cell_id = csr_col_index[i + row_index];
        APsi_x += A_csr[row_index + i] * psi[num_cells * 0 + neighbor_cell_id];
        APsi_y += A_csr[row_index + i] * psi[num_cells * 1 + neighbor_cell_id];
        APsi_z += A_csr[row_index + i] * psi[num_cells * 2 + neighbor_cell_id];
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_cell_id = csr_col_index[i + row_index];
        APsi_x += A_csr[row_index + i] * psi[num_cells * 0 + neighbor_cell_id];
        APsi_y += A_csr[row_index + i] * psi[num_cells * 1 + neighbor_cell_id];
        APsi_z += A_csr[row_index + i] * psi[num_cells * 2 + neighbor_cell_id];
    }

    H[num_cells * 0 + index] = H[num_cells * 0 + index] - APsi_x + b[num_cells * 0 + index];
    H[num_cells * 1 + index] = H[num_cells * 1 + index] - APsi_y + b[num_cells * 1 + index];
    H[num_cells * 2 + index] = H[num_cells * 2 + index] - APsi_z + b[num_cells * 2 + index];

    double vol = volume[index];
    H[num_cells * 0 + index] = H[num_cells * 0 + index] / vol;
    H[num_cells * 1 + index] = H[num_cells * 1 + index] / vol;
    H[num_cells * 2 + index] = H[num_cells * 2 + index] / vol;
}

__global__ void addBoundarySource(int num_cells, int num_boundary_cells,
                                  const int *csr_row_index, const int *csr_diag_index,
                                  const int *boundary_cell_offset, const int *boundary_cell_id,
                                  const double *ueqn_internal_coeffs, const double *ueqn_boundary_coeffs,
                                  const double *volume, double *H)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int cell_index = boundary_cell_id[cell_offset];

    double vol = volume[index];

    H[num_cells * 0 + index] = H[num_cells * 0 + index] + ueqn_boundary_coeffs[cell_index * 3 + 0] / vol;
    H[num_cells * 1 + index] = H[num_cells * 1 + index] + ueqn_boundary_coeffs[cell_index * 3 + 1] / vol;
    H[num_cells * 2 + index] = H[num_cells * 2 + index] + ueqn_boundary_coeffs[cell_index * 3 + 2] / vol;
}

__global__ void addAveInternaltoDiag(int num_cells, int num_boundary_cells,
                                     const int *csr_row_index, const int *csr_diag_index,
                                     const int *boundary_cell_offset, const int *boundary_cell_id,
                                     const double *ueqn_internal_coeffs, const double *ueqn_boundary_coeffs, double *A)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    double internal_x = ueqn_internal_coeffs[cell_index * 3 + 0];
    double internal_y = ueqn_internal_coeffs[cell_index * 3 + 1];
    double internal_z = ueqn_internal_coeffs[cell_index * 3 + 2];

    double ave_internal = (internal_x + internal_y + internal_z) / 3;

    A[cell_index] = ave_internal;
}

__global__ void addDiagDivVolume(int num_cells, const int *csr_row_index,
                                 const int *csr_diag_index, const double *A_csr, const double *volume,
                                 double *ueqn_internal_coeffs, const double *A_input, double *A_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    int row_index = csr_row_index[index];
    int diag_index = csr_diag_index[index];
    int csr_index = row_index + diag_index;

    double vol = volume[index];

    A_output[index] = (A_input[index] + A_csr[csr_index] - ueqn_internal_coeffs[index * 3]) / vol;
}

__global__ void ueqn_update_BoundaryCoeffs_kernel(int num_boundary_faces, const double *boundary_phi, double *internal_coeffs,
                                                  double *boundary_coeffs, double *laplac_internal_coeffs,
                                                  double *laplac_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    // zeroGradient
    double valueInternalCoeffs = 1.;
    double valueBoundaryCoeffs = 0.;
    double gradientInternalCoeffs = 0.;
    double gradientBoundaryCoeffs = 0.;

    internal_coeffs[index * 3 + 0] = boundary_phi[index] * valueInternalCoeffs;
    internal_coeffs[index * 3 + 1] = boundary_phi[index] * valueInternalCoeffs;
    internal_coeffs[index * 3 + 2] = boundary_phi[index] * valueInternalCoeffs;
    boundary_coeffs[index * 3 + 0] = -boundary_phi[index] * valueBoundaryCoeffs;
    boundary_coeffs[index * 3 + 1] = -boundary_phi[index] * valueBoundaryCoeffs;
    boundary_coeffs[index * 3 + 2] = -boundary_phi[index] * valueBoundaryCoeffs;
    laplac_internal_coeffs[index * 3 + 0] = gradientInternalCoeffs;
    laplac_internal_coeffs[index * 3 + 1] = gradientInternalCoeffs;
    laplac_internal_coeffs[index * 3 + 2] = gradientInternalCoeffs;
    laplac_boundary_coeffs[index * 3 + 0] = gradientBoundaryCoeffs;
    laplac_boundary_coeffs[index * 3 + 1] = gradientBoundaryCoeffs;
    laplac_boundary_coeffs[index * 3 + 2] = gradientBoundaryCoeffs;
}

// constructor
dfUEqn::dfUEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile)
    : dataBase_(dataBase)
{
    stream = dataBase_.stream;

    UxSolver = new AmgXSolver(modeStr, cfgFile);
    UySolver = new AmgXSolver(modeStr, cfgFile);
    UzSolver = new AmgXSolver(modeStr, cfgFile);

    num_cells = dataBase_.num_cells;
    cell_bytes = dataBase_.cell_bytes;
    num_faces = dataBase_.num_faces;
    cell_vec_bytes = dataBase_.cell_vec_bytes;
    csr_value_vec_bytes = dataBase_.csr_value_vec_bytes;
    num_boundary_cells = dataBase_.num_boundary_cells;
    num_surfaces = dataBase_.num_surfaces;

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    h_A_csr = new double[(num_cells + num_faces) * 3];
    h_b = new double[num_cells * 3];
    cudaMallocHost(&h_psi, cell_vec_bytes);
    cudaMallocHost(&h_H, cell_vec_bytes);
    cudaMallocHost(&h_A, cell_bytes);

    checkCudaErrors(cudaMalloc((void **)&d_A_csr, csr_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_psi, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_H, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_A, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_ueqn_internal_coeffs, cell_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_ueqn_boundary_coeffs, cell_vec_bytes));
}

void dfUEqn::fvm_ddt(double *vector_old)
{
    // Copy the host input array in host memory to the device input array in device memory
    clock_t start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_velocity_old, vector_old, cell_vec_bytes, cudaMemcpyHostToDevice, stream));
    clock_t end = std::clock();
    time_monitor_GPU_memcpy += double(end - start) / double(CLOCKS_PER_SEC);

    // launch cuda kernel
    start = std::clock();
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_ddt_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, dataBase_.rdelta_t,
                                                                      d_A_csr_row_index, d_A_csr_diag_index,
                                                                      dataBase_.d_rho_old, dataBase_.d_rho_new, dataBase_.d_volume, dataBase_.d_velocity_old, d_A_csr, d_b, d_A_csr, d_b, d_psi);
    end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::fvm_div(double *boundary_pressure_init, double *boundary_velocity_init,
                     double *boundary_nuEff_init, double *boundary_rho_init)
{
    // copy and permutate boundary variable
    clock_t start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_velocity_init, boundary_velocity_init, dataBase_.boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_pressure_init, boundary_pressure_init, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_nuEff_init, boundary_nuEff_init, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_rho_init, boundary_rho_init, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    clock_t end = std::clock();
    time_monitor_GPU_memcpy += double(end - start) / double(CLOCKS_PER_SEC);

    start = std::clock();
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (dataBase_.num_boundary_faces + threads_per_block - 1) / threads_per_block;
    boundaryPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_faces, dataBase_.d_bouPermedIndex, dataBase_.d_boundary_pressure_init,
                                                                           dataBase_.d_boundary_velocity_init, dataBase_.d_boundary_pressure, dataBase_.d_boundary_velocity, 
                                                                           dataBase_.d_boundary_nuEff_init, dataBase_.d_boundary_nuEff, dataBase_.d_boundary_rho_init, dataBase_.d_boundary_rho);

    // launch cuda kernel
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_div_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
                                                                        d_A_csr_row_index, d_A_csr_diag_index,
                                                                        dataBase_.d_weight, dataBase_.d_phi, d_A_csr, d_b, d_A_csr, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvm_div_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
                                                                        d_A_csr_row_index, d_A_csr_diag_index,
                                                                        dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                        dataBase_.d_internal_coeffs, dataBase_.d_boundary_coeffs, d_A_csr, d_b, d_A_csr, d_b,
                                                                        d_ueqn_internal_coeffs, d_ueqn_boundary_coeffs);
    end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::fvc_grad(double *pressure)
{
    // Copy the host input array in host memory to the device input array in device memory
    clock_t start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_pressure, pressure, cell_bytes, cudaMemcpyHostToDevice, stream));
    clock_t end = std::clock();
    time_monitor_GPU_memcpy += double(end - start) / double(CLOCKS_PER_SEC);

    // launch cuda kernel
    start = std::clock();
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_internal_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                              d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                              dataBase_.d_face_vector, dataBase_.d_weight, dataBase_.d_pressure, d_b, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_boundary_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
                                                                              dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                              dataBase_.d_boundary_face_vector, dataBase_.d_boundary_pressure, d_b, d_b);
    end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::fvc_grad_vector()
{
    clock_t start = std::clock();
    // launch CUDA kernal
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                                dataBase_.d_face_vector, dataBase_.d_velocity_old, dataBase_.d_weight, dataBase_.d_volume, dataBase_.d_grad);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
                                                                                dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id, dataBase_.d_boundary_face_vector, dataBase_.d_boundary_velocity,
                                                                                dataBase_.d_volume, dataBase_.d_grad, dataBase_.d_grad_boundary_init);

    correct_boundary_conditions<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                                                                                   dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id, dataBase_.d_boundary_face_vector, dataBase_.d_boundary_face,
                                                                                   dataBase_.d_grad_boundary_init, dataBase_.d_grad_boundary);
    clock_t end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::dev2T()
{
    clock_t start = std::clock();
    // launch CUDA kernal
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    dev2_t_tensor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.d_grad);

    blocks_per_grid = (dataBase_.num_boundary_faces + threads_per_block - 1) / threads_per_block;
    dev2_t_tensor<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_faces, dataBase_.d_grad_boundary);
    clock_t end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::fvc_div_tensor(const double *nuEff)
{
    clock_t start = std::clock();
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_nuEff, nuEff, cell_bytes, cudaMemcpyHostToDevice, stream));
    clock_t end = std::clock();
    time_monitor_GPU_memcpy += double(end - start) / double(CLOCKS_PER_SEC);

    // launch cuda kernel
    start = std::clock();
    size_t threads_per_block = 512;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_tensor_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                               d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                               dataBase_.d_nuEff, dataBase_.d_rho_new, dataBase_.d_face_vector, dataBase_.d_grad, dataBase_.d_weight,
                                                                               dataBase_.d_volume, 1., d_b, d_b);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_tensor_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
                                                                               dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                               dataBase_.d_boundary_nuEff, dataBase_.d_boundary_rho, dataBase_.d_boundary_face_vector, dataBase_.d_grad_boundary,
                                                                               dataBase_.d_volume, 1., d_b, d_b);
    end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::fvm_laplacian()
{
    clock_t start = std::clock();
    // launch CUDA kernels
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_uncorrected_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
                                                                                                 d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index, dataBase_.d_rho_new, dataBase_.d_nuEff, dataBase_.d_weight,
                                                                                                 dataBase_.d_face, dataBase_.d_deltaCoeffs, -1., d_A_csr, d_A_csr);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_uncorrected_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
                                                                                                 d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                                 dataBase_.d_boundary_nuEff, dataBase_.d_boundary_rho, dataBase_.d_boundary_face, dataBase_.d_laplac_internal_coeffs,
                                                                                                 dataBase_.d_laplac_boundary_coeffs, -1., d_A_csr, d_b, d_A_csr, d_b, d_ueqn_internal_coeffs, d_ueqn_boundary_coeffs);
    clock_t end = std::clock();
    time_monitor_GPU_kernel += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfUEqn::A(double *Psi)
{
    checkCudaErrors(cudaMemsetAsync(d_A, 0, cell_bytes, stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    addAveInternaltoDiag<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells, d_A_csr_row_index, d_A_csr_diag_index,
                                                                            dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                            d_ueqn_internal_coeffs, d_ueqn_boundary_coeffs, d_A);
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    addDiagDivVolume<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, d_A_csr_row_index, d_A_csr_diag_index, d_A_csr,
                                                                        dataBase_.d_volume, d_ueqn_internal_coeffs, d_A, d_A);

    checkCudaErrors(cudaMemcpyAsync(h_A, d_A, cell_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < num_cells; i++)
        Psi[i] = h_A[i];
}

void dfUEqn::H(double *Psi)
{
    checkCudaErrors(cudaMemsetAsync(d_H, 0, cell_bytes * 3, stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    addBoundaryDiag<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells, d_A_csr_row_index, d_A_csr_diag_index,
                                                                       dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                       d_ueqn_internal_coeffs, d_ueqn_boundary_coeffs,
                                                                       d_psi, d_H);

    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    lduMatrix_H<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                   dataBase_.d_volume, d_psi, d_A_csr, d_b, d_ueqn_boundary_coeffs, d_H);

    checkCudaErrors(cudaMemcpyAsync(h_H, d_H, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < num_cells; i++)
    {
        Psi[i * 3] = h_H[i];
        Psi[i * 3 + 1] = h_H[num_cells + i];
        Psi[i * 3 + 2] = h_H[num_cells * 2 + i];
    }

    // for (int i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_H_GPU[%d]: (%.5e, %.5e, %.5e)\n", i, h_H[i], h_H[num_cells + i], h_H[num_cells * 2 + i]);
}

void dfUEqn::initializeTimeStep()
{
    // initialize matrix value
    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));
    // initialize boundary coeffs
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (dataBase_.num_boundary_faces + threads_per_block - 1) / threads_per_block;
    ueqn_update_BoundaryCoeffs_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_faces, dataBase_.d_boundary_phi,
                                                                                         dataBase_.d_internal_coeffs, dataBase_.d_boundary_coeffs,
                                                                                         dataBase_.d_laplac_internal_coeffs, dataBase_.d_laplac_boundary_coeffs);
}

void dfUEqn::checkValue(bool print)
{
    checkCudaErrors(cudaMemcpyAsync(h_A_csr, d_A_csr, csr_value_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_b, d_b, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize stream
    checkCudaErrors(cudaStreamSynchronize(stream));
    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            fprintf(stderr, "h_A_csr[%d]: %.5e\n", i, h_A_csr[i]);
        for (int i = 0; i < num_cells * 3; i++)
            fprintf(stderr, "h_b[%d]: %.5e\n", i, h_b[i]);
    }

    char *input_file = "of_output.txt";
    FILE *fp = fopen(input_file, "rb+");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open input file: %s!\n", input_file);
    }
    int readfile = 0;
    double *of_b = new double[3 * num_cells];
    double *of_A = new double[num_faces + num_cells];
    readfile = fread(of_b, num_cells * 3 * sizeof(double), 1, fp);
    readfile = fread(of_A, (num_faces + num_cells) * sizeof(double), 1, fp);

    std::vector<double> h_A_of_init_vec(num_cells + num_faces);
    std::copy(of_A, of_A + num_cells + num_faces, h_A_of_init_vec.begin());

    std::vector<double> h_A_of_vec_1mtx(num_faces + num_cells, 0);
    for (int i = 0; i < num_faces + num_cells; i++)
    {
        h_A_of_vec_1mtx[i] = h_A_of_init_vec[dataBase_.tmpPermutatedList[i]];
    }

    std::vector<double> h_A_of_vec((num_faces + num_cells) * 3);
    for (int i = 0; i < 3; i++)
    {
        std::copy(h_A_of_vec_1mtx.begin(), h_A_of_vec_1mtx.end(), h_A_of_vec.begin() + i * (num_faces + num_cells));
    }

    // b
    std::vector<double> h_b_of_init_vec(3 * num_cells);
    std::copy(of_b, of_b + 3 * num_cells, h_b_of_init_vec.begin());
    std::vector<double> h_b_of_vec;
    for (int i = 0; i < 3 * num_cells; i += 3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_y
    for (int i = 1; i < 3 * num_cells; i += 3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_z
    for (int i = 2; i < 3 * num_cells; i += 3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }

    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            fprintf(stderr, "h_A_of_vec_1mtx[%d]: %.5e\n", i, h_A_of_vec_1mtx[i]);
        for (int i = 0; i < 3 * num_cells; i++)
            fprintf(stderr, "h_b_of_vec[%d]: %.5e\n", i, h_b_of_vec[i]);
    }

    // check
    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual(num_faces + num_cells, h_A_of_vec_1mtx.data(), h_A_csr, 1e-5);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(3 * num_cells, h_b_of_vec.data(), h_b, 1e-5);
}

void dfUEqn::solve()
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
    printf("GPU Time (kernel launch)   = %.6lf s\n", time_monitor_GPU_kernel);
    printf("GPU Time (memcpy)          = %.6lf s\n", time_monitor_GPU_memcpy);
    time_monitor_CPU = 0;
    time_monitor_GPU_kernel = 0;
    time_monitor_GPU_memcpy = 0;

    // nvtxRangePush("solve");

    int nNz = num_cells + num_faces; // matrix entries
    if (num_iteration == 0)          // first interation
    {
        printf("Initializing AmgX Linear Solver\n");
        UxSolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr);
        UySolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr + nNz);
        UzSolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr + 2 * nNz);
    }
    else
    {
        UxSolver->updateOperator(num_cells, nNz, d_A_csr);
        UySolver->updateOperator(num_cells, nNz, d_A_csr + nNz);
        UzSolver->updateOperator(num_cells, nNz, d_A_csr + 2 * nNz);
    }
    UxSolver->solve(num_cells, d_psi, d_b);
    UySolver->solve(num_cells, d_psi + num_cells, d_b + num_cells);
    UzSolver->solve(num_cells, d_psi + 2 * num_cells, d_b + 2 * num_cells);
    num_iteration++;

    checkCudaErrors(cudaMemcpyAsync(h_psi, d_psi, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
    // for (size_t i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_velocity_after[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_psi[i],
    //     h_psi[num_cells + i], h_psi[num_cells*2 + i]);
}

void dfUEqn::updatePsi(double *Psi)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < num_cells; i++)
    {
        Psi[i * 3] = h_psi[i];
        Psi[i * 3 + 1] = h_psi[num_cells + i];
        Psi[i * 3 + 2] = h_psi[num_cells * 2 + i];
    }
}

// correct volecity in pEqn
void dfUEqn::correctPsi(double *Psi)
{
    for (size_t i = 0; i < num_cells; i++)
    {
        h_psi[i] = Psi[i * 3];
        h_psi[num_cells + i] = Psi[i * 3 + 1];
        h_psi[num_cells * 2 + i] = Psi[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpyAsync(d_psi, h_psi, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
}

dfUEqn::~dfUEqn()
{
}

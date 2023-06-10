#include "dfYEqn.H"

// kernel functions
__global__ void getUpwindWeight(int num_faces, double *phi, double *weight)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_faces)
        return;
    if (phi[index] >= 0)
        weight[index] = 1.;
    else
        weight[index] = 0.;
}

__global__ void fvc_grad_internal(int num_cells,
        const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
        const double *face_vector, const double *weight, const double *species,
        const double *volume, double *grady)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_Y = species[index];
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
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
            double neighbor_cell_Y = species[neighbor_cell_id];
            double face_Y = w * (neighbor_cell_Y - own_cell_Y) + own_cell_Y;
            grad_bx -= face_Y * sfx;
            grad_by -= face_Y * sfy;
            grad_bz -= face_Y * sfz;
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
            double neighbor_cell_Y = species[neighbor_cell_id];
            double face_Y = w * (own_cell_Y - neighbor_cell_Y) + neighbor_cell_Y;
            grad_bx += face_Y * sfx;
            grad_by += face_Y * sfy;
            grad_bz += face_Y * sfz;
        }
    }
    double vol = volume[index];
    grady[index * 3 + 0] = grad_bx / vol;
    grady[index * 3 + 1] = grad_by / vol;
    grady[index * 3 + 2] = grad_bz / vol;
}

__global__ void fvc_grad_boundary(int num_boundary_cells,
        const int *boundary_cell_offset, const int *boundary_cell_id, const int *bouPermedIndex,
        const double *boundary_face_vector, const double *boundary_species_init,
        const double *volume, double *grady)
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
        int permute_index = bouPermedIndex[i];
        double face_Y = boundary_species_init[permute_index];
        grad_bx += face_Y * sfx;
        grad_by += face_Y * sfy;
        grad_bz += face_Y * sfz;
    }

    double vol = volume[index];
    grady[cell_index * 3 + 0] += grad_bx / vol;
    grady[cell_index * 3 + 1] += grad_by / vol;
    grady[cell_index * 3 + 2] += grad_bz / vol;
}

__global__ void correct_boundary_conditions(int num_boundary_cells,
                                                const int *boundary_cell_offset, const int *boundary_cell_id,
                                                const double *boundary_sf, const double *mag_sf,
                                                const double *grady, double* boundary_grady)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // initialize boundary_sumYDiffError
    double grady_x = grady[cell_index * 3 + 0];
    double grady_y = grady[cell_index * 3 + 1];
    double grady_z = grady[cell_index * 3 + 2];

    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double n_x = boundary_sf[i * 3 + 0] / mag_sf[i];
        double n_y = boundary_sf[i * 3 + 1] / mag_sf[i];
        double n_z = boundary_sf[i * 3 + 2] / mag_sf[i];
        double sn_grad = 0;
        double grad_correction = sn_grad - (n_x * grady_x + n_y * grady_y + n_z * grady_z);
        boundary_grady[i * 3 + 0] = grady_x + grad_correction * n_x;
        boundary_grady[i * 3 + 1] = grady_y + grad_correction * n_y;
        boundary_grady[i * 3 + 2] = grady_z + grad_correction * n_z;
    }
}

__global__ void sumError_internal(int num_cells,
        const double *hai, const double *rhoD, const double *y, const double *grady,
        double *sum_hai_rhoD_grady, double *sum_rhoD_grady, double *sum_hai_y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    sum_hai_rhoD_grady[index * 3 + 0] += hai[index] * rhoD[index] * grady[index * 3 + 0];
    sum_hai_rhoD_grady[index * 3 + 1] += hai[index] * rhoD[index] * grady[index * 3 + 1];
    sum_hai_rhoD_grady[index * 3 + 2] += hai[index] * rhoD[index] * grady[index * 3 + 2];

    sum_rhoD_grady[index * 3 + 0] += rhoD[index] * grady[index * 3 + 0];
    sum_rhoD_grady[index * 3 + 1] += rhoD[index] * grady[index * 3 + 1];
    sum_rhoD_grady[index * 3 + 2] += rhoD[index] * grady[index * 3 + 2];

    sum_hai_y[index] += hai[index] * y[index];
}

__global__ void sumError_boundary(int num_boundary_faces, const int *bouPermedIndex,
        const double *boundary_hai, const double *boundary_rhoD, const double *boundary_y, const double *boundary_grady,
        double *sum_boundary_hai_rhoD_grady, double *sum_boundary_rhoD_grady, double *sum_boundary_hai_y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    int permute_index = bouPermedIndex[index];
    sum_boundary_hai_rhoD_grady[index * 3 + 0] += boundary_hai[permute_index] * boundary_rhoD[permute_index] * boundary_grady[index * 3 + 0];
    sum_boundary_hai_rhoD_grady[index * 3 + 1] += boundary_hai[permute_index] * boundary_rhoD[permute_index] * boundary_grady[index * 3 + 1];
    sum_boundary_hai_rhoD_grady[index * 3 + 2] += boundary_hai[permute_index] * boundary_rhoD[permute_index] * boundary_grady[index * 3 + 2];

    sum_boundary_rhoD_grady[index * 3 + 0] += boundary_rhoD[permute_index] * boundary_grady[index * 3 + 0];
    sum_boundary_rhoD_grady[index * 3 + 1] += boundary_rhoD[permute_index] * boundary_grady[index * 3 + 1];
    sum_boundary_rhoD_grady[index * 3 + 2] += boundary_rhoD[permute_index] * boundary_grady[index * 3 + 2];

    sum_boundary_hai_y[index] += boundary_hai[permute_index] * boundary_y[permute_index];
}

__global__ void calculate_hDiffCorrFlux(int num,
        const double *sum_hai_rhoD_grady, const double *sum_rhoD_grady, const double *sum_hai_y, double *hDiffCorrFlux)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    hDiffCorrFlux[index * 3 + 0] += (sum_hai_rhoD_grady[index * 3 + 0] - sum_hai_y[index] * sum_rhoD_grady[index * 3 + 0]);
    hDiffCorrFlux[index * 3 + 1] += (sum_hai_rhoD_grady[index * 3 + 1] - sum_hai_y[index] * sum_rhoD_grady[index * 3 + 1]);
    hDiffCorrFlux[index * 3 + 2] += (sum_hai_rhoD_grady[index * 3 + 2] - sum_hai_y[index] * sum_rhoD_grady[index * 3 + 2]);
}

__global__ void calculate_phiUc_internal(int num_cells,
        const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
        const double *face_vector, const double *weight, const double *sumYDiffError, double *phiUc)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_sumYDiffError_x = sumYDiffError[index * 3 + 0];
    double own_cell_sumYDiffError_y = sumYDiffError[index * 3 + 1];
    double own_cell_sumYDiffError_z = sumYDiffError[index * 3 + 2];

    // lower
    for (int i = 0; i < diag_index; i++)
    {
        double phiUc_face = 0;

        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = weight[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        double neighbor_cell_sumYDiffError_x = sumYDiffError[neighbor_cell_id * 3 + 0];
        double neighbor_cell_sumYDiffError_y = sumYDiffError[neighbor_cell_id * 3 + 1];
        double neighbor_cell_sumYDiffError_z = sumYDiffError[neighbor_cell_id * 3 + 2];
        double face_x = w * (neighbor_cell_sumYDiffError_x - own_cell_sumYDiffError_x) + own_cell_sumYDiffError_x;
        double face_y = w * (neighbor_cell_sumYDiffError_y - own_cell_sumYDiffError_y) + own_cell_sumYDiffError_y;
        double face_z = w * (neighbor_cell_sumYDiffError_z - own_cell_sumYDiffError_z) + own_cell_sumYDiffError_z;

        phiUc_face = face_x * sfx + face_y * sfy + face_z * sfz;
        phiUc[neighbor_index] = phiUc_face;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        double phiUc_face = 0;

        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = weight[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        double neighbor_cell_sumYDiffError_x = sumYDiffError[neighbor_cell_id * 3 + 0];
        double neighbor_cell_sumYDiffError_y = sumYDiffError[neighbor_cell_id * 3 + 1];
        double neighbor_cell_sumYDiffError_z = sumYDiffError[neighbor_cell_id * 3 + 2];
        double face_x = w * (own_cell_sumYDiffError_x - neighbor_cell_sumYDiffError_x) + neighbor_cell_sumYDiffError_x;
        double face_y = w * (own_cell_sumYDiffError_y - neighbor_cell_sumYDiffError_y) + neighbor_cell_sumYDiffError_y;
        double face_z = w * (own_cell_sumYDiffError_z - neighbor_cell_sumYDiffError_z) + neighbor_cell_sumYDiffError_z;

        phiUc_face = face_x * sfx + face_y * sfy + face_z * sfz;
        phiUc[neighbor_index] = phiUc_face;
    }
}

__global__ void calculate_phiUc_boundary(int num_boundary_faces,
                                         const int *boundary_cell_offset, const int *boundary_cell_id,
                                         const double *boundary_sf, const double *boundary_sumYDiffError,
                                         double *boundary_phiUc)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    double n_x = boundary_sf[index * 3 + 0];
    double n_y = boundary_sf[index * 3 + 1];
    double n_z = boundary_sf[index * 3 + 2];

    double err_x = boundary_sumYDiffError[index * 3 + 0];
    double err_y = boundary_sumYDiffError[index * 3 + 1];
    double err_z = boundary_sumYDiffError[index * 3 + 2];

    boundary_phiUc[index] = n_x * err_x + n_y * err_y + n_z * err_z;
}

__global__ void fvm_ddt_kernel_scalar(int num_cells, int num_faces, const double rdelta_t,
                                      const int *csr_row_index, const int *csr_diag_index,
                                      const double *rho_old, const double *rho_new, const double *volume, const double *species_old,
                                      const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int diag_index = csr_diag_index[index];
    int csr_index = row_index + diag_index;

    double ddt_diag = rdelta_t * rho_new[index] * volume[index];
    A_csr_output[csr_index] = A_csr_input[csr_index] + ddt_diag;

    double ddt_part_term = rdelta_t * rho_old[index] * volume[index];
    b_output[index] = b_input[index] + ddt_part_term * species_old[index];
}

__global__ void compute_inertIndex_y(int num_cells, int num_species, int inertIndex, double *y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double sum_yi = 0;
    for (int i = 0; i < num_species; i++)
    {
        if (i == inertIndex) continue;

        double yi = y[num_cells * i + index];
        sum_yi += yi > 0 ? yi : 0;
    }
    sum_yi = 1 - sum_yi;
    y[num_cells * inertIndex + index] = (sum_yi > 0 ? sum_yi : 0);
}

__global__ void fvm_div_internal_scalar(int num_cells, int num_faces,
                                        const int *csr_row_index, const int *csr_diag_index,
                                        const double *div_weight, const double *phi,
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

    double div_diag = 0;
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index;
            double w = div_weight[neighbor_index];
            double f = phi[neighbor_index];
            A_csr_output[i] = A_csr_input[i] + (-w) * f;
            // lower neighbors contribute to sum of -1
            div_diag += (w - 1) * f;
        }
        // upper
        if (inner_index > diag_index)
        {
            // upper, index - 1, consider of diag
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = div_weight[neighbor_index];
            double f = phi[neighbor_index];
            A_csr_output[i] = A_csr_input[i] + (1 - w) * f;
            // upper neighbors contribute to sum of 1
            div_diag += w * f;
        }
    }
    A_csr_output[row_index + diag_index] = A_csr_input[row_index + diag_index] + div_diag; // diag
}

__global__ void fvm_div_boundary_scalar(int num_cells, int num_faces, int num_boundary_cells,
                                        const int *csr_row_index, const int *csr_diag_index, const double *boundary_phi,
                                        const int *boundary_cell_offset, const int *boundary_cell_id,
                                        double *internal_coeffs, const double *boundary_coeffs,
                                        const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output)
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

    // construct internalCoeffs & boundaryCoeffs
    double internal_coeffs_own = 0;
    double boundary_coeffs_own = 0;
    for (int i = 0; i < loop_size; i++)
    {
        // zeroGradient
        internal_coeffs[cell_offset + i] = 1.;
        internal_coeffs_own += boundary_phi[cell_offset + i] * internal_coeffs[cell_offset + i];
        boundary_coeffs_own += -boundary_phi[cell_offset + i] * boundary_coeffs[cell_offset + i];
    }
    A_csr_output[csr_index] = A_csr_input[csr_index] + internal_coeffs_own;
    b_output[cell_index] = b_input[cell_index] + boundary_coeffs_own;
}

__global__ void fvm_laplacian_uncorrected_scalar_internal(int num_cells,
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

    double own_coeff = scalar0[index] + scalar1[index];
    double sum_diag = 0;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_coeff = scalar0[neighbor_cell_id] + scalar1[neighbor_cell_id];
        double gamma = w * (nei_coeff - own_coeff) + own_coeff;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[row_index + i] = A_csr_input[row_index + i] + coeff * sign;

        sum_diag += (-coeff);
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_coeff = scalar0[neighbor_cell_id] + scalar1[neighbor_cell_id];
        double gamma = w * (own_coeff - nei_coeff) + nei_coeff;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[row_index + i] = A_csr_input[row_index + i] + coeff * sign;

        sum_diag += (-coeff);
    }
    // diag
    A_csr_output[row_index + diag_index] = A_csr_input[row_index + diag_index] + sum_diag * sign;
}

__global__ void fvm_laplacian_uncorrected_scalar_boundary(int num_boundary_cells,
        const int *csr_row_index, const int *csr_diag_index, const int *boundary_cell_offset,
        const int *boundary_cell_id, const double *boundary_scalar0, const double *boundary_scalar1,
        const double *boundary_magsf, const int *bouPermedIndex,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        const double sign, const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    int row_index = csr_row_index[cell_index];
    int diag_index = csr_diag_index[cell_index];
    int csr_index = row_index + diag_index;

    double internal_coeffs = 0;
    double boundary_coeffs = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        int permute_index = bouPermedIndex[i];
        double gamma = boundary_scalar0[permute_index] + boundary_scalar1[permute_index];
        double gamma_magsf = gamma * boundary_magsf[i];
        // internal_coeffs += gamma_magsf * gradient_internal_coeffs[i * 3 + 0];
        // boundary_coeffs += gamma_magsf * gradient_boundary_coeffs[i * 3 + 0];
        internal_coeffs += gamma_magsf * 0.;
        boundary_coeffs += gamma_magsf * 0.;
    }

    A_csr_output[csr_index] = A_csr_input[csr_index] + internal_coeffs * sign;
    b_output[cell_index] = b_input[cell_index] + boundary_coeffs * sign;
}

__global__ void fvc_laplacian_internal(int num_cells,
        const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
        const double *alpha, const double *hai, const double* y,
        const double *weight, const double *magsf, const double *distance,
        const double* volume, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_vf = y[index];
    double own_coeff = alpha[index] * hai[index];
    double sum = 0;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_vf = y[neighbor_cell_id];
        double nei_coeff = alpha[neighbor_cell_id] * hai[neighbor_cell_id];
        double face_gamma = (1 - w) * own_coeff + w * nei_coeff;
        double sngrad = distance[neighbor_index] * (own_vf - nei_vf);
        double value = face_gamma * sngrad * magsf[neighbor_index];
        sum -= value;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_vf = y[neighbor_cell_id];
        double nei_coeff = alpha[neighbor_cell_id] * hai[neighbor_cell_id];
        double face_gamma = w * own_coeff + (1 - w) * nei_coeff;
        double sngrad = distance[neighbor_index] * (nei_vf - own_vf);
        double value = face_gamma * sngrad * magsf[neighbor_index];
        sum += value;
    }
    double vol = volume[index];
    output[index] += sum / vol;
}

dfYEqn::dfYEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile, const int inertIndex)
    : dataBase_(dataBase), inertIndex(inertIndex)
{
    stream = dataBase_.stream;
    num_species = dataBase_.num_species;
    num_cells = dataBase_.num_cells;
    num_faces = dataBase_.num_faces;
    num_surfaces = dataBase_.num_surfaces;
    num_boundary_cells = dataBase_.num_boundary_cells;
    num_boundary_faces = dataBase_.num_boundary_faces;
    cell_bytes = dataBase_.cell_bytes;
    boundary_face_bytes = dataBase_.boundary_face_bytes;

    YSolverSet.resize(num_species - 1); // consider inert species
    for (auto &solver : YSolverSet)
        solver = new AmgXSolver(modeStr, cfgFile);

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    h_A_csr = new double[(num_cells + num_faces) * (num_species - 1)];
    h_b = new double[num_cells * (num_species - 1)];
    cudaMallocHost(&h_psi, num_cells * num_species * sizeof(double));

    checkCudaErrors(cudaMalloc((void **)&d_A_csr, (num_cells + num_faces) * (num_species - 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_b, cell_bytes * (num_species - 1)));
    checkCudaErrors(cudaMalloc((void **)&d_psi, cell_bytes * (num_species - 1)));
    checkCudaErrors(cudaMalloc((void **)&d_phiUc, num_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_phiUc_boundary, num_boundary_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_mut_Sct, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_mut_sct, boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_boundary_Y, boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_hai, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_hai, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_rhoD, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_rhoD, boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_sum_rhoD_grady, 3 * cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sum_boundary_rhoD_grady, 3 * boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sum_hai_rhoD_grady, 3 * cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sum_boundary_hai_rhoD_grady, 3 * boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sum_hai_y, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sum_boundary_hai_y, boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_grady, 3 * cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_grady, 3 * boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_alpha, cell_bytes));

    // zeroGradient
    for (size_t i = 0; i < num_species; i++)
    {
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_internal_coeffs_Y_vector[i], 1, boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_boundary_coeffs_Y_vector[i], 0, boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_laplac_internal_coeffs_Y_vector[i], 0, boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_laplac_boundary_coeffs_Y_vector[i], 0, boundary_face_bytes, stream));
    }
}

void dfYEqn::initializeTimeStep()
{
    // consider inert species
    // initialize matrix value
    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, (num_cells + num_faces) * (num_species - 1) * sizeof(double), stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_bytes * (num_species - 1), stream));
    // initialize variables in each time step
    checkCudaErrors(cudaMemsetAsync(d_psi, 0, cell_bytes * (num_species - 1), stream));
}

void dfYEqn::upwindWeight()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_faces + threads_per_block - 1) / threads_per_block;
    getUpwindWeight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_faces, dataBase_.d_phi, dataBase_.d_weight_upwind);
}

void dfYEqn::fvm_laplacian_and_sumYDiffError_diffAlphaD_hDiffCorrFlux(std::vector<double *> Y_old, std::vector<double *> boundary_Y,
        std::vector<const double *> hai, std::vector<double *> boundary_hai,
        std::vector<const double *> rhoD, std::vector<double *> boundary_rhoD,
        const double *mut_Sct, const double *boundary_mut_Sct, const double *alpha)
{
    // initialize variables in each time step
    checkCudaErrors(cudaMemcpyAsync(d_boundary_mut_sct, boundary_mut_Sct, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mut_Sct, mut_Sct, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_alpha, alpha, cell_bytes, cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMemsetAsync(d_sum_rhoD_grady, 0, 3 * cell_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_sum_boundary_rhoD_grady, 0, 3 * boundary_face_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_sum_hai_rhoD_grady, 0, 3 * cell_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_sum_boundary_hai_rhoD_grady, 0, 3 * boundary_face_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_sum_hai_y, 0, cell_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_sum_boundary_hai_y, 0, boundary_face_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_hDiffCorrFlux, 0, 3 * cell_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_boundary_hDiffCorrFlux, 0, 3 * boundary_face_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(dataBase_.d_diffAlphaD, 0, cell_bytes, stream));

    size_t threads_per_block, blocks_per_grid;
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_Y + i * num_cells, Y_old[i], cell_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_boundary_Y, boundary_Y[i], boundary_face_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_hai, hai[i], cell_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_boundary_hai, boundary_hai[i], boundary_face_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_rhoD, rhoD[i], cell_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_boundary_rhoD, boundary_rhoD[i], boundary_face_bytes, cudaMemcpyHostToDevice, stream));

        checkCudaErrors(cudaMemsetAsync(d_grady, 0, 3 * cell_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(d_boundary_grady, 0, 3 * boundary_face_bytes, stream));

        // fvc::grad(Yi)
        threads_per_block = 1024;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvc_grad_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                dataBase_.d_face_vector, dataBase_.d_weight, dataBase_.d_Y + i * num_cells,
                dataBase_.d_volume, d_grady);
        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        fvc_grad_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id, dataBase_.d_bouPermedIndex,
                dataBase_.d_boundary_face_vector, d_boundary_Y,
                dataBase_.d_volume, d_grady);
        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        correct_boundary_conditions<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                dataBase_.d_boundary_face_vector, dataBase_.d_boundary_face,
                d_grady, d_boundary_grady);

        // sum(chemistry->hai(i)*chemistry->rhoD(i)*fvc::grad(Yi))
        // sum(chemistry->rhoD(i)*fvc::grad(Yi)), also be called sumYDiffError
        // sum(chemistry->hai(i)*Yi)
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        sumError_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                d_hai, d_rhoD, dataBase_.d_Y + i * num_cells, d_grady,
                d_sum_hai_rhoD_grady, d_sum_rhoD_grady, d_sum_hai_y);
        blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
        sumError_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces,
                dataBase_.d_bouPermedIndex,
                d_boundary_hai, d_boundary_rhoD, d_boundary_Y, d_boundary_grady,
                d_sum_boundary_hai_rhoD_grady, d_sum_boundary_rhoD_grady, d_sum_boundary_hai_y);

        // compute diffAlphaD
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvc_laplacian_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                d_alpha, d_hai, dataBase_.d_Y + i * num_cells,
                dataBase_.d_weight, dataBase_.d_face, dataBase_.d_deltaCoeffs,
                dataBase_.d_volume, dataBase_.d_diffAlphaD);

        // fvm::laplacian
        if (i != inertIndex)
        {
            blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
            fvm_laplacian_uncorrected_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                    d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                    d_mut_Sct, d_rhoD, dataBase_.d_weight, dataBase_.d_face, dataBase_.d_deltaCoeffs,
                    -1., d_A_csr + mtxIndex * (num_cells + num_faces), d_A_csr + mtxIndex * (num_cells + num_faces));
            blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
            fvm_laplacian_uncorrected_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                    d_A_csr_row_index, d_A_csr_diag_index,
                    dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                    d_boundary_rhoD, d_boundary_mut_sct, dataBase_.d_boundary_face, dataBase_.d_bouPermedIndex,
                    dataBase_.d_laplac_internal_coeffs_Y_vector[i], dataBase_.d_laplac_boundary_coeffs_Y_vector[i],
                    -1., d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                    d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);
        }
        ++mtxIndex;
    }

    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_hDiffCorrFlux<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            d_sum_hai_rhoD_grady, d_sum_rhoD_grady, d_sum_hai_y, dataBase_.d_hDiffCorrFlux);
    blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
    calculate_hDiffCorrFlux<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces,
            d_sum_boundary_hai_rhoD_grady, d_sum_boundary_rhoD_grady, d_sum_boundary_hai_y, dataBase_.d_boundary_hDiffCorrFlux);
}

void dfYEqn::fvm_ddt()
{
    size_t threads_per_block, blocks_per_grid;
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;

        // launch cuda kernel
        threads_per_block = 1024;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvm_ddt_kernel_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, dataBase_.rdelta_t,
                d_A_csr_row_index, d_A_csr_diag_index,
                dataBase_.d_rho_old, dataBase_.d_rho_new, dataBase_.d_volume, dataBase_.d_Y + i * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);
                //d_psi + mtxIndex * num_cells);
        ++mtxIndex;
    }
}

void dfYEqn::fvm_div_phi()
{
    size_t threads_per_block, blocks_per_grid;
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;

        // mvConvection->fvmDiv(phi, Yi)
        threads_per_block = 512;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvm_div_internal_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
                d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_weight_upwind, dataBase_.d_phi,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);
        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        fvm_div_boundary_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
                d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_boundary_phi,
                dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                dataBase_.d_internal_coeffs_Y_vector[i], dataBase_.d_boundary_coeffs_Y_vector[i],
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);
        ++mtxIndex;
    }
}

void dfYEqn::fvm_div_phiUc()
{
    size_t threads_per_block, blocks_per_grid;

    // compue phiUc
    threads_per_block = 512;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_phiUc_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
            dataBase_.d_face_vector, dataBase_.d_weight, d_sum_rhoD_grady, d_phiUc);
    blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
    calculate_phiUc_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces,
            dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
            dataBase_.d_boundary_face_vector, d_sum_boundary_rhoD_grady, d_phiUc_boundary);

    // mvConvection->fvmDiv(phiUc, Yi)
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvm_div_internal_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
                d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_weight_upwind, d_phiUc,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);

        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        fvm_div_boundary_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
                d_A_csr_row_index, d_A_csr_diag_index, d_phiUc_boundary,
                dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                dataBase_.d_internal_coeffs_Y_vector[i], dataBase_.d_boundary_coeffs_Y_vector[i],
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells);
        ++mtxIndex;
    }
}

void dfYEqn::checkValue(bool print, char *filename)
{
    checkCudaErrors(cudaMemcpyAsync(h_A_csr, d_A_csr, (num_cells + num_faces) * sizeof(double), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_b, d_b, num_cells * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Synchronize stream
    checkCudaErrors(cudaStreamSynchronize(stream));
    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            fprintf(stderr, "h_A_csr[%d]: %.15lf\n", i, h_A_csr[i]);
        for (int i = 0; i < num_cells; i++)
            fprintf(stderr, "h_b[%d]: %.15lf\n", i, h_b[i]);
    }

    char *input_file = filename;
    FILE *fp = fopen(input_file, "rb+");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open input file: %s!\n", input_file);
    }

    int readfile = 0;
    double *of_b = new double[num_cells];
    double *of_A = new double[num_faces + num_cells];
    readfile = fread(of_b, num_cells * sizeof(double), 1, fp);
    readfile = fread(of_A, (num_faces + num_cells) * sizeof(double), 1, fp);

    std::vector<double> h_A_of_vec_1mtx(num_faces + num_cells, 0);
    for (int i = 0; i < num_faces + num_cells; i++)
    {
        h_A_of_vec_1mtx[i] = of_A[dataBase_.tmpPermutatedList[i]];
    }
    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            printf("h_A_of_vec_1mtx[%d]: %.15lf\n", i, h_A_of_vec_1mtx[i]);
        for (int i = 0; i < num_cells; i++)
            printf("h_b_of_vec[%d]: %.15lf\n", i, of_b[i]);
    }

    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual(num_faces + num_cells, h_A_of_vec_1mtx.data(), h_A_csr, 1e-5);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(num_cells, of_b, h_b, 1e-5);
}

void dfYEqn::solve()
{
    checkCudaErrors(cudaStreamSynchronize(stream));

    int nNz = num_cells + num_faces; // matrix entries
    if (num_iteration == 0)          // first interation
    {
        printf("Initializing AmgX Linear Solver\n");
        int solverIndex = 0;
        for (auto &solver : YSolverSet)
        {
            solver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr + solverIndex * nNz);
            ++solverIndex;
        }
    }
    else
    {
        int solverIndex = 0;
        for (auto &solver : YSolverSet)
        {
            solver->updateOperator(num_cells, nNz, d_A_csr + solverIndex * nNz);
            ++solverIndex;
        }
    }
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;

        YSolverSet[mtxIndex]->solve(num_cells, dataBase_.d_Y + i * num_cells, d_b + mtxIndex * num_cells);
        ++mtxIndex;
    }

    size_t threads_per_block, blocks_per_grid;
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    compute_inertIndex_y<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, inertIndex, dataBase_.d_Y);
    checkCudaErrors(cudaMemcpyAsync(h_psi, dataBase_.d_Y, num_species * cell_bytes, cudaMemcpyDeviceToHost, stream));

    num_iteration++;
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // for (size_t i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_species_gpu[%d]: %.5e\n", i, h_psi[i + 0 * num_cells]);
}

void dfYEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfYEqn::updatePsi(double *Psi, int speciesIndex)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    memcpy(Psi, h_psi + speciesIndex * num_cells, cell_bytes);
}

dfYEqn::~dfYEqn()
{
}

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

__global__ void fvc_grad_internal_face_Y(int num_cells,
                                         const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index, const double *volume,
                                         const double *face_vector, const double *weight, const double *species, const double *rhoD,
                                         const double *sumYDiffError, double *sumYDiffError_output)
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
    double grad_bx_upper = 0;
    double grad_by_upper = 0;
    double grad_bz_upper = 0;
    double grad_bx_lower = 0;
    double grad_by_lower = 0;
    double grad_bz_lower = 0;
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
            grad_bx_lower -= face_Y * sfx;
            grad_by_lower -= face_Y * sfy;
            grad_bz_lower -= face_Y * sfz;
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
            grad_bx_upper += face_Y * sfx;
            grad_by_upper += face_Y * sfy;
            grad_bz_upper += face_Y * sfz;
        }
    }
    double vol = volume[index];

    sumYDiffError_output[index * 3 + 0] = sumYDiffError[index * 3 + 0] + (grad_bx_upper + grad_bx_lower);
    sumYDiffError_output[index * 3 + 1] = sumYDiffError[index * 3 + 1] + (grad_by_upper + grad_by_lower);
    sumYDiffError_output[index * 3 + 2] = sumYDiffError[index * 3 + 2] + (grad_bz_upper + grad_bz_lower);

    // if (index == 0)
    // {
    //     printf("grad_bz_upper = %e\n", grad_bz_upper);
    //     printf("grad_bz_lower = %e\n", grad_bz_lower);
    //     printf("(grad_bz_upper + grad_bz_lower) = %e\n", (grad_bz_upper + grad_bz_lower));
    // }
}

__global__ void fvc_grad_boundary_face_Y(int num_cells, int num_boundary_cells,
                                         const int *boundary_cell_offset, const int *boundary_cell_id, const double *rhoD, const int *bouPermedIndex,
                                         const double *boundary_face_vector, const double *boundary_species_init, double *boundary_species,
                                         const double *volume, const double *sumYDiffError, double *sumYDiffError_output)
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
        boundary_species[i] = face_Y;
        grad_bx += face_Y * sfx;
        grad_by += face_Y * sfy;
        grad_bz += face_Y * sfz;
        // if (index == 0)
        // {
        //     printf("face_Y = %e\n", face_Y);
        //     printf("sfz = %e\n", sfz);
        //     printf("face_Y * sfz = %e\n", face_Y * sfz);
        // }
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

    sumYDiffError_output[cell_index * 3 + 0] = sumYDiffError[cell_index * 3 + 0] + grad_bx;
    sumYDiffError_output[cell_index * 3 + 1] = sumYDiffError[cell_index * 3 + 1] + grad_by;
    sumYDiffError_output[cell_index * 3 + 2] = sumYDiffError[cell_index * 3 + 2] + grad_bz;
}

__global__ void sumError(int num_cells, const double *volume, const double *rhoD,
                         const double *sumYDiffErrorTmp, double *sumYDiffError_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    sumYDiffError_output[index * 3 + 0] = sumYDiffError_output[index * 3 + 0] + sumYDiffErrorTmp[index * 3 + 0] * rhoD[index];
    sumYDiffError_output[index * 3 + 1] = sumYDiffError_output[index * 3 + 1] + sumYDiffErrorTmp[index * 3 + 1] * rhoD[index];
    sumYDiffError_output[index * 3 + 2] = sumYDiffError_output[index * 3 + 2] + sumYDiffErrorTmp[index * 3 + 2] * rhoD[index];
}

__global__ void divide_vol(int num_cells, const double *volume,
                           const double *sumYDiffError, double *sumYDiffError_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    double vol = volume[index];

    sumYDiffError_output[index * 3 + 0] = sumYDiffError[index * 3 + 0] / vol;
    sumYDiffError_output[index * 3 + 1] = sumYDiffError[index * 3 + 1] / vol;
    sumYDiffError_output[index * 3 + 2] = sumYDiffError[index * 3 + 2] / vol;
}

__global__ void correct_boundary_conditions_vec(int num_boundary_cells,
                                                const int *boundary_cell_offset, const int *boundary_cell_id,
                                                const double *boundary_sf, const double *mag_sf,
                                                double *boundary_sumYDiffError, double *sumYDiffError)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // initialize boundary_sumYDiffError
    double sumYDiffError_x = sumYDiffError[cell_index * 3 + 0];
    double sumYDiffError_y = sumYDiffError[cell_index * 3 + 1];
    double sumYDiffError_z = sumYDiffError[cell_index * 3 + 2];

    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double n_x = boundary_sf[i * 3 + 0] / mag_sf[i];
        double n_y = boundary_sf[i * 3 + 1] / mag_sf[i];
        double n_z = boundary_sf[i * 3 + 2] / mag_sf[i];
        double sn_grad = 0;
        double grad_correction = sn_grad - (n_x * sumYDiffError_x + n_y * sumYDiffError_y + n_z * sumYDiffError_z);
        boundary_sumYDiffError[i * 3 + 0] = sumYDiffError_x + grad_correction * n_x;
        boundary_sumYDiffError[i * 3 + 1] = sumYDiffError_y + grad_correction * n_y;
        boundary_sumYDiffError[i * 3 + 2] = sumYDiffError_z + grad_correction * n_z;
    }
}

__global__ void calculate_phiUc(int num_cells, const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
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
                                      const double *A_csr_input, const double *b_input, double *A_csr_output, double *b_output, double *psi)
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

    psi[index] = species_old[index];
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

__global__ void fvm_laplacian_uncorrected_scalar_boundary(int num_cells, int num_boundary_cells,
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

dfYEqn::dfYEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile, const int inertIndex)
    : dataBase_(dataBase), inertIndex(inertIndex)
{
    num_species = dataBase_.num_species;
    num_cells = dataBase_.num_cells;
    num_faces = dataBase_.num_faces;
    num_surfaces = dataBase_.num_surfaces;
    num_boundary_cells = dataBase_.num_boundary_cells;
    num_boundary_faces = dataBase_.num_boundary_faces;
    cell_bytes = dataBase_.cell_bytes;

    YSolverSet.resize(num_species - 1); // consider inert species
    for (auto &solver : YSolverSet)
        solver = new AmgXSolver(modeStr, cfgFile);

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    h_A_csr = new double[(num_cells + num_faces) * (num_species - 1)];
    h_b = new double[num_cells * (num_species - 1)];
    cudaMallocHost(&h_psi, num_cells * (num_species - 1) * sizeof(double));

    checkCudaErrors(cudaMalloc((void **)&d_A_csr, (num_cells + num_faces) * (num_species - 1) * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_b, cell_bytes * (num_species - 1)));
    checkCudaErrors(cudaMalloc((void **)&d_psi, cell_bytes * (num_species - 1)));
    checkCudaErrors(cudaMalloc((void **)&d_sumYDiffError, 3 * cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_sumYDiffError_tmp, 3 * cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_phiUc, num_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_phiUc, num_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_sumYDiffError_boundary, 3 * num_boundary_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_phiUc_boundary, num_boundary_faces * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_mut_Sct, cell_bytes));

    checkCudaErrors(cudaStreamCreate(&stream));
    // zeroGradient
    for (size_t i = 0; i < num_species; i++)
    {
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_internal_coeffs_Y_vector[i], 1, dataBase_.boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_boundary_coeffs_Y_vector[i], 0, dataBase_.boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_laplac_internal_coeffs_Y_vector[i], 0, dataBase_.boundary_face_bytes, stream));
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_laplac_boundary_coeffs_Y_vector[i], 0, dataBase_.boundary_face_bytes, stream));
    }
}

void dfYEqn::upwindWeight()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_faces + threads_per_block - 1) / threads_per_block;
    getUpwindWeight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_faces, dataBase_.d_phi, dataBase_.d_weight_upwind);
}

void dfYEqn::correctVelocity(std::vector<double *> Y_new, std::vector<double *> boundary_Y_init, std::vector<const double *> rhoD_GPU)
{
    // initialize variables in each time step
    checkCudaErrors(cudaMemsetAsync(d_sumYDiffError, 0, 3 * cell_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_phiUc, 0, num_faces * sizeof(double), stream));

    size_t threads_per_block, blocks_per_grid;
    for (size_t i = 0; i < num_species; ++i)
    {
        checkCudaErrors(cudaMemsetAsync(d_sumYDiffError_tmp, 0, 3 * cell_bytes, stream));
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_Y_new_vector[i], Y_new[i], cell_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_Y_init_vector[i], boundary_Y_init[i], dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_rhoD_vector[i], rhoD_GPU[i], cell_bytes, cudaMemcpyHostToDevice, stream));

        // launch cuda kernel
        threads_per_block = 1024;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvc_grad_internal_face_Y<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                    d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index, dataBase_.d_volume,
                                                                                    dataBase_.d_face_vector, dataBase_.d_weight, dataBase_.d_Y_new_vector[i],
                                                                                    dataBase_.d_rhoD_vector[i], d_sumYDiffError_tmp, d_sumYDiffError_tmp);
        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        fvc_grad_boundary_face_Y<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells, dataBase_.d_boundary_cell_offset,
                                                                                    dataBase_.d_boundary_cell_id, dataBase_.d_rhoD_vector[i], dataBase_.d_bouPermedIndex,
                                                                                    dataBase_.d_boundary_face_vector, dataBase_.d_boundary_Y_init_vector[i], dataBase_.d_boundary_Y_vector[i],
                                                                                    dataBase_.d_volume, d_sumYDiffError_tmp, d_sumYDiffError_tmp);
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        sumError<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.d_volume, dataBase_.d_rhoD_vector[i], d_sumYDiffError_tmp, d_sumYDiffError);
    }
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_vol<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.d_volume, d_sumYDiffError, d_sumYDiffError);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    correct_boundary_conditions_vec<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells, dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                       dataBase_.d_boundary_face_vector, dataBase_.d_boundary_face, d_sumYDiffError_boundary, d_sumYDiffError);

    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    calculate_phiUc<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                       d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index, dataBase_.d_face_vector,
                                                                       dataBase_.d_weight, d_sumYDiffError, d_phiUc);

    blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
    calculate_phiUc_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces, dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                dataBase_.d_boundary_face_vector, d_sumYDiffError_boundary, d_phiUc_boundary);
}

void dfYEqn::fvm_ddt(std::vector<double *> Y_old)
{
    // initialize variables in each time step
    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, (num_cells + num_faces) * (num_species - 1) * sizeof(double), stream)); // consider inert species
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_bytes * (num_species - 1), stream));
    checkCudaErrors(cudaMemsetAsync(d_psi, 0, cell_bytes * (num_species - 1), stream));

    size_t threads_per_block, blocks_per_grid;
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_Y_old_vector[i], Y_old[i], cell_bytes, cudaMemcpyHostToDevice, stream));

        // launch cuda kernel
        threads_per_block = 1024;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvm_ddt_kernel_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, dataBase_.rdelta_t,
                                                                                 d_A_csr_row_index, d_A_csr_diag_index,
                                                                                 dataBase_.d_rho_old, dataBase_.d_rho_new, dataBase_.d_volume, dataBase_.d_Y_old_vector[i],
                                                                                 d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                                                                                 d_A_csr + mtxIndex * (num_cells + num_faces), d_b + mtxIndex * num_cells,
                                                                                 d_psi + mtxIndex * num_cells);
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

        // launch cuda kernel
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
    int mtxIndex = 0;
    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;
        // launch cuda kernel
        threads_per_block = 512;
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

void dfYEqn::fvm_laplacian(double *mut_Sct, double *boundary_mut_Sct, std::vector<double *> boundary_rhoD)
{
    size_t threads_per_block, blocks_per_grid;
    int mtxIndex = 0;

    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_mut_sct, boundary_mut_Sct, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mut_Sct, mut_Sct, cell_bytes, cudaMemcpyHostToDevice, stream));

    for (size_t i = 0; i < num_species; ++i)
    {
        if (i == inertIndex)
            continue;
        checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_rhoD_vector[i], boundary_rhoD[i], dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));

        // launch cuda kernel
        threads_per_block = 1024;
        blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
        fvm_laplacian_uncorrected_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                                                     d_mut_Sct, dataBase_.d_rhoD_vector[i], dataBase_.d_weight, dataBase_.d_face,
                                                                                                     dataBase_.d_deltaCoeffs, -1., d_A_csr + mtxIndex * (num_cells + num_faces),
                                                                                                     d_A_csr + mtxIndex * (num_cells + num_faces));
        blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
        fvm_laplacian_uncorrected_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
                                                                                                     d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_boundary_cell_offset,
                                                                                                     dataBase_.d_boundary_cell_id, dataBase_.d_boundary_rhoD_vector[i],
                                                                                                     dataBase_.d_boundary_mut_sct, dataBase_.d_boundary_face, dataBase_.d_bouPermedIndex,
                                                                                                     dataBase_.d_laplac_internal_coeffs_Y_vector[i], dataBase_.d_laplac_boundary_coeffs_Y_vector[i], -1.,
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
    int solverIndex = 0;
    for (auto &solver : YSolverSet)
    {
        solver->solve(num_cells, d_psi + solverIndex * num_cells, d_b + solverIndex * num_cells);
        ++solverIndex;
    }
    num_iteration++;
    checkCudaErrors(cudaMemcpyAsync(h_psi, d_psi, num_cells * (num_species - 1) * sizeof(double), cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // for (size_t i = 0; i < num_cells; i++)
    //     fprintf(stderr, "h_species_gpu[%d]: %.5e\n", i, h_psi[i + 0 * num_cells]);
}

void dfYEqn::updatePsi(double *Psi, int speciesIndex)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < num_cells; i++)
        Psi[i] = h_psi[i + speciesIndex * num_cells];
}

dfYEqn::~dfYEqn()
{
}
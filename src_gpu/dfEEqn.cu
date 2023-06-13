#include "dfEEqn.H"

// kernel functions

__global__ void eeqn_fvm_ddt_kernel(int num_cells, const double rdelta_t,
                                    const int *csr_row_index, const int *csr_diag_index,
                                    const double *rho_old, const double *rho_new, const double *volume, const double *he_old,
                                    const double sign, const double *A_csr_input, const double *b_input,
                                    double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int diag_index = csr_diag_index[index];
    int csr_index = row_index + diag_index;
    double ddt_diag = rdelta_t * rho_new[index] * volume[index];
    A_csr_output[csr_index] = A_csr_input[csr_index] + ddt_diag * sign;

    double ddt_part_term = rdelta_t * rho_old[index] * volume[index];
    b_output[index] = b_input[index] + ddt_part_term * he_old[index] * sign;
}

__global__ void eeqn_fvm_div_internal(int num_cells,
                                      const int *csr_row_index, const int *csr_diag_index,
                                      const double *weight, const double *phi,
                                      const double sign, const double *A_csr_input, const double *b_input,
                                      double *A_csr_output, double *b_output)
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
            double w = weight[neighbor_index];
            double f = phi[neighbor_index];
            A_csr_output[i] = A_csr_input[i] + (-w) * f * sign;
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
            A_csr_output[i] = A_csr_input[i] + (1 - w) * f * sign;
            // upper neighbors contribute to sum of 1
            div_diag += w * f;
        }
    }
    A_csr_output[row_index + diag_index] = A_csr_input[row_index + diag_index] + div_diag * sign; // diag
}

__global__ void eeqn_fvm_div_boundary(int num_boundary_cells,
                                      const int *csr_row_index, const int *csr_diag_index,
                                      const int *boundary_cell_offset, const int *boundary_cell_id,
                                      const double *value_internal_coeffs, const double *value_boundary_coeffs,
                                      const double sign, const double *A_csr_input, const double *b_input,
                                      double *A_csr_output, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int cell_index = boundary_cell_id[cell_offset];
    int loop_size = boundary_cell_offset[index + 1] - cell_offset;

    int row_index = csr_row_index[cell_index];
    int diag_index = csr_diag_index[cell_index];
    int csr_index = row_index + diag_index;

    // construct internalCoeffs & boundaryCoeffs
    double internal_coeffs = 0;
    double boundary_coeffs = 0;
    for (int i = 0; i < loop_size; i++)
    {
        internal_coeffs += value_internal_coeffs[cell_offset + i];
        boundary_coeffs += value_boundary_coeffs[cell_offset + i];
    }
    A_csr_output[csr_index] = A_csr_input[csr_index] + internal_coeffs * sign;
    b_output[cell_index] = b_input[cell_index] + boundary_coeffs * sign;
}

__global__ void eeqn_fvm_laplacian_uncorrected_internal(int num_cells,
                                                        const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                                        const double *alphaEff, const double *weight,
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

    double own_alphaEff = alphaEff[index];
    // fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm.negSumDiag();
    double sum_diag = 0;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[i + row_index];
        double w = weight[neighbor_index];
        double nei_alphaEff = alphaEff[neighbor_cell_id];
        double gamma = w * (nei_alphaEff - own_alphaEff) + own_alphaEff;
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
        double nei_alphaEff = alphaEff[neighbor_cell_id];
        double gamma = w * (nei_alphaEff - own_alphaEff) + own_alphaEff;
        double gamma_magsf = gamma * magsf[neighbor_index];
        double coeff = gamma_magsf * distance[neighbor_index];
        A_csr_output[row_index + i] = A_csr_input[row_index + i] + coeff * sign;
        sum_diag += (-coeff);
    }
    A_csr_output[row_index + diag_index] = A_csr_input[row_index + diag_index] + sum_diag * sign; // diag
}

__global__ void eeqn_fvm_laplacian_uncorrected_boundary(int num_boundary_cells,
                                                        const int *csr_row_index, const int *csr_diag_index,
                                                        const int *boundary_cell_offset, const int *boundary_cell_id,
                                                        const double *boundary_alphaEff, const double *boundary_magsf,
                                                        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
                                                        const double sign, const double *A_csr_input, const double *b_input,
                                                        double *A_csr_output, double *b_output)
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
    double internal_coeffs = 0;
    double boundary_coeffs = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double gamma = boundary_alphaEff[i];
        double gamma_magsf = gamma * boundary_magsf[i];
        internal_coeffs += gamma_magsf * gradient_internal_coeffs[i];
        boundary_coeffs += gamma_magsf * gradient_boundary_coeffs[i];
    }

    A_csr_output[csr_index] = A_csr_input[csr_index] + internal_coeffs * sign;
    b_output[cell_index] = b_input[cell_index] + boundary_coeffs * sign;
}

__global__ void eeqn_fvc_ddt_kernel(int num_cells, const double rdelta_t,
                                    const double *rho_old, const double *rho_new,
                                    const double *K_old, const double *K,
                                    const double *volume,
                                    const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double fvc_ddt_term = rdelta_t * (rho_new[index] * K[index] - rho_old[index] * K_old[index]) * volume[index];
    b_output[index] = b_input[index] + fvc_ddt_term * sign;
}

__global__ void eeqn_fvc_div_vector_internal(int num_cells,
                                             const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                             const double *sf, const double *vf, const double *tlambdas,
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

    double own_vf_x = vf[index * 3 + 0];
    double own_vf_y = vf[index * 3 + 1];
    double own_vf_z = vf[index * 3 + 2];
    double sum = 0;
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
        sum -= sf_x * face_x + sf_y * face_y + sf_z * face_z;
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
        sum += sf_x * face_x + sf_y * face_y + sf_z * face_z;
    }
    b_output[index] = b_input[index] + sum * sign;
}

__global__ void eeqn_fvc_div_vector_boundary(int num_boundary_cells,
                                             const int *boundary_cell_offset, const int *boundary_cell_id,
                                             const double *boundary_sf, const double *boundary_vf,
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
    double sum = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        double sf_x = boundary_sf[i * 3 + 0];
        double sf_y = boundary_sf[i * 3 + 1];
        double sf_z = boundary_sf[i * 3 + 2];
        double face_x = boundary_vf[i * 3 + 0];
        double face_y = boundary_vf[i * 3 + 1];
        double face_z = boundary_vf[i * 3 + 2];

        // if not coupled
        sum += (sf_x * face_x + sf_y * face_y + sf_z * face_z);
    }
    b_output[cell_index] = b_input[cell_index] + sum * sign;
}

__global__ void eeqn_fvc_div_phi_scalar_internal(int num_cells,
                                                 const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                                 const double *weight, const double *phi, const double *K,
                                                 const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_k = K[index];
    double interp = 0;
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index;
            double w = weight[neighbor_index];
            double p = phi[neighbor_index];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_k = K[neighbor_cell_id];
            double face_k = (1 - w) * own_cell_k + w * neighbor_cell_k;
            interp -= p * face_k;
        }
        // upper
        if (inner_index > diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = weight[neighbor_index];
            double p = phi[neighbor_index];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_k = K[neighbor_cell_id];
            double face_k = w * own_cell_k + (1 - w) * neighbor_cell_k;
            interp += p * face_k;
        }
    }
    b_output[index] = b_input[index] + interp * sign;
}

__global__ void eeqn_fvc_div_phi_scalar_boundary(int num_boundary_cells,
                                                 const int *boundary_cell_offset, const int *boundary_cell_id,
                                                 const double *boundary_phi, const double *boundary_K,
                                                 const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // boundary interplate
    double interp = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        interp += boundary_phi[i] * boundary_K[i];
    }

    b_output[cell_index] = b_input[cell_index] + interp * sign;
}

__global__ void eeqn_add_to_source_kernel(int num_cells,
                                          const double sign_dpdt, const double *dpdt,
                                          const double sign_diffAlphaD, const double *diffAlphaD,
                                          const double *volume,
                                          const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    b_output[index] = b_input[index] + sign_dpdt * dpdt[index] * volume[index] + sign_diffAlphaD * diffAlphaD[index] * volume[index];
}

__global__ void eeqn_boundaryPermutation(const int num_boundary_faces, const int *bouPermedIndex,
                                         const double *boundary_K_init,
                                        //  const double *boundary_alphaEff_init, 
                                         const double *boundary_gradient_init,
                                         double *boundary_K,
                                        //  double *boundary_alphaEff,
                                         double *boundary_gradient)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    int p = bouPermedIndex[index];

    boundary_K[index] = boundary_K_init[p];
    // boundary_alphaEff[index] = boundary_alphaEff_init[p];
    boundary_gradient[index] = boundary_gradient_init[p];
}

__global__ void eeqn_update_BoundaryCoeffs_kernel(int num_boundary_faces, const double *boundary_phi,
                                                  double *gradient, const double *boundary_deltaCoeffs,
                                                  double *internal_coeffs,
                                                  double *boundary_coeffs, double *laplac_internal_coeffs,
                                                  double *laplac_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_faces)
        return;

    double grad = gradient[index];
    // energyGradient
    double valueInternalCoeffs = 1.;
    double valueBoundaryCoeffs = grad / boundary_deltaCoeffs[index];
    double gradientInternalCoeffs = 0.;
    double gradientBoundaryCoeffs = grad;

    internal_coeffs[index] = boundary_phi[index] * valueInternalCoeffs;
    boundary_coeffs[index] = -boundary_phi[index] * valueBoundaryCoeffs;
    laplac_internal_coeffs[index] = gradientInternalCoeffs;
    laplac_boundary_coeffs[index] = gradientBoundaryCoeffs;
}

// constructor
dfEEqn::dfEEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile)
    : dataBase_(dataBase)
{
    ESolver = new AmgXSolver(modeStr, cfgFile);

    stream = dataBase_.stream;
    // checkCudaErrors(cudaEventCreate(&event));

    num_cells = dataBase_.num_cells;
    cell_bytes = dataBase_.cell_bytes;
    num_faces = dataBase_.num_faces;
    cell_vec_bytes = dataBase_.cell_vec_bytes;
    csr_value_vec_bytes = dataBase_.csr_value_vec_bytes;
    num_boundary_cells = dataBase_.num_boundary_cells;
    num_surfaces = dataBase_.num_surfaces;
    num_boundary_faces = dataBase_.num_boundary_faces;
    boundary_face_bytes = dataBase_.boundary_face_bytes;

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;
    d_A_csr_col_index = dataBase_.d_A_csr_col_index;

    h_A_csr = new double[(num_cells + num_faces) * 3];
    h_b = new double[num_cells * 3];
    cudaMallocHost(&h_he_new, cell_bytes);

    checkCudaErrors(cudaMalloc((void **)&d_A_csr, csr_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, cell_vec_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_he_old, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_K, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_K_old, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_alphaEff, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_dpdt, cell_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_boundary_K_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_K, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_alphaEff_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_alphaEff, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_gradient_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_boundary_gradient, boundary_face_bytes));

    checkCudaErrors(cudaMalloc((void **)&d_value_internal_coeffs_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_value_boundary_coeffs_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_gradient_internal_coeffs_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_gradient_boundary_coeffs_init, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_value_internal_coeffs, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_value_boundary_coeffs, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_gradient_internal_coeffs, boundary_face_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_gradient_boundary_coeffs, boundary_face_bytes));
}

void dfEEqn::prepare_data(const double *he_old, const double *K, const double *K_old, const double *alphaEff,
                          const double *dpdt, const double *boundary_K, const double *boundary_alphaEff, const double *boundary_gradient)
{
    // TODO not real async copy now, because some host array are not in pinned memory.

    // copy the host input array in host memory to the device input array in device memory
    checkCudaErrors(cudaMemcpyAsync(d_he_old, he_old, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_K, K, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_K_old, K_old, cell_bytes, cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_alphaEff, alphaEff, cell_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_dpdt, dpdt, cell_bytes, cudaMemcpyHostToDevice, stream));

    // copy and permutate boundary variable
    checkCudaErrors(cudaMemcpyAsync(d_boundary_K_init, boundary_K, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    // checkCudaErrors(cudaMemcpyAsync(d_boundary_alphaEff_init, boundary_alphaEff, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_gradient_init, boundary_gradient, boundary_face_bytes, cudaMemcpyHostToDevice, stream));

    // UnityLewis
    d_alphaEff = dataBase_.d_alpha;
    d_boundary_alphaEff = dataBase_.d_boundary_alpha;

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_boundary_faces + threads_per_block - 1) / threads_per_block;
    eeqn_boundaryPermutation<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_faces, dataBase_.d_bouPermedIndex,
                                                                                d_boundary_K_init, d_boundary_gradient_init,
                                                                                d_boundary_K, d_boundary_gradient);
}

void dfEEqn::initializeTimeStep()
{
    // initialize matrix value
    checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));
    // initialize boundary coeffs
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (dataBase_.num_boundary_faces + threads_per_block - 1) / threads_per_block;
    eeqn_update_BoundaryCoeffs_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase_.num_boundary_faces, dataBase_.d_boundary_phi,
                                                                                         d_boundary_gradient, dataBase_.d_boundary_deltaCoeffs,
                                                                                         d_value_internal_coeffs, d_value_boundary_coeffs,
                                                                                         d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
}

void dfEEqn::fvm_ddt()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvm_ddt_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.rdelta_t,
                                                                           d_A_csr_row_index, d_A_csr_diag_index,
                                                                           dataBase_.d_rho_old, dataBase_.d_rho_new, dataBase_.d_volume, d_he_old,
                                                                           1., d_A_csr, d_b, d_A_csr, d_b);
}

void dfEEqn::fvm_div()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvm_div_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                             d_A_csr_row_index, d_A_csr_diag_index,
                                                                             dataBase_.d_weight, dataBase_.d_phi,
                                                                             1., d_A_csr, d_b, d_A_csr, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvm_div_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                                                                             d_A_csr_row_index, d_A_csr_diag_index,
                                                                             dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                             d_value_internal_coeffs, d_value_boundary_coeffs,
                                                                             1., d_A_csr, d_b, d_A_csr, d_b);
}

void dfEEqn::fvm_laplacian()
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvm_laplacian_uncorrected_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                               d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index, d_alphaEff, dataBase_.d_weight,
                                                                                               dataBase_.d_face, dataBase_.d_deltaCoeffs,
                                                                                               -1., d_A_csr, d_A_csr);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvm_laplacian_uncorrected_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                                                                                               d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                               d_boundary_alphaEff, dataBase_.d_boundary_face, d_gradient_internal_coeffs, d_gradient_boundary_coeffs,
                                                                                               -1., d_A_csr, d_b, d_A_csr, d_b);
}

void dfEEqn::fvc_ddt()
{
    // " + fvc::ddt(rho，K)" is on the left side of "==", thus should minus from source.
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvc_ddt_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.rdelta_t,
                                                                           dataBase_.d_rho_old, dataBase_.d_rho_new, d_K_old, d_K, dataBase_.d_volume,
                                                                           -1., d_b, d_b);
}

void dfEEqn::fvc_div_vector()
{
    // " + fvc::div(hDiffCorrFlux)" is on the right side of "==", thus should add to source.
    size_t threads_per_block = 512;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvc_div_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                    d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                                    dataBase_.d_face_vector, dataBase_.d_hDiffCorrFlux, dataBase_.d_weight,
                                                                                    1., d_b, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvc_div_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                                                                                    dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                    dataBase_.d_boundary_face_vector, dataBase_.d_boundary_hDiffCorrFlux,
                                                                                    1., d_b, d_b);
}

void dfEEqn::fvc_div_phi_scalar()
{
    // " + fvc::div(phi，K)" is on the left side of "==", thus should minus from source.
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvc_div_phi_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                        d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
                                                                                        dataBase_.d_weight, dataBase_.d_phi, d_K,
                                                                                        -1., d_b, d_b);
    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    eeqn_fvc_div_phi_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_cells,
                                                                                        dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                                        dataBase_.d_boundary_phi, d_boundary_K,
                                                                                        -1., d_b, d_b);
}

void dfEEqn::add_to_source()
{
    // " - dpdt" is on the left side of "==", thus should add to source.
    // "+ diffAlphaD" is on the left side of "==", thus should minus from source.
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    // " + fvc::ddt(rho，K)" is on the left side of "==", thus should minus from source.
    eeqn_add_to_source_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
                                                                                 // 1., d_dpdt, -1., d_diffAlphaD, dataBase_.d_volume, d_b, d_b);
                                                                                 1., d_dpdt, -1., dataBase_.d_diffAlphaD, dataBase_.d_volume, d_b, d_b);
}

void dfEEqn::checkValue(bool print)
{
    checkCudaErrors(cudaMemcpyAsync(h_A_csr, d_A_csr, csr_value_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_b, d_b, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));

    // Synchronize stream
    checkCudaErrors(cudaStreamSynchronize(stream));
    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            fprintf(stderr, "h_A_csr[%d]: %.16lf\n", i, h_A_csr[i]);
        for (int i = 0; i < num_cells; i++)
            fprintf(stderr, "h_b[%d]: %.16lf\n", i, h_b[i]);
    }

    char *input_file = "of_output.txt";
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

    std::vector<double> h_A_of_init_vec(num_cells + num_faces);
    std::copy(of_A, of_A + num_cells + num_faces, h_A_of_init_vec.begin());

    std::vector<double> h_A_of_vec_1mtx(num_faces + num_cells, 0);
    for (int i = 0; i < num_faces + num_cells; i++)
    {
        h_A_of_vec_1mtx[i] = h_A_of_init_vec[dataBase_.tmpPermutatedList[i]];
    }

    // b
    std::vector<double> h_b_of_vec(num_cells);
    std::copy(of_b, of_b + num_cells, h_b_of_vec.begin());

    if (print)
    {
        for (int i = 0; i < (num_faces + num_cells); i++)
            fprintf(stderr, "h_A_of_vec_1mtx[%d]: %.16lf\n", i, h_A_of_vec_1mtx[i]);
        for (int i = 0; i < num_cells; i++)
            fprintf(stderr, "h_b_of_vec[%d]: %.16lf\n", i, h_b_of_vec[i]);
    }

    // check
    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual(num_faces + num_cells, h_A_of_vec_1mtx.data(), h_A_csr, 1e-6);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(num_cells, h_b_of_vec.data(), h_b, 1e-6);
}

void dfEEqn::solve()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    // nvtxRangePush("solve");

    int nNz = num_cells + num_faces; // matrix entries
    if (num_iteration == 0)          // first interation
    {
        printf("Initializing AmgX Linear Solver\n");
        ESolver->setOperator(num_cells, nNz, d_A_csr_row_index, d_A_csr_col_index, d_A_csr);
    }
    else
    {
        ESolver->updateOperator(num_cells, nNz, d_A_csr);
    }
    ESolver->solve(num_cells, d_he_old, d_b);
    num_iteration++;

    checkCudaErrors(cudaMemcpyAsync(h_he_new, d_he_old, cell_bytes, cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaEventRecord(event, stream));
    //  checkCudaErrors(cudaStreamSynchronize(stream));
    //  for (size_t i = 0; i < num_cells; i++)
    //      fprintf(stderr, "h_he_after[%d]: %.16lf\n", i, h_he_new[i]);
}

void dfEEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfEEqn::updatePsi(double *Psi)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    // checkCudaErrors(cudaEventSynchronize(event));
    memcpy(Psi, h_he_new, cell_bytes);
}

dfEEqn::~dfEEqn()
{
    delete h_A_csr;
    delete h_b;

    checkCudaErrors(cudaFreeHost(h_he_new));

    checkCudaErrors(cudaFree(d_A_csr));
    checkCudaErrors(cudaFree(d_b));

    checkCudaErrors(cudaFree(d_he_old));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_K_old));
    // checkCudaErrors(cudaFree(d_alphaEff));
    checkCudaErrors(cudaFree(d_dpdt));

    checkCudaErrors(cudaFree(d_boundary_K_init));
    checkCudaErrors(cudaFree(d_boundary_K));
    checkCudaErrors(cudaFree(d_boundary_alphaEff_init));
    checkCudaErrors(cudaFree(d_boundary_alphaEff));
    checkCudaErrors(cudaFree(d_boundary_gradient_init));
    checkCudaErrors(cudaFree(d_boundary_gradient));

    checkCudaErrors(cudaFree(d_value_internal_coeffs_init));
    checkCudaErrors(cudaFree(d_value_boundary_coeffs_init));
    checkCudaErrors(cudaFree(d_gradient_internal_coeffs_init));
    checkCudaErrors(cudaFree(d_gradient_boundary_coeffs_init));
    checkCudaErrors(cudaFree(d_value_internal_coeffs));
    checkCudaErrors(cudaFree(d_value_boundary_coeffs));
    checkCudaErrors(cudaFree(d_gradient_internal_coeffs));
    checkCudaErrors(cudaFree(d_gradient_boundary_coeffs));

    // checkCudaErrors(cudaEventDestroy(event));
}

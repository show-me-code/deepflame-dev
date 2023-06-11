#include "dfRhoEqn.H"

// kernel functions
__global__ void fvc_div_internal_rho(int num_cells, const int *csr_row_index,
                                     const int *csr_diag_index, const int *permedIndex, const double *phi_init,
                                     double *phi_out, const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double sum = 0;

    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int permute_index = permedIndex[neighbor_index];
        double phi = phi_init[permute_index];
        phi_out[neighbor_index] = phi;
        sum -= phi;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int permute_index = permedIndex[neighbor_index];
        double phi = phi_init[permute_index];
        phi_out[neighbor_index] = phi;
        sum += phi;
    }

    b_output[index] = b_input[index] + sum * sign;
}

__global__ void fvc_div_boundary_rho(int num_cells, int num_boundary_cells, const int *boundary_cell_offset,
                                     const int *boundary_cell_id, const int *bouPermedIndex, const double *boundary_phi_init,
                                     double *boundary_phi, const double sign, const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    double sum = 0;

    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        int permute_index = bouPermedIndex[i];
        double phi = boundary_phi_init[permute_index];
        boundary_phi[i] = phi;
        sum += phi;
    }

    b_output[cell_index] = b_input[cell_index] + sum * sign;
}

__global__ void fvm_ddt_rho(int num_cells, const double rdelta_t,
                            const double *rho_old, double *rho_new, const double *volume, const double *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double ddt_diag = rdelta_t * volume[index];
    double ddt_source = rdelta_t * rho_old[index] * volume[index];
    double source_sum = ddt_source - b[index];

    rho_new[index] = source_sum / ddt_diag;
}

// constructor
dfRhoEqn::dfRhoEqn(dfMatrixDataBase &dataBase)
    : dataBase_(dataBase)
{
    stream = dataBase_.stream;
    num_cells = dataBase_.num_cells;
    cell_bytes = dataBase_.cell_bytes;
    num_surfaces = dataBase_.num_surfaces;
    num_boundary_cells = dataBase_.num_boundary_cells;

    d_A_csr_row_index = dataBase_.d_A_csr_row_index;
    d_A_csr_diag_index = dataBase_.d_A_csr_diag_index;

    cudaMallocHost(&h_psi, cell_bytes);

    checkCudaErrors(cudaMalloc((void **)&d_b, cell_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_psi, cell_bytes));
}

void dfRhoEqn::initializeTimeStep()
{
    // initialize matrix value
    checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_bytes, stream));
}

void dfRhoEqn::fvc_div(double *phi, double *boundary_phi_init)
{
    memcpy(dataBase_.h_phi_init, phi, num_surfaces * sizeof(double));

    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi_init, dataBase_.h_phi_init, num_surfaces * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi_init + num_surfaces, dataBase_.d_phi_init, num_surfaces * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi_init, boundary_phi_init, dataBase_.boundary_face_bytes, cudaMemcpyHostToDevice, stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_internal_rho<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, d_A_csr_row_index, d_A_csr_diag_index, dataBase_.d_permedIndex,
                                                                            dataBase_.d_phi_init, dataBase_.d_phi, 1., d_b, d_b);

    blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_div_boundary_rho<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells, dataBase_.d_boundary_cell_offset, dataBase_.d_boundary_cell_id,
                                                                            dataBase_.d_bouPermedIndex, dataBase_.d_boundary_phi_init, dataBase_.d_boundary_phi, 1., d_b, d_b);
}

void dfRhoEqn::fvm_ddt(double *rho_old)
{
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_rho_old, rho_old, cell_bytes, cudaMemcpyHostToDevice, stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_ddt_rho<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, dataBase_.rdelta_t, dataBase_.d_rho_old, dataBase_.d_rho_new, dataBase_.d_volume, d_b);
    checkCudaErrors(cudaMemcpyAsync(h_psi, dataBase_.d_rho_new, cell_bytes, cudaMemcpyDeviceToHost, stream));
}

void dfRhoEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfRhoEqn::updatePsi(double *Psi)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < num_cells; i++)
        Psi[i] = h_psi[i];
}
dfRhoEqn::~dfRhoEqn(){}

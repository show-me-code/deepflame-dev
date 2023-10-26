#include "dfChemistrySolver.H"

__global__ void construct_init_input(int num_cells, int dim, const double *T, const double *p,
        const double *y, double *y_input_BCT, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    output[index * dim] = T[index];
    output[index * dim + 1] = p[index];
    double y_BCT;
    for (int i = 0; i < dim - 2; ++i) {
        y_BCT = (pow(y[i * num_cells + index], 0.1) - 1) * 10; // BCT: lambda = 0.1
        output[index * dim + 2 + i] = y_BCT;
        y_input_BCT[i * num_cells + index] = y_BCT;
    }
}

__global__ void normalize_input(int num_cells, int dim, const double *input, 
        const double *Xmu, const double *Xstd, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    for (int i = 0; i < dim; ++i) {
        output[index * dim + i] = (input[index * dim + i] - Xmu[i]) / Xstd[i];
    }
}

__global__ void calculate_y_new(int num_cells, int num_species, const double *output_init, 
        const double *y_input_BCT, const double *Ymu, const double *Ystd, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double RR_tmp;
    for (int i = 0; i < num_species; ++i) {
        RR_tmp = output_init[i * num_cells + index] * Ystd[i] + Ymu[i] + y_input_BCT[i * num_cells + index];
        RR_tmp = pow((RR_tmp * 0.1 + 1), 10); // rev-BCT: lambda = 0.1
        output[i * num_cells + index] = RR_tmp;
    }
}

__global__ void calculate_RR(int num_cells, int num_species, double delta_t,
        const double *rho, const double *y_old, double *y_new, double *RR)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    // normalize
    double y_ave = 0.;
    for (int i = 0; i < num_species; ++i) {
        y_ave += y_new[i * num_cells + index];
    }
    for (int i = 0; i < num_species; ++i) {
        y_new[i * num_cells + index] = y_new[i * num_cells + index] / y_ave;
        RR[i * num_cells + index] = (y_new[i * num_cells + index] - y_old[i * num_cells + index]) * rho[index] / delta_t;
    }
}

dfChemistrySolver::dfChemistrySolver(int num_cells, int num_species)
    : device_(torch::kCUDA), num_cells_(num_cells), num_species_(num_species)
{
    dim_input_ = num_species + 2; // p, T, y
    cudaMalloc(&init_input_, sizeof(double) * num_cells * dim_input_);
    cudaMalloc(&y_input_BCT, sizeof(double) * num_cells * num_species_);
    cudaMalloc(&Xmu_, sizeof(double) * dim_input_);
    cudaMalloc(&Xstd_, sizeof(double) * dim_input_);
    cudaMalloc(&Ymu_, sizeof(double) * num_species_);
    cudaMalloc(&Ystd_, sizeof(double) * num_species_);
    modules_.reserve(num_species_);

    // now norm paras are set in constructor manually
    at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA);
    std::vector<double> Xmu_vec = {2.1996457626e+03,  1.0281762632e+05, -5.2533840071e+00,
                -4.5623405933e+00, -1.5305375824e+00, -3.7453838144e+00,
                -3.3097212360e+00, -4.3339657954e+00, -2.9028421546e-01};
    std::vector<double> Xstd_vec = {3.8304388695e+02, 7.1905832810e+02, 4.6788389177e-01, 
                5.3034657693e-01, 5.0325864222e-01, 5.1058993718e-01, 
                7.9704668543e-01, 5.1327160125e-01, 4.6827591193e-03};
    std::vector<double> Ymu_vec = {0.0085609265, -0.0082998877, 0.0030108739,
                -0.0067360325, 0.0037464590, 0.0024258509};
    std::vector<double> Ystd_vec = {0.0085609265, -0.0082998877, 0.0030108739,
                -0.0067360325, 0.0037464590, 0.0024258509};
    
    cudaMemcpy(Xmu_, Xmu_vec.data(), sizeof(double) * dim_input_, cudaMemcpyHostToDevice);
    cudaMemcpy(Xstd_, Xstd_vec.data(), sizeof(double) * dim_input_, cudaMemcpyHostToDevice);
    cudaMemcpy(Ymu_, Ymu_vec.data(), sizeof(double) * num_species_, cudaMemcpyHostToDevice);
    cudaMemcpy(Ystd_, Ystd_vec.data(), sizeof(double) * num_species_, cudaMemcpyHostToDevice);
}

dfChemistrySolver::~dfChemistrySolver() {
    cudaFree(init_input_);
}

void dfChemistrySolver::Inference(const double *T, const double *p, const double *y,
        const double *rho, double *RR) {
    // construct input
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells_ + threads_per_block - 1) / threads_per_block;
    construct_init_input<<<blocks_per_grid, threads_per_block>>>(num_cells_, dim_input_, T, p, y, y_input_BCT, init_input_);
    normalize_input<<<blocks_per_grid, threads_per_block>>>(num_cells_, dim_input_, init_input_, Xmu_, Xstd_, init_input_);

    // inference by torch
    at::Tensor torch_input = torch::from_blob(init_input_, {num_cells_, dim_input_}, device_);
    torch_input = torch_input.to(at::kFloat);
    std::vector<torch::jit::IValue> INPUTS;
    INPUTS.push_back(torch_input);

    std::vector<at::Tensor> output(num_species_);
    for (int i = 0; i < num_species_; ++i) {
        output[i] = modules_[i].forward(INPUTS).toTensor();
        output[i] = output[i].to(at::kDouble);
    }

    // post process
    double *d_output;
    for (int i = 0; i < num_species_; ++i) {
        d_output = output[i].data_ptr<double>();
        cudaMemcpy(RR + i * num_cells_, d_output, sizeof(double) * num_cells_, cudaMemcpyDeviceToDevice);
    }
    calculate_y_new<<<blocks_per_grid, threads_per_block>>>(num_cells_, num_species_, RR, y_input_BCT, Ymu_, Ystd_, RR);
    calculate_RR<<<blocks_per_grid, threads_per_block>>>(num_cells_, num_species_, 1e-6, rho, y, RR, RR);
}
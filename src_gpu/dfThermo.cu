#include "dfThermo.H"
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cassert>
#include <cstring>
#include "device_launch_parameters.h"

#define TICK_INIT_EVENT \
    float time_elapsed_kernel=0;\
    cudaEvent_t start_kernel, stop_kernel;\
    checkCudaErrors(cudaEventCreate(&start_kernel));\
    checkCudaErrors(cudaEventCreate(&stop_kernel));

#define TICK_START_EVENT \
    checkCudaErrors(cudaEventRecord(start_kernel,0));

#define TICK_END_EVENT(prefix) \
    checkCudaErrors(cudaEventRecord(stop_kernel,0));\
    checkCudaErrors(cudaEventSynchronize(start_kernel));\
    checkCudaErrors(cudaEventSynchronize(stop_kernel));\
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed_kernel,start_kernel,stop_kernel));\
    printf("try %s 执行时间：%lf(ms)\n", #prefix, time_elapsed_kernel);

#define GAS_CANSTANT 8314.46261815324
#define SQRT8 2.8284271247461903
#define NUM_SPECIES 7

// constant memory
__constant__ __device__ double d_nasa_coeffs[NUM_SPECIES*15];
__constant__ __device__ double d_viscosity_coeffs[NUM_SPECIES*5];
__constant__ __device__ double d_thermal_conductivity_coeffs[NUM_SPECIES*5];
__constant__ __device__ double d_binary_diffusion_coeffs[NUM_SPECIES*NUM_SPECIES*5];
__constant__ __device__ double d_molecular_weights[NUM_SPECIES];
__constant__ __device__ double d_viscosity_conatant1[NUM_SPECIES*NUM_SPECIES];
__constant__ __device__ double d_viscosity_conatant2[NUM_SPECIES*NUM_SPECIES];

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights)
{
    double *d_tmp;
    checkCudaErrors(cudaMalloc((void**)&d_tmp, sizeof(double) * NUM_SPECIES * 15));
    double nasa_coeffs_tmp[NUM_SPECIES*15];
    double viscosity_coeffs_tmp[NUM_SPECIES*5];
    double thermal_conductivity_coeffs_tmp[NUM_SPECIES*5];
    double binary_diffusion_coeffs_tmp[NUM_SPECIES*NUM_SPECIES*5];
    double viscosity_conatant1_tmp[NUM_SPECIES*NUM_SPECIES];
    double viscosity_conatant2_tmp[NUM_SPECIES*NUM_SPECIES];

    for (int i = 0; i < NUM_SPECIES; i++) {
        std::copy(nasa_coeffs[i].begin(), nasa_coeffs[i].end(), nasa_coeffs_tmp + i * 15);
        std::copy(viscosity_coeffs[i].begin(), viscosity_coeffs[i].end(), viscosity_coeffs_tmp + i * 5);
        std::copy(thermal_conductivity_coeffs[i].begin(), thermal_conductivity_coeffs[i].end(), thermal_conductivity_coeffs_tmp + i * 5);
        std::copy(binary_diffusion_coeffs[i].begin(), binary_diffusion_coeffs[i].end(), binary_diffusion_coeffs_tmp + i * 5 * NUM_SPECIES);
        for (int j = 0; j < NUM_SPECIES; j++) {
            viscosity_conatant1_tmp[i * NUM_SPECIES + j] = pow((1 + molecular_weights[i] / molecular_weights[j]), -0.5);
            viscosity_conatant2_tmp[i * NUM_SPECIES + j] = pow(molecular_weights[j] / molecular_weights[i], 0.25);
        }
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_nasa_coeffs, nasa_coeffs_tmp, sizeof(double) * 15 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_coeffs, viscosity_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_thermal_conductivity_coeffs, thermal_conductivity_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_binary_diffusion_coeffs, binary_diffusion_coeffs_tmp, sizeof(double) * 5 * NUM_SPECIES * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_molecular_weights, molecular_weights.data(), sizeof(double) * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant1, viscosity_conatant1_tmp, sizeof(double) * NUM_SPECIES * NUM_SPECIES));
    checkCudaErrors(cudaMemcpyToSymbol(d_viscosity_conatant2, viscosity_conatant2_tmp, sizeof(double) * NUM_SPECIES * NUM_SPECIES));
}

__global__ void get_mole_fraction_mean_mole_weight(int num_cells, int num_species, const double *d_Y, 
        double *mole_fraction, double *mean_mole_weight)
{   
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double sum = 0.;
    for (int i = 0; i < num_species; i++) {
        sum += d_Y[i * num_cells + index] / d_molecular_weights[i];
    }
    double meanMoleWeight = 0.;
    for (int i = 0; i < num_species; i++) {
        mole_fraction[i * num_cells + index] = d_Y[i * num_cells + index] / (d_molecular_weights[i] * sum);
        meanMoleWeight += mole_fraction[i * num_cells + index] * d_molecular_weights[i];
    }
    mean_mole_weight[index] = meanMoleWeight;
}

__global__ void calculate_TPoly_kernel(int num_cells, const double *T, double *d_T_poly)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    d_T_poly[num_cells * 0 + index] = 1.0;
    d_T_poly[num_cells * 1 + index] = log(T[index]);
    d_T_poly[num_cells * 2 + index] = d_T_poly[num_cells * 1 + index] * d_T_poly[num_cells * 1 + index];
    d_T_poly[num_cells * 3 + index] = d_T_poly[num_cells * 1 + index] * d_T_poly[num_cells * 2 + index];
    d_T_poly[num_cells * 4 + index] = d_T_poly[num_cells * 2 + index] * d_T_poly[num_cells * 2 + index];
}

__global__ void calculate_psi_kernel(int num_cells, const double *T, const double *mean_mole_weight,
        double *psi)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    psi[index] = mean_mole_weight[index] / (GAS_CANSTANT * T[index]);
}

__global__ void calculate_rho_kernel(int num_cells, const double *p, const double *psi, double *rho)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    rho[index] = p[index] * psi[index];
}

__global__ void calculate_viscosity_kernel(int num_cells, int num_species,
        const double *T_poly, const double *T, const double *mole_fraction,
        double *species_viscosities, double *mu)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double dot_product;
    double local_T = T[index];

    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += d_viscosity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
        }
        species_viscosities[i * num_cells + index] = (dot_product * dot_product) * sqrt(local_T);
    }

    double mu_mix = 0.;
    for (int i = 0; i < num_species; i++) {
        double temp = 0.;
        for (int j = 0; j < num_species; j++) {
            temp += mole_fraction[num_cells * j + index] / SQRT8 *
            d_viscosity_conatant1[i * NUM_SPECIES + j] * // constant 1
            pow(1.0 + sqrt(species_viscosities[i * num_cells + index] / species_viscosities[j * num_cells + index]) *
            d_viscosity_conatant2[i * NUM_SPECIES + j], 2.0); // constant 2
        }
        mu_mix += mole_fraction[num_cells * i + index] * species_viscosities[i * num_cells + index] / temp;
    }

    mu[index] = mu_mix;
}

__device__ double calculate_cp_device_kernel(int num_cells, int num_species, int index, 
        const double local_T, const double *mass_fraction)
{   
    double cp = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            cp += mass_fraction[i * num_cells + index] * (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T + d_nasa_coeffs[i * 15 + 3] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        } else {
            cp += mass_fraction[i * num_cells + index] * (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T + d_nasa_coeffs[i * 15 + 10] * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T + 
                    d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T) * GAS_CANSTANT / d_molecular_weights[i];
        }
    }
    return cp;
}

__global__ void calculate_thermoConductivity_kernel(int num_cells, int num_species, 
        const double *nasa_coeffs, const double *mass_fraction,
        const double *T_poly, const double *T, const double *mole_fraction,
        double *species_thermal_conductivities, double *thermal_conductivity)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double dot_product;
    double local_T = T[index];

    for (int i = 0; i < num_species; i++) {
        dot_product = 0.;
        for (int j = 0; j < 5; j++) {
            dot_product += d_thermal_conductivity_coeffs[i * 5 + j] * T_poly[num_cells * j + index];
        }
        species_thermal_conductivities[i * num_cells + index] = dot_product * sqrt(local_T);
    }

    double sum_conductivity = 0.;
    double sum_inv_conductivity = 0.;

    for (int i = 0; i < num_species; i++) {
        sum_conductivity += mole_fraction[num_cells * i + index] * species_thermal_conductivities[i * num_cells + index];
        sum_inv_conductivity += mole_fraction[num_cells * i + index] / species_thermal_conductivities[i * num_cells + index];
    }
    double lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);

    double cp = calculate_cp_device_kernel(num_cells, num_species, index, local_T, mass_fraction);

    thermal_conductivity[index] = lambda_mix / cp;
}

__device__ double calculate_enthalpy_device_kernel(int num_cells, int num_species, int index, const double local_T,
        const double *mass_fraction)
{
    double h = 0.;

    for (int i = 0; i < num_species; i++) {
        if (local_T > d_nasa_coeffs[i * 15 + 0]) {
            h += (d_nasa_coeffs[i * 15 + 1] + d_nasa_coeffs[i * 15 + 2] * local_T / 2 + d_nasa_coeffs[i * 15 + 3] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 4] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 5] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 6] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_cells + index];
        } else {
            h += (d_nasa_coeffs[i * 15 + 8] + d_nasa_coeffs[i * 15 + 9] * local_T / 2 + d_nasa_coeffs[i * 15 + 10] * local_T * local_T / 3 + 
                    d_nasa_coeffs[i * 15 + 11] * local_T * local_T * local_T / 4 + d_nasa_coeffs[i * 15 + 12] * local_T * local_T * local_T * local_T / 5 + 
                    d_nasa_coeffs[i * 15 + 13] / local_T) * GAS_CANSTANT * local_T / d_molecular_weights[i] * mass_fraction[i * num_cells + index];
        }
    }
    return h;
}

__global__ void calculate_temperature_kernel(int num_cells, int num_species, 
        const double *T_init, const double *h_target, const double *mass_fraction,
        double *T_est, double atol, double rtol, int max_iter)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double local_T = T_init[index];
    double local_h_target = h_target[index];
    double h, cp, delta_T;
    for (int n = 0; n < max_iter; ++n) {
        h = calculate_enthalpy_device_kernel(num_cells, num_species, index, local_T, mass_fraction);
        cp = calculate_cp_device_kernel(num_cells, num_species, index, local_T, mass_fraction);
        delta_T = (h - local_h_target) / cp;
        local_T -= delta_T;
        if (fabs(h - local_h_target) < atol || fabs(delta_T / local_T) < rtol) {
            break;
        }
    }
    T_est[index] = local_T;
}

__global__ void calculate_enthalpy_kernel(int num, int offset, int num_cells, int num_species, 
        const double *T, const double *mass_fraction, double *enthalpy)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int startIndex = index + offset;
    
    enthalpy[startIndex] = calculate_enthalpy_device_kernel(num_cells, num_species, startIndex, T[startIndex], mass_fraction);
}

void dfThermo::setConstantValue(std::string mechanism_file, int num_cells, int num_species)
{
    this->mechanism_file = mechanism_file;
    this->num_cells = num_cells;
    this->num_species = num_species;
    // get thermo_coeff_file from mechanism_file
    std::string prefix = "thermo_";
    std::string suffix = ".txt";
    std::string baseName = std::filesystem::path(mechanism_file).stem().string();
    thermo_coeff_file = prefix + baseName + suffix;

    // check if thermo_coeff_file exists
    if (!std::filesystem::exists(thermo_coeff_file))
    {
        std::cout << "Thermo coefficient file does not exist!" << std::endl;
        exit(1);
    }

    // read binary file
    FILE *fp = NULL;
    char *c_thermo_file = new char[thermo_coeff_file.length() + 1];
    strcpy(c_thermo_file, thermo_coeff_file.c_str());

    fp = fopen(c_thermo_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", c_thermo_file);
        exit(EXIT_FAILURE);
    }

    fread(&num_species, sizeof(int), 1, fp);

    molecular_weights.resize(num_species);
    fread(molecular_weights.data(), sizeof(double), num_species, fp);

    mass_fraction.resize(num_species);
    mole_fraction.resize(num_species);

    initCoeffsfromBinaryFile(fp);

    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaMalloc((void**)&d_mass_fraction, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_mole_fraction, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_mean_mole_weight, sizeof(double) * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_T_poly, sizeof(double) * 5 * num_cells));

    checkCudaErrors(cudaMalloc((void**)&d_species_viscosities, sizeof(double) * num_species * num_cells));
    checkCudaErrors(cudaMalloc((void**)&d_species_thermal_conductivities, sizeof(double) * num_species * num_cells));
    std::cout << "dfThermo initialized" << std::endl;
}

void dfThermo::readCoeffsBinary(FILE* fp, int dimension, std::vector<std::vector<double>>& coeffs)
{
    coeffs.resize(num_species);
    for (int i = 0; i < num_species; i++) {
        coeffs[i].resize(dimension);
        fread(coeffs[i].data(), sizeof(double), dimension, fp);
    }
}

void dfThermo::initCoeffsfromBinaryFile(FILE* fp)
{
    readCoeffsBinary(fp, 15, nasa_coeffs);
    readCoeffsBinary(fp, 5, viscosity_coeffs);
    readCoeffsBinary(fp, 5, thermal_conductivity_coeffs);
    readCoeffsBinary(fp, num_species * 5, binary_diffusion_coeffs);
}

void dfThermo::sync()
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void dfThermo::setMassFraction(const double *d_mass_fraction)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;

    TICK_START_EVENT;
    get_mole_fraction_mean_mole_weight<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_mass_fraction, d_mole_fraction, d_mean_mole_weight);
    TICK_END_EVENT(get_mole_fraction_mean_mole_weight);
}

void dfThermo::setConstantFields(const std::vector<int> patch_type) 
{
    dataBase_.patch_type_T = patch_type;
}

void dfThermo::initNonConstantFields(double *h_T, double *h_he, double *h_boundary_T, double *h_boundary_he)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_T, h_T, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_he, h_he, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_T, h_boundary_T, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_he, h_boundary_he, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfThermo::calculateTPolyGPU(const double *T)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_TPoly_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_T_poly);
    TICK_END_EVENT(calculate_TPoly_kernel);
}

void dfThermo::calculatePsiGPU(const double *T, double *d_psi)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_psi_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, T, d_mean_mole_weight, d_psi);
    TICK_END_EVENT(calculate_psi_kernel);
}

void dfThermo::calculateRhoGPU(const double *p, const double *psi, double *rho)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_rho_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, p, psi, rho);
    TICK_END_EVENT(calculate_rho_kernel);
}

void dfThermo::calculateViscosityGPU(const double *T, double *viscosity)
{
    calculateTPolyGPU(T);

    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_viscosity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_T_poly, T, d_mole_fraction, d_species_viscosities, viscosity);
    TICK_END_EVENT(calculate_viscosity_kernel);
}

void dfThermo::calculateThermoConductivityGPU(const double *T, const double *d_mass_fraction, double *thermal_conductivity)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    TICK_START_EVENT;
    calculate_thermoConductivity_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            d_nasa_coeffs, d_mass_fraction, d_T_poly, T, d_mole_fraction,
            d_species_thermal_conductivities, thermal_conductivity);
    TICK_END_EVENT(calculate_thermoConductivity_kernel);
}

void dfThermo::calculateTemperatureGPU(const double *T_init, const double *target_h, double *T, const double *d_mass_fraction, double atol, 
            double rtol, int max_iter)
{
    TICK_INIT_EVENT;
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;

    TICK_START_EVENT;
    calculate_temperature_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_species, 
            T_init, target_h, d_mass_fraction, T, atol, rtol, max_iter);
    TICK_END_EVENT(calculate_temperature_kernel);
}

void dfThermo::calculateEnthalpyGPU(int num, const double *T, double *enthalpy, const double *d_mass_fraction, int offset)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num + threads_per_block - 1) / threads_per_block;

    calculate_enthalpy_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num, offset, num_cells, num_species, 
            T, d_mass_fraction, enthalpy);
}

void dfThermo::updateEnergy()
{
    calculateEnthalpyGPU(num_cells, dataBase_.d_T, dataBase_.d_he, dataBase_.d_y);
    
    // int offset = 0;
    // for (int i = 0; i < dataBase_.num_patches; i++) {
    //     calculateEnthalpyGPU(dataBase_.patch_size[i], dataBase_.d_boundary_T + offset, dataBase_.d_he + offset, dataBase_.d_y, offset);
    //     if (dataBase_.patch_type_T[i] == boundaryConditions::processor) {
    //         offset += 2 * dataBase_.patch_size[i];
    //     } else {
    //         offset += dataBase_.patch_size[i];
    //     }
    // }
}

void dfThermo::compareThermoConductivity(const double *d_thermal_conductivity, const double *thermal_conductivity, 
        bool printFlag)
{
    std::vector<double> h_thermal_conductivity(num_cells);
    checkCudaErrors(cudaMemcpy(h_thermal_conductivity.data(), d_thermal_conductivity, sizeof(double) * num_cells, cudaMemcpyDeviceToHost));
    printf("compare thermal_conductivity\n");
    checkVectorEqual(num_cells, thermal_conductivity, h_thermal_conductivity.data(), 1e-14, printFlag);
}

void dfThermo::compareViscosity(const double *d_viscosity, const double *viscosity, bool printFlag)
{
    std::vector<double> h_viscosity(num_cells);
    checkCudaErrors(cudaMemcpy(h_viscosity.data(), d_viscosity, sizeof(double) * num_cells, cudaMemcpyDeviceToHost));
    printf("compare viscosity\n");
    checkVectorEqual(num_cells, viscosity, h_viscosity.data(), 1e-14, printFlag);
}

void dfThermo::compareHe(const double *he)
{
    std::vector<double> h_he(num_cells);
    checkCudaErrors(cudaMemcpy(h_he.data(), dataBase_.d_he, sizeof(double) * num_cells, cudaMemcpyDeviceToHost));
    printf("compare he\n");
    checkVectorEqual(num_cells, he, h_he.data(), 1e-14, true);
}
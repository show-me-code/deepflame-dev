#pragma once

#include <stdio.h>
#include <unistd.h>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include "nvtx3/nvToolsExt.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <iostream>
#include <ctime>
#include <cmath>
#include <unordered_map>

#define DEBUG_TRACE fprintf(stderr, "%s %d\n", __FILE__, __LINE__);

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
        int const line) {
  if (result) {
    fprintf(stderr, "cuda error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void checkVectorEqual(int count, const double* basevec, double* vec, double max_relative_error, bool print = false) {
    for (size_t i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        if (print)
            fprintf(stderr, "index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
        // if (abs_diff > 1e-12 && rel_diff > max_relative_error && !std::isinf(rel_diff))
        if (abs_diff > 1e-15 && rel_diff > max_relative_error)
            fprintf(stderr, "mismatch index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }
}

enum location {
    cpu,
    gpu
};

enum position {
    internal,
    boundary
};

enum boundaryConditions{
    zeroGradient,
    fixedValue,
    coupled,
    empty
};

void constructBoundarySelectorPerPatch(int *patchTypeSelector, const std::string& patchTypeStr);
void constructBoundarySelector(std::vector<int>& patchTypeSelector, const std::string& patchTypeStr, const int patchSize);

struct dfMatrixDataBase
{
    // cuda resource
    cudaStream_t stream;
    // maybe one graph for one eqn before using self-developed solver
    // and should be located in each eqn.
    cudaGraph_t graph;
    cudaGraphExec_t graph_instance;
    bool graph_created=false;

    // constant values -- basic
    int num_cells = 0;
    int num_surfaces = 0;
    int num_boundary_surfaces = 0;
    int num_patches = 0;
    int num_species = 0;
    std::vector<int> patch_size;
    double rdelta_t = 0;

    // constant values -- ldu bytesize
    size_t cell_value_bytes = 0;
    size_t cell_value_vec_bytes = 0;
    size_t cell_value_tsr_bytes = 0;
    size_t cell_index_bytes = 0;
    size_t surface_value_bytes = 0;
    size_t surface_index_bytes = 0;
    size_t surface_value_vec_bytes = 0;
    size_t boundary_surface_value_bytes = 0;
    size_t boundary_surface_value_vec_bytes = 0;
    size_t boundary_surface_value_tsr_bytes = 0;
    size_t boundary_surface_index_bytes = 0;

    // constant values -- csr bytesize
    size_t csr_row_index_bytes = 0;
    size_t csr_col_index_bytes = 0;
    size_t csr_value_bytes = 0;
    size_t csr_value_vec_bytes = 0;

    // constant indexes
    int *d_owner = nullptr;
    int *d_neighbor = nullptr;
    int *d_lower_to_csr_index = nullptr;
    int *d_diag_to_csr_index= nullptr;
    int *d_upper_to_csr_index= nullptr;
    int *d_csr_row_index= nullptr;
    int *d_csr_col_index= nullptr;

    // constant fields - internal
    double *d_sf = nullptr;
    double *d_mag_sf = nullptr;
    double *d_weight = nullptr;
    double *d_delta_coeffs = nullptr;
    double *d_volume = nullptr;
    
    // constant fields - boundary
    double *d_boundary_sf = nullptr;
    double *d_boundary_mag_sf = nullptr;
    double *d_boundary_weight = nullptr;
    double *d_boundary_delta_coeffs = nullptr;

    // non-constant fields - internal 
    // TODO: further estimate
    // fields solved by eqns - new
    double *d_rho = nullptr;
    double *d_u = nullptr;
    double *d_y = nullptr;
    double *d_he = nullptr;
    double *d_p = nullptr;
    // fields solved by eqns - old 
    // TODO: not all fields need to store oldTime
    double *d_rho_old = nullptr;
    //double *d_u_old = nullptr;
    //double *d_y_old = nullptr;
    //double *d_he_old = nullptr;
    //double *d_p_old = nullptr;
    // other shared fields between eqns
    double *d_phi = nullptr;
    // computed on GPU, used on CPU, need memcpyd2h - host
    double *h_rho = nullptr;
    double *h_u= nullptr;
    double *h_y= nullptr;
    double *h_he= nullptr;
    // computed on CPU, used on GPU, need memcpyh2d - host
    double *h_p= nullptr;
    double *h_phi= nullptr;

    // non-constant fields - boundary
    // TODO: further estimate
    // fields solved by eqns - new
    double *d_boundary_rho = nullptr;
    double *d_boundary_u = nullptr;
    double *d_boundary_y = nullptr;
    double *d_boundary_he = nullptr;
    double *d_boundary_p = nullptr;
    // fields solved by eqns - old
    double *d_boundary_rho_old = nullptr;
    //double *d_boundary_u_old = nullptr;
    //double *d_boundary_y_old = nullptr;
    //double *d_boundary_he_old = nullptr;
    //double *d_boundary_p_old = nullptr;
    // other shared fields between eqns
    double *d_boundary_phi = nullptr;
    // computed on GPU, used on CPU, need memcpyd2h - host
    double *h_boundary_rho = nullptr;
    double *h_boundary_u= nullptr;
    double *h_boundary_y= nullptr;
    double *h_boundary_he= nullptr;
    // computed on CPU, used on GPU, need memcpyh2d - host
    double *h_boundary_p= nullptr;
    double *h_boundary_phi= nullptr;

    std::unordered_map<std::string, double*> fieldPointerMap;

    // constructor
    dfMatrixDataBase();

    // deconstructor
    ~dfMatrixDataBase();

    // member function
    void setConstantValues(int num_cells, int num_surfaces, int num_boundary_surfaces,
                   int num_patches, std::vector<int> patch_size,
                   int num_species, double rdelta_t);             
    void setConstantIndexes(const int *owner, const int *neighbor);

    void createConstantFieldsInternal();
    void createConstantFieldsBoundary();
    void initConstantFieldsInternal(const double *sf, const double *mag_sf, 
        const double *weight, const double *delta_coeffs, const double *volume);
    void initConstantFieldsBoundary(const double *boundary_sf, const double *boundary_mag_sf, 
        const double *boundary_delta_coeffs);

    void createNonConstantFieldsInternal();
    void createNonConstantFieldsBoundary();
    void initNonConstantFieldsInternal(const double *y);
    void initNonConstantFieldsBoundary(const double *boundary_y);

    // getter
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);
};


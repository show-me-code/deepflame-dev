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

inline void checkVectorEqual(int count, const double* basevec, double* vec, double max_relative_error) {
    for (size_t i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        // if (abs_diff > 1e-12 && rel_diff > max_relative_error && !std::isinf(rel_diff))
        if (abs_diff > 1e-15 && rel_diff > max_relative_error)
            fprintf(stderr, "mismatch index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }
}


struct dfMatrixDataBase
{
    // - cuda resource
    cudaStream_t stream;

    // - number of cell size
    int num_cells;
    // - number of face size
    int num_surfaces;
    // - number of offdiagnal entry size (2*num_surfaces)
    int num_faces;
    // - number of boundary cells
    int num_boundary_cells;
    // - number of boundary faces
    int num_boundary_faces;

    int num_species;

    // - mesh variables
    // - csr_row_index
    int *h_A_csr_row_index=nullptr, *d_A_csr_row_index=nullptr;
    // - csr_col_index
    int *h_A_csr_col_index=nullptr, *d_A_csr_col_index=nullptr;
    // - csr_diag_index
    int *h_A_csr_diag_index=nullptr, *d_A_csr_diag_index=nullptr;

    // - the pre-permutated and post-permutated interpolation weight list
    std::vector<double> h_weight_vec_init, h_weight_vec;
    // - the pre-permutated and post-permutated flux (phi) list
    std::vector<double> h_phi_vec_init, h_phi_vec;
    // - the pre-permutated and post-permutated cell face vector list
    std::vector<double> h_face_vector_vec_init, h_face_vector_vec;
    std::vector<double> h_face_vec_init, h_face_vec;
    std::vector<double> h_deltaCoeffs_vec_init, h_deltaCoeffs_vec;
    // - the host pointer to rho_new, rho_old, velocity_old, pressure and volume list
    double *h_rho_new = nullptr, *h_rho_old = nullptr, *h_velocity_old = nullptr, 
    *h_pressure = nullptr;
    const double *h_volume = nullptr;
    // - the host pointer to the pre-permutated and post-permutated interpolation weight list
    double *h_weight_init = nullptr, *h_weight = nullptr;
    // - the host pointer to the pre-permutated and post-permutated flux (phi) list
    double *h_phi_init = nullptr, *h_phi = nullptr;
    // - the host pointer to the pre-permutated and post-permutated cell face vector list
    double *h_face_vector_init = nullptr, *h_face_vector = nullptr;
    double *h_face_init = nullptr, *h_face = nullptr;
    double *h_deltaCoeffs_init = nullptr, *h_deltaCoeffs = nullptr;
    // - the device pointer to rho_new, rho_old, velocity_old, pressure and volume list
    double *d_rho_new = nullptr, *d_rho_old = nullptr, *d_velocity_old = nullptr, 
    *d_pressure = nullptr, *d_volume = nullptr;
    // - the device pointer to Y_new and Y_old
    std::vector<double*> d_Y_new_vector;
    std::vector<double*> d_Y_old_vector;
    // - the device pointer to the pre-permutated and post-permutated interpolation weight list
    double *d_weight_init = nullptr, *d_weight = nullptr;
    double *d_weight_upwind = nullptr;
    // - the device pointer to the pre-permutated and post-permutated flux (phi) list
    double *d_phi_init = nullptr, *d_phi = nullptr;
    // - the device pointer to the pre-permutated and post-permutated cell face vector list
    double *d_face_vector_init = nullptr, *d_face_vector = nullptr;
    double *d_face_init = nullptr, *d_face = nullptr;
    double *d_deltaCoeffs_init = nullptr, *d_deltaCoeffs = nullptr;
    std::vector<double*> d_rhoD_vector;

    double *d_hDiffCorrFlux = nullptr;
    double *d_diffAlphaD = nullptr;

    double rdelta_t = 1/1e-6;

    /**
     * @brief boundary related variables
     */
    int *h_boundary_cell_offset = nullptr, *d_boundary_cell_offset=nullptr;
    int *h_boundary_cell_id = nullptr, *d_boundary_cell_id = nullptr;
    double *h_internal_coeffs = nullptr, *h_boundary_coeffs = nullptr,
    *h_boundary_pressure = nullptr, *h_boundary_face_vector = nullptr,
    *h_boundary_face = nullptr, *d_boundary_face = nullptr,
    *h_boundary_deltaCoeffs = nullptr, *d_boundary_deltaCoeffs = nullptr, 
    *d_internal_coeffs = nullptr, *d_boundary_coeffs = nullptr,
    *d_internal_coeffs_init = nullptr, *d_boundary_coeffs_init = nullptr,
    *d_laplac_internal_coeffs = nullptr, *d_laplac_boundary_coeffs = nullptr,
    *d_laplac_internal_coeffs_init = nullptr, *d_laplac_boundary_coeffs_init = nullptr,
    *d_boundary_pressure = nullptr, *d_boundary_face_vector = nullptr,
    *d_boundary_pressure_init = nullptr,
    *d_boundary_phi = nullptr, *d_boundary_phi_init = nullptr, 
    *d_boundary_velocity = nullptr, *d_boundary_velocity_init = nullptr,
    *d_boundary_nuEff = nullptr, *d_boundary_nuEff_init = nullptr,
    *d_boundary_rho = nullptr, *d_boundary_rho_init = nullptr;
    std::vector<double*> d_boundary_Y_vector;
    std::vector<double*> d_boundary_Y_init_vector;
    std::vector<double*> d_internal_coeffs_Y_vector;
    std::vector<double*> d_boundary_coeffs_Y_vector;
    std::vector<double*> d_laplac_internal_coeffs_Y_vector;
    std::vector<double*> d_laplac_boundary_coeffs_Y_vector;
    std::vector<double*> d_boundary_rhoD_vector;
    double *d_boundary_mut_sct = nullptr;

    double *d_boundary_hDiffCorrFlux = nullptr;

    std::vector<int> boundPermutationList;
    std::vector<double> ueqn_internalCoeffs, ueqn_boundaryCoeffs;
    std::vector<double> boundary_face_vector;
    std::vector<double> boundary_pressure;
    std::vector<double> boundary_face;
    std::vector<double> boundary_deltaCoeffs;

    // - the device pointer to the permutated index list
    std::vector<int> permedIndex;
    int *d_permedIndex=nullptr;
    int *d_bouPermedIndex = nullptr;


    // bytesize
    // - bytes of diagnal entries
    size_t cell_bytes;
    // - bytes of diagnal entries (vector)
    size_t cell_vec_bytes;
    // - bytes of diagnal index
    size_t cell_index_bytes;
     // - bytes of diagnal index
    size_t face_bytes;
    size_t face_vec_bytes;
    size_t face_index_bytes;

    size_t boundary_cell_bytes;
    size_t boundary_cell_vec_bytes;
    size_t boundary_cell_index_bytes;

    size_t boundary_face_bytes;
    size_t boundary_face_vec_bytes;
    size_t boundary_face_index_bytes;

    // A_csr has one more element in each row: itself
    size_t csr_row_index_bytes;
    size_t csr_col_index_bytes;
    size_t csr_value_bytes;
    size_t csr_value_vec_bytes;

    // extra matrix information
    double *d_turbSrc_A = nullptr, *d_turbSrc_b = nullptr, *d_turbSrc_A_init = nullptr;
    std::vector<double> h_turbSrc_init_mtx_vec, h_turbSrc_init_1mtx;
    std::vector<double> h_turbSrc_init_src_vec, h_turbSrc_src_vec;
    std::vector<int> tmpPermutatedList;
    int * d_tmpPermutatedList = nullptr;

    // double *h_A_csr = nullptr, *h_b = nullptr, *h_psi = nullptr;
    // double *d_A_csr = nullptr, *d_b = nullptr, *d_psi = nullptr;

    int num_iteration;

    double time_monitor_CPU;
    double time_monitor_GPU_kernel, time_monitor_GPU_memcpy, time_monitor_GPU_memcpy_test;

    double* d_grad = nullptr; 
    double* d_grad_boundary = nullptr, *d_grad_boundary_init = nullptr;
    double* d_nuEff = nullptr;

    // constructor
    dfMatrixDataBase();
    dfMatrixDataBase(int num_surfaces, int num_cells, int num_boundary_faces, int num_species, int & num_boundary_cells_output,
        const int *neighbour, const int *owner, const double* volume, const double* weight, const double* face_vector, const double* face, 
        const double* deltaCoeffs, std::vector<double> boundary_face_vector_init, std::vector<double> boundary_face_init, 
        std::vector<double> boundary_deltaCoeffs_init, std::vector<int> boundary_cell_id_init)
    : num_cells(num_cells), num_faces(num_surfaces*2), num_surfaces(num_surfaces), num_species(num_species), num_iteration(0),
      num_boundary_faces(num_boundary_faces), h_volume(volume)
    {
        // create cuda stream
        checkCudaErrors(cudaStreamCreate(&stream));

        // allocate field pointer in pin memory
        cudaMallocHost(&h_phi_init, num_faces * sizeof(double));
        cudaMallocHost(&h_rho_old, num_cells * sizeof(double));

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

        d_Y_new_vector.resize(num_species);
        d_Y_old_vector.resize(num_species);
        d_rhoD_vector.resize(num_species);
        d_boundary_Y_vector.resize(num_species);
        d_boundary_Y_init_vector.resize(num_species);
        d_internal_coeffs_Y_vector.resize(num_species);
        d_boundary_coeffs_Y_vector.resize(num_species);
        d_laplac_internal_coeffs_Y_vector.resize(num_species);
        d_laplac_boundary_coeffs_Y_vector.resize(num_species);
        d_boundary_rhoD_vector.resize(num_species);

        for (size_t i = 0; i < num_species; ++i){
            checkCudaErrors(cudaMalloc((void**)&d_Y_new_vector[i], cell_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_Y_old_vector[i], cell_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_rhoD_vector[i], cell_bytes));
        }
        checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_rho_new, cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_volume, cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_pressure, cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_velocity_old, cell_vec_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_weight, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_weight_upwind, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_face, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_deltaCoeffs, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_phi, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_phi_init, face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_face_vector, face_vec_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_nuEff, cell_bytes));
        total_bytes += (cell_bytes * (5 + 2*num_species) + face_bytes * 6 + cell_vec_bytes + face_vec_bytes);

        checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_id, boundary_face_index_bytes));
        total_bytes += (boundary_face_index_bytes + (num_boundary_cells+1) * sizeof(int));

        checkCudaErrors(cudaMalloc((void**)&d_boundary_pressure_init, boundary_face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_boundary_pressure, boundary_face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_boundary_phi_init, boundary_face_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_boundary_phi, boundary_face_bytes));
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
        checkCudaErrors(cudaMalloc((void**)&d_boundary_mut_sct, boundary_face_bytes));
        for (size_t i = 0; i < num_species; ++i){
            checkCudaErrors(cudaMalloc((void**)&d_boundary_Y_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_boundary_Y_init_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs_Y_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs_Y_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_laplac_internal_coeffs_Y_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_laplac_boundary_coeffs_Y_vector[i], boundary_face_bytes));
            checkCudaErrors(cudaMalloc((void**)&d_boundary_rhoD_vector[i], boundary_face_bytes));
        }
        
        total_bytes += (boundary_face_bytes*10 + boundary_face_vec_bytes * 11);

        // checkCudaErrors(cudaMalloc((void**)&d_A_csr, csr_value_vec_bytes));
        // checkCudaErrors(cudaMalloc((void**)&d_b, cell_vec_bytes));
        // checkCudaErrors(cudaMalloc((void**)&d_psi, cell_vec_bytes));
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

        checkCudaErrors(cudaMalloc((void**)&d_hDiffCorrFlux, 3 * cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_diffAlphaD, cell_bytes));
        checkCudaErrors(cudaMalloc((void**)&d_boundary_hDiffCorrFlux, 3 * boundary_face_bytes));

        fprintf(stderr, "Total bytes malloc on GPU: %.2fMB\n", total_bytes * 1.0 / 1024 / 1024);

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
    };

    ~dfMatrixDataBase(){
        std::cout << "Destructor called." << std::endl;
        // TODO: free pointers
        
    };
};


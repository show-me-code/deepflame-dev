#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"

class dfYEqn
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;

    std::vector<AmgXSolver *> YSolverSet;
    int num_iteration = 0;
    double time_monitor_CPU;
    double time_monitor_GPU_kernel, time_monitor_GPU_memcpy, time_monitor_GPU_memcpy_test;

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, num_boundary_cells, num_boundary_faces, num_species, inertIndex;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;

    // Matrix variables
    double *d_A_csr, *d_b, *d_psi = nullptr;
    double *h_A_csr, *h_b, *h_psi = nullptr;
    double *d_sumYDiffError = nullptr, *d_sumYDiffError_tmp = nullptr;
    double *d_sumYDiffError_boundary = nullptr;
    double *d_phiUc_boundary = nullptr;
    double *d_phiUc = nullptr;
    double *d_mut_Sct = nullptr;

public:
    dfYEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile, const int inertIndex);

    ~dfYEqn();

    void checkValue(bool print, char *filename);

    void correctVelocity(std::vector<double *> Y_new, std::vector<double *> boundary_Y_init, std::vector<const double *> rhoD_GPU,
                         const double *phiUcRef, double *boundary_sumYDiffError_ref, double *phiUcBouRef);

    void upwindWeight(double *refWeight);

    void fvm_ddt(std::vector<double *> Y_old);

    void fvm_div_phi();

    void fvm_div_phiUc();

    void fvm_laplacian(double *mut_Sct, double *boundary_mut_Sct, std::vector<double*>boundary_rhoD);

    void solve();

    void updatePsi(double *Psi, int speciesIndex);
};

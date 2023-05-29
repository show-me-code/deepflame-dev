#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"

class dfUEqn
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;
    AmgXSolver *UxSolver, *UySolver, *UzSolver = nullptr;
    int num_iteration;
    double time_monitor_CPU;
    double time_monitor_GPU_kernel, time_monitor_GPU_memcpy, time_monitor_GPU_memcpy_test;

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, cell_vec_bytes, csr_value_vec_bytes, num_boundary_cells;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;

    // Matrix variables
    double *d_A_csr, *d_b, *d_psi = nullptr;
    double *h_A_csr, *h_b, *h_psi = nullptr;

public:
    dfUEqn();
    dfUEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile);
    ~dfUEqn();

    void checkValue(bool print);

    void fvm_ddt(double *vector_old);

    void fvm_div(double *phi, double *ueqn_internalCoeffs_init, double *ueqn_boundaryCoeffs_init,
                 double *ueqn_laplac_internalCoeffs_init, double *ueqn_laplac_boundaryCoeffs_init,
                 double *boundary_pressure_init, double *boundary_velocity_init,
                 double *boundary_nuEff_init, double *boundary_rho_init);

    void fvc_grad(double *pressure);

    void fvc_grad_vector();

    void dev2T();

    void fvc_div_tensor(const double *nuEff);

    void fvm_laplacian();

    void add_fvMatrix(double *turbSrc_low, double *turbSrc_diag, double *turbSrc_upp, double *turbSrc_source);

    void solve();
    void updatePsi(double *Psi);
};
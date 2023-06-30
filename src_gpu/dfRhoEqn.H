#pragma once
#include "dfMatrixDataBase.H"

/*
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
        + fvc::div(phi)
    );

    rhoEqn.solve();
*/

class dfRhoEqn
{
private:
    dfMatrixDataBase& dataBase_;
    cudaStream_t stream;

    int num_iteration;
    double time_monitor_CPU;
    double time_monitor_GPU_kernel, time_monitor_GPU_memcpy, time_monitor_GPU_memcpy_test;

    // common variables
    int num_cells, cell_bytes, num_surfaces, num_boundary_cells;
    int *d_A_csr_row_index, *d_A_csr_diag_index;

    // Matrix variables
    double *d_b, *d_psi = nullptr;
    double *h_b, *h_psi = nullptr;

public:
    dfRhoEqn();
    dfRhoEqn(dfMatrixDataBase& dataBase);
    ~dfRhoEqn();

    void initializeTimeStep();

    void checkValue(bool print);

    void fvc_div(double *phi, double *boundary_phi_init);

    void fvm_ddt(double *rho_old);

    void sync();

    void updatePsi(double* Psi);
};

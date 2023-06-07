#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"

class dfEEqn
{
private:
    dfMatrixDataBase &dataBase_;
    cudaStream_t stream;
    AmgXSolver *ESolver = nullptr;
    int num_iteration;

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, cell_vec_bytes, csr_value_vec_bytes, num_boundary_cells;
    int num_boundary_faces, boundary_face_bytes;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;

    // Matrix variables
    double *d_A_csr, *d_b = nullptr;
    double *h_A_csr, *h_b = nullptr;
    double *d_he_old = nullptr;
    double *h_he_new = nullptr;

    // fields used by EEqn
    double *d_alphaEff = nullptr;
    double *d_K = nullptr;
    double *d_K_old = nullptr;
    double *d_dpdt = nullptr;
    double *d_diffAlphaD = nullptr;
    double *d_hDiffCorrFlux = nullptr;
    double *d_boundary_K_init = nullptr;
    double *d_boundary_K = nullptr;
    double *d_boundary_hDiffCorrFlux_init = nullptr;
    double *d_boundary_hDiffCorrFlux = nullptr;
    double *d_boundary_alphaEff_init = nullptr;
    double *d_boundary_alphaEff = nullptr;
    double *d_value_internal_coeffs_init = nullptr;
    double *d_value_boundary_coeffs_init = nullptr;
    double *d_gradient_internal_coeffs_init = nullptr;
    double *d_gradient_boundary_coeffs_init = nullptr;
    double *d_value_internal_coeffs = nullptr;
    double *d_value_boundary_coeffs = nullptr;
    double *d_gradient_internal_coeffs = nullptr;
    double *d_gradient_boundary_coeffs = nullptr;

public:
    dfEEqn();
    dfEEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile);
    ~dfEEqn();

    void prepare_data(const double *he_old, const double *K, const double *K_old, const double *alphaEff,
            const double *hDiffCorrFlux, const double *dpdt, const double *diffAlphaD,
            const double *boundary_K, const double *boundary_hDiffCorrFlux, const double *boundary_alphaEff);

    void initializeTimeStep();

    void fvm_ddt();
    void fvm_div();
    void fvm_laplacian();

    void fvc_ddt();
    void fvc_div_phi_scalar();
    void fvc_div_vector();
    void add_to_source();

    void solve();
    void checkValue(bool print);
    void updatePsi(double *Psi);

    void sync();
};

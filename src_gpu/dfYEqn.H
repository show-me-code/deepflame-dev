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

    // common variables
    int num_cells, cell_bytes, num_faces, num_surfaces, num_boundary_cells, num_boundary_faces, num_species, boundary_face_bytes, inertIndex;
    int *d_A_csr_row_index, *d_A_csr_diag_index, *d_A_csr_col_index;

    // Matrix variables
    double *d_A_csr, *d_b, *d_psi = nullptr;
    double *h_A_csr, *h_b, *h_psi = nullptr;

    double *d_alpha = nullptr;
    double *d_mut_Sct, *d_boundary_mut_sct = nullptr;
    double *d_hai, *d_boundary_hai = nullptr;
    double *d_rhoD, *d_boundary_rhoD = nullptr;
    double *d_sum_hai_rhoD_grady, *d_sum_boundary_hai_rhoD_grady = nullptr; 
    double *d_sum_rhoD_grady, *d_sum_boundary_rhoD_grady = nullptr;
    double *d_sum_hai_y, *d_sum_boundary_hai_y = nullptr;
    double *d_phiUc, *d_phiUc_boundary = nullptr;
    double *d_boundary_Y = nullptr;
    double *d_grady, *d_boundary_grady = nullptr;

    bool uploadData = true;

public:
    dfYEqn(dfMatrixDataBase &dataBase, const std::string &modeStr, const std::string &cfgFile, const int inertIndex);

    ~dfYEqn();

    void initializeTimeStep();

    void checkValue(bool print, char *filename);

    void upwindWeight();

    void fvm_laplacian_and_sumYDiffError_diffAlphaD_hDiffCorrFlux(
            std::vector<double *> Y_old, std::vector<double *> boundary_Y,
            std::vector<const double *> hai, std::vector<double *> boundary_hai,
            std::vector<const double *> rhoD, std::vector<double *> boundary_rhoD,
            const double *mut_Sct, const double *boundary_mut_Sct, const double *alpha);

    void fvm_ddt();

    void fvm_div_phi();

    void fvm_div_phiUc();

    void solve();

    void correctBoundaryConditions();

    void sync();

    void updatePsi(double *Psi, int speciesIndex);
};

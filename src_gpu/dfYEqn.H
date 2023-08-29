#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

class dfYEqn
{
private:
	dfMatrixDataBase &dataBase_;

    // cuda resource
    // one graph for one eqn before using self-developed solver
    cudaGraph_t graph;
    cudaGraphExec_t graph_instance;
    bool graph_created=false;

	// constant values -- basic
	std::string mode_string;
	std::string setting_path;
    int inertIndex;

	// constant values -- amgx solvers
    std::vector<AmgXSolver *> YSolverSet;
    int num_iteration = 0;

	// constant fields - internal
	// 无

	// constant fields - boundary
	std::vector<int> patch_type;

	// non-constant fields - internal
	// thermophysical fields - read
    double *d_rhoD = nullptr;
    double *d_hai = nullptr;
    double *d_mut_sct = nullptr;
    // thermophysical fields - write
    double *d_thermo_alpha = nullptr;
    double *d_hDiff_corr_flux = nullptr;
    double *d_diff_alphaD = nullptr;
    // intermediate fields
    double *d_grad_y = nullptr;
    double *d_sumY_diff_error = nullptr;
    double *d_phiUc = nullptr;
    double *d_phi_special_weight = nullptr;
    double *d_phiUc_special_weight = nullptr;
    double *d_DEff = nullptr;
    // computed on CPU, used on GPU, need memcpyh2d
	double *h_hai = nullptr;
	double *h_rhoD = nullptr;
	double *h_mut_sct = nullptr;

	// non-constant fields - boundary
	// thermophysical fields - read
    double *d_boundary_hai = nullptr;
    double *d_boundary_rhoD = nullptr;
    double *d_boundary_mut_sct = nullptr;
    // thermophysical fields - write
    double *d_boundary_thermo_alpha = nullptr;
    double *d_boundary_hDiff_corr_flux = nullptr;
    double *d_boundary_diff_alphaD = nullptr;
    // intermediate fields
    double *d_boundary_grad_y = nullptr;
    double *d_boundary_sumY_diff_error = nullptr;
    double *d_boundary_phiUc = nullptr;
    double *d_boundary_DEff = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_boundary_hai = nullptr;
	double *h_boundary_rhoD = nullptr;
	double *h_boundary_mut_sct = nullptr;
    // boundary coeff fields
	double *d_value_internal_coeffs = nullptr;
	double *d_value_boundary_coeffs= nullptr;
	double *d_gradient_internal_coeffs= nullptr;
	double *d_gradient_boundary_coeffs= nullptr;

	// non-constant fields - ldu
    double *d_ldu = nullptr;
	double *d_lower = nullptr;
	double *d_upper = nullptr;
	double *d_diag = nullptr;
	double *d_source = nullptr;
	double *d_internal_coeffs = nullptr;
	double *d_boundary_coeffs = nullptr;

	// non-constant fields - csr
	double *d_A = nullptr;
	double *d_b = nullptr;

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
	// 构造函数
    dfYEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// 析构函数
	~dfYEqn(){
        if (graph_created) {
            checkCudaErrors(cudaGraphExecDestroy(graph_instance));
            checkCudaErrors(cudaGraphDestroy(graph));
        }
    }

	// 成员函数

    // getter函数
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// 初始化构建
	void setConstantValues(const std::string &mode_string, const std::string &setting_path, const int inertIndex);
	void setConstantFields(const std::vector<int> patch_type);
	void createNonConstantFieldsInternal();
	void createNonConstantFieldsBoundary();
	void createNonConstantLduAndCsrFields();
	void initNonConstantFieldsInternal();
	void initNonConstantFieldsBoundary();

	// 方程运行
    void preProcess(const double *h_rhoD, const double *h_boundary_rhoD,
        const double *h_hai, const double *h_boundary_hai,
        const double *h_mut_sct, const double *h_boundary_mut_sct);
	void process();
	void postProcess(double *h_y);
    void solve(int solverIndex);

    // 方程特化版离散函数
    void yeqn_fvc_laplacian_scalar(cudaStream_t stream, int num_species, int num_cells, int num_surfaces, int num_boundary_surfaces,
            const int *lowerAddr, const int *upperAddr,
            const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
            const double *rhoD, const double *hai, const double *vf, double *output, // end for internal
            int num_patches, const int *patch_size, const int *patch_type,
            const int *boundary_cell_face,
            const double *boundary_mag_sf, const double *boundary_delta_coeffs,
            const double *boundary_rhoD, const double *boundary_hai, const double *boundary_vf,
            double *boundary_output, double sign);
    void yeqn_compute_sumYDiffError_and_hDiffCorrFlux(cudaStream_t stream, int num_species, int num_cells, int num_boundary_surfaces,
        const double *rhoD, const double *hai, const double *y, const double *grad_y,
        double *sum_rhoD_grady, double *hDiff_corr_flux,
        const double *boundary_rhoD, const double *boundary_hai, const double *boundary_y, const double *boundary_grad_y,
        double *boundary_sum_rhoD_grady, double *boundary_hDiff_corr_flux);
    void yeqn_compute_phiUc(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *sf, const double *sumY_diff_error, double *phiUc,
        const double *boundary_sf, const double *boundary_sumY_diff_error, double *boundary_phiUc);
    void yeqn_compute_y_inertIndex(cudaStream_t stream, int num_species, int inertIndex, int num_cells, double *y);

#if defined DEBUG_
    void comparediffAlphaD(const double *diffAlphaD, const double *boundary_diffAlphaD, bool printFlag);
    void comparegrady0(const double *grad_y0, const double *boundary_grad_y0, bool printFlag);
    void comparesumYDiffError(const double *sumYDiffError, const double *boundary_sumYDiffError, bool printFlag);
    void comparehDiffCorrFlux(const double *hDiffCorrFlux, const double *boundary_hDiffCorrFlux, bool printFlag);
    void comparephiUc(const double *phiUc, const double *boundary_phiUc, bool printFlag);
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source,
            const double *internal_coeffs, const double *boundary_coeffs, bool printFlag);
#endif
};

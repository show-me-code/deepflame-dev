#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

class dfUEqn
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

	// constant values -- amgx solvers
	AmgXSolver *UxSolver = nullptr;
	AmgXSolver *UySolver = nullptr;
	AmgXSolver *UzSolver = nullptr;
    int num_iteration = 0;

	// constant fields - internal
	// 无

	// constant fields - boundary
	std::vector<int> patch_type;

	// non-constant fields - internal
	// thermophysical fields
	double *d_nu_eff = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_nu_eff = nullptr;
	// intermediate fields
	double *d_grad_u = nullptr;
	double *d_rho_nueff = nullptr;
	double *d_permute = nullptr;
    double *d_fvc_output = nullptr;

	// non-constant fields - boundary
	// thermophysical fields
	double *d_boundary_nu_eff = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_boundary_nu_eff = nullptr;
	// intermediate fields
	double *d_boundary_grad_u = nullptr;
	double *d_boundary_rho_nueff = nullptr;
    // boundary coeff fields
	double *d_value_internal_coeffs = nullptr;
	double *d_value_boundary_coeffs= nullptr;
	double *d_gradient_internal_coeffs= nullptr;
	double *d_gradient_boundary_coeffs= nullptr;

	// non-constant fields - ldu
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
    dfUEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// 析构函数
	~dfUEqn(){
        if (graph_created) {
            checkCudaErrors(cudaGraphExecDestroy(graph_instance));
            checkCudaErrors(cudaGraphDestroy(graph));
        }
    }

	// 成员函数

    // getter函数
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// 初始化构建
	void setConstantValues(const std::string &mode_string, const std::string &setting_path); 
	void setConstantFields(const std::vector<int> patch_type);
	void createNonConstantFieldsInternal();
	void createNonConstantFieldsBoundary();
	void createNonConstantLduAndCsrFields();
	// dfUEqn has no internal non-constant fields to be init
	//void initNonConstantFieldsInternal(xxx);
	void initNonConstantFieldsBoundary();

	// 方程运行
    void preProcessForRhoEqn(const double *h_rho, const double *h_phi, const double *h_boundary_phi);
	void preProcess(const double *h_u, const double *h_boundary_u, const double *h_p, const double *h_boundary_p, const double *h_nu_eff, const double *h_boundary_nu_eff, const double *h_boundary_rho);
	void process();
	void postProcess(double *h_u);

    void solve();
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs, bool printFlag);
};

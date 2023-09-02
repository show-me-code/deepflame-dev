#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

class dfEEqn
{
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
    AmgXSolver *ESolver = nullptr;
    int num_iteration = 0;

    // constant fields - boundary
	std::vector<int> patch_type_he;
    std::vector<int> patch_type_k;

    // non-constant fields - internal
    // dpdt
    double *d_dpdt = nullptr;
    double *h_dpdt = nullptr;

    // non-constant fields - boundary
    // gradient
    double *d_boundary_heGradient = nullptr;
    double *h_boundary_heGradient = nullptr;
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
	double *d_b = nullptr; // TODO: needless

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
    // constructor
    dfEEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// destructor
	~dfEEqn(){
        if (graph_created) {
            checkCudaErrors(cudaGraphExecDestroy(graph_instance));
            checkCudaErrors(cudaGraphDestroy(graph));
        }
    }

    // member function

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// initialization
	void setConstantValues(const std::string &mode_string, const std::string &setting_path); 
	void setConstantFields(const std::vector<int> patch_type_he, const std::vector<int> patch_type_k);
    void createNonConstantFieldsInternal();
    void createNonConstantFieldsBoundary();
    void createNonConstantLduAndCsrFields();

    // run equation
    // volScalarField diffAlphaD, volVectorField hDiffCorrFlux, 
    void preProcessForYEqn();
    void preProcess(const double *h_he, const double *h_k, const double *h_k_old, const double *h_dpdt, const double *h_boundary_k, const double *h_boundary_heGradient);
	void process();
	void postProcess(double *h_he);

    void solve();
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs, bool printFlag);

    void sync();

};
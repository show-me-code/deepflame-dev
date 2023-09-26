#pragma once

#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

class dfRhoEqn
{
private:
	dfMatrixDataBase &dataBase_;

    // cuda resource
    // one graph for one eqn before using self-developed solver
    cudaGraph_t graph;
    cudaGraphExec_t graph_instance;
    bool graph_created=false;

    // constant fields - boundary
	std::vector<int> patch_type;

    // non-constant fields - ldu
    double *d_diag = nullptr;
    double *d_source = nullptr;

public:
    // constructor
    dfRhoEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// destructor
	~dfRhoEqn(){
        if (graph_created) {
            checkCudaErrors(cudaGraphExecDestroy(graph_instance));
            checkCudaErrors(cudaGraphDestroy(graph));
        }
    }

    // member function

    // initialization
    void setConstantFields(const std::vector<int> patch_type);
    void initNonConstantFields(const double *rho, const double *phi, 
            const double *boundary_rho, const double *boudnary_phi);
    void createNonConstantLduAndCsrFields();

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

    // run equations
    void preProcess();
    void process();
    void postProcess(double *h_rho);

    void solve();

    // other functions
    void compareResult(const double *diag, const double *source, bool printFlag);
    void compareRho(const double *rho, const double *boundary_rho, bool printFlag);
    void sync();
    void correctPsi(const double *rho, const double *boundary_rho);
};
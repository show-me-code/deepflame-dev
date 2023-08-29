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
    void createNonConstantLduAndCsrFields();

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

    // run equations
    void preProcess(const double *h_phi, const double *h_boundary_phi);
    void process();
    void postProcess(double *h_rho);

    void solve();

    // other functions
    void compareResult(const double *diag, const double *source, bool printFlag);
    void sync();
};
#pragma once

#include "dfMatrixDataBaseOrig.H"
#include "dfMatrixDataBase.H"

void fvc_grad_vector_orig(cudaStream_t stream, dfMatrixDataBaseOrig* dataBaseOrig, dfMatrixDataBase& dataBase, double *d_grad, 
        double *d_grad_boundary_init, double *d_grad_boundary);

void fvc_grad_scalar_orig(cudaStream_t stream, dfMatrixDataBaseOrig* dataBaseOrig, dfMatrixDataBase& dataBase, double *d_grad);
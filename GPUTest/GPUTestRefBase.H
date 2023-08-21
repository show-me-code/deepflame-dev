
// unittest of fvc::grad(U)
void test_fvc_grad_vector_orig(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volVectorField& U, initType type,
        dfMatrixDataBaseOrig* dfDataBaseOrig) 
{
    if (type == initType::randomInit) {
        U.oldTime();
        randomInitField<volVectorField>(U);
    }

    // run CPU
    // volTensorField fvc_ouput_tensor = fvc::grad(U);
    volTensorField fvc_ouput_tensor = gaussGradSchemeGrad(U);

    // prepare for run GPU
    // prepare U on GPU
    uploadRegisteredField<volVectorField>(dfDataBase, U, "u");
    
    double *d_fvc_ouput_tensor = nullptr, *d_fvc_ouput_boundary_tensor = nullptr, *d_fvc_ouput_boundary_tensor_init = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_tensor, dfDataBase.cell_value_tsr_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_boundary_tensor, dfDataBase.boundary_surface_value_tsr_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_boundary_tensor_init, dfDataBase.boundary_surface_value_tsr_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_tensor, 0, dfDataBase.cell_value_tsr_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_boundary_tensor, 0, dfDataBase.boundary_surface_value_tsr_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_boundary_tensor_init, 0, dfDataBase.boundary_surface_value_tsr_bytes));

    fvc_grad_vector_orig(dfDataBase.stream, dfDataBaseOrig, dfDataBase, d_fvc_ouput_tensor, d_fvc_ouput_boundary_tensor_init, d_fvc_ouput_boundary_tensor);

    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_tensor(dfDataBase.num_cells * 9);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_tensor.data(), d_fvc_ouput_tensor, dfDataBase.cell_value_tsr_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells * 9, &fvc_ouput_tensor[0][0], h_fvc_ouput_tensor.data(), 1e-14, printFlag);
}

void test_fvc_grad_scalar_orig(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volScalarField& p, initType type,
        dfMatrixDataBaseOrig* dfDataBaseOrig) 
{
    if (type == initType::randomInit) {
        p.oldTime();
        randomInitField<volScalarField>(p);
    }

    // run CPU
    // volVectorField fvc_ouput_vector = fvc::grad(p);
    volVectorField fvc_ouput_vector = gaussGradSchemeGrad(p);

    // prepare for run GPU
    // prepare p on GPU
    uploadRegisteredField<volScalarField>(dfDataBase, p, "p");

    double *d_fvc_ouput_vector = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_vector, dfDataBase.cell_value_vec_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_vector, 0, dfDataBase.cell_value_vec_bytes));

    fvc_grad_scalar_orig(dfDataBase.stream, dfDataBaseOrig, dfDataBase, d_fvc_ouput_vector);

    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_vector(dfDataBase.num_cells * 3);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_vector.data(), d_fvc_ouput_vector, dfDataBase.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells * 3, &fvc_ouput_vector[0][0], h_fvc_ouput_vector.data(), 1e-14, printFlag);
}

enum initType{
    original,
    randomInit
};

// unittest of fvm::div(phi, U)
void test_fvm_div_scalar(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, surfaceScalarField& phi, volVectorField& U, initType type) {
    int offset = 0;

    // deal with init type
    if (type == initType::randomInit) {
      // random init phi to (-0.5, 0.5)
      // internal
      double *phi_internal_ptr = &phi[0];
      std::vector<double> init_phi_internal;
      init_phi_internal.resize(dfDataBase.num_surfaces);
      for (int i = 0; i < dfDataBase.num_surfaces; i++) {
          init_phi_internal[i] = (rand() % 10000 - 5000) / 10000.0;
      }
      memcpy(phi_internal_ptr, init_phi_internal.data(), dfDataBase.surface_value_bytes);
      // boundary
      offset = 0;
      forAll(U.boundaryField(), patchi)
      {
          fvsPatchScalarField& patchPhi = phi.boundaryFieldRef()[patchi];
          int patchsize = patchPhi.size();
          double *phi_boundary_ptr = &patchPhi[0];
          std::vector<double> init_phi_boundary;
          init_phi_boundary.resize(patchsize);
          for (int i = 0; i < patchsize; i++) {
              init_phi_boundary[i] = (rand() % 10000 - 5000) / 10000.0;
          }
          memcpy(phi_boundary_ptr, init_phi_boundary.data(), patchsize * sizeof(double));
          offset += patchsize;
      }
      // TODO: random init weight to (0, 1)
      // failed, weight is const. 
    }

    // run CPU
    fvVectorMatrix df_U = fvm::div(phi, U);

    // run GPU
    // run GPU - preProcess
    // prepare phi
    memcpy(dfDataBase.h_phi, &phi[0], dfDataBase.surface_value_bytes);
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
        int patchsize = patchPhi.size();
        memcpy(dfDataBase.h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize;
    }
    checkCudaErrors(cudaMemcpyAsync(dfDataBase.d_phi, dfDataBase.h_phi, dfDataBase.surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(dfDataBase.d_boundary_phi, dfDataBase.h_boundary_phi, dfDataBase.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    // prepare patch_type
    std::vector<int> patch_type_U;
    patch_type_U.resize(dfDataBase.num_patches);
    forAll(U.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_U[patchi]), U.boundaryField()[patchi].type());
    }
    // prepare boundary coeffs
    // TODO: updating boundary coeffs should be complemented later
    double *d_value_internal_coeffs_U = nullptr;
    double *d_value_boundary_coeffs_U = nullptr;
    double *d_gradient_internal_coeffs_U = nullptr;
    double *d_gradient_boundary_coeffs_U = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs_U, dfDataBase.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs_U, dfDataBase.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs_U, dfDataBase.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs_U, dfDataBase.boundary_surface_value_vec_bytes));
    update_boundary_coeffs_vector(dfDataBase.stream, dfDataBase.num_patches,
            dfDataBase.patch_size.data(), patch_type_U.data(),
            d_value_internal_coeffs_U, d_value_boundary_coeffs_U,
            d_gradient_internal_coeffs_U, d_gradient_boundary_coeffs_U);
    // prepare ldu
    double *d_lower = nullptr;
    double *d_upper = nullptr;
    double *d_diag = nullptr;
    double *d_internal_coeffs = nullptr;
    double *d_boundary_coeffs = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_lower, dfDataBase.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_upper, dfDataBase.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
    // run GPU - Process
    fvm_div_scalar(dfDataBase.stream, dfDataBase.num_surfaces, dfDataBase.d_owner, dfDataBase.d_neighbor,
            dfDataBase.d_phi, dfDataBase.d_weight,
            d_lower, d_upper, d_diag, // end for internal
            dfDataBase.num_patches, dfDataBase.patch_size.data(), patch_type_U.data(),
            dfDataBase.d_boundary_phi, d_value_internal_coeffs_U, d_value_boundary_coeffs_U,
            d_internal_coeffs, d_boundary_coeffs);
    // run GPU - postProcess
    std::vector<double> h_lower;
    h_lower.resize(dfDataBase.num_surfaces);
    std::vector<double> h_upper;
    h_upper.resize(dfDataBase.num_surfaces);
    std::vector<double> h_diag;
    h_diag.resize(dfDataBase.num_cells);
    std::vector<double> h_internal_coeffs;
    h_internal_coeffs.resize(dfDataBase.num_boundary_surfaces * 3);
    std::vector<double> h_boundary_coeffs;
    h_boundary_coeffs.resize(dfDataBase.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dfDataBase.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dfDataBase.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dfDataBase.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_lower));
    checkCudaErrors(cudaFree(d_upper));
    checkCudaErrors(cudaFree(d_diag));
    checkCudaErrors(cudaFree(d_internal_coeffs));
    checkCudaErrors(cudaFree(d_boundary_coeffs));
    checkCudaErrors(cudaFree(d_value_internal_coeffs_U));
    checkCudaErrors(cudaFree(d_value_boundary_coeffs_U));
    checkCudaErrors(cudaFree(d_gradient_internal_coeffs_U));
    checkCudaErrors(cudaFree(d_gradient_boundary_coeffs_U));

    // compare CPU and GPU results
    checkVectorEqual(dfDataBase.num_surfaces, &df_U.lower()[0], h_lower.data(), 1e-14);
    checkVectorEqual(dfDataBase.num_surfaces, &df_U.upper()[0], h_upper.data(), 1e-14);
    checkVectorEqual(dfDataBase.num_cells, &df_U.diag()[0], h_diag.data(), 1e-14);
    std::vector<double> cpu_internal_coeffs(dfDataBase.num_boundary_surfaces * 3);
    std::vector<double> cpu_boundary_coeffs(dfDataBase.num_boundary_surfaces * 3);
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        int patchSize = U.boundaryField()[patchi].size();
        const double* internal_coeff_ptr = &df_U.internalCoeffs()[patchi][0][0];
        const double* boundary_coeff_ptr = &df_U.boundaryCoeffs()[patchi][0][0];
        memcpy(cpu_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchSize * 3 * sizeof(double));
        memcpy(cpu_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchSize * 3 * sizeof(double));
        offset += patchSize;
    }
    checkVectorEqual(dfDataBase.num_boundary_surfaces * 3, cpu_internal_coeffs.data(), h_internal_coeffs.data(), 1e-14);
    checkVectorEqual(dfDataBase.num_boundary_surfaces * 3, cpu_boundary_coeffs.data(), h_boundary_coeffs.data(), 1e-14);
}

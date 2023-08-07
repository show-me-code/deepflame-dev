
enum initType{
    original,
    randomInit
};

struct testGPUDataBase {
    // some fvm ops don't use d_source;
    // some fvm ops don't use d_internal_coeffs and d_boundary_coeffs;
    // all the fvc ops only use d_source
    double *d_lower = nullptr;
    double *d_upper = nullptr;
    double *d_diag = nullptr;
    double *d_source = nullptr;
    double *d_internal_coeffs = nullptr;
    double *d_boundary_coeffs = nullptr;

    double *d_value_internal_coeffs = nullptr;
    double *d_value_boundary_coeffs = nullptr;
    double *d_gradient_internal_coeffs = nullptr;
    double *d_gradient_boundary_coeffs = nullptr;

    std::vector<int> patch_type;

    // constructor
    testGPUDataBase() {}

    // deconstructor
    ~testGPUDataBase() {
      if (d_lower) checkCudaErrors(cudaFree(d_lower));
      if (d_upper) checkCudaErrors(cudaFree(d_upper));
      if (d_diag) checkCudaErrors(cudaFree(d_diag));
      if (d_source) checkCudaErrors(cudaFree(d_source));
      if (d_internal_coeffs) checkCudaErrors(cudaFree(d_internal_coeffs));
      if (d_boundary_coeffs) checkCudaErrors(cudaFree(d_boundary_coeffs));

      if (d_value_internal_coeffs) checkCudaErrors(cudaFree(d_value_internal_coeffs));
      if (d_value_boundary_coeffs) checkCudaErrors(cudaFree(d_value_boundary_coeffs));
      if (d_gradient_internal_coeffs) checkCudaErrors(cudaFree(d_gradient_internal_coeffs));
      if (d_gradient_boundary_coeffs) checkCudaErrors(cudaFree(d_gradient_boundary_coeffs));
    }
};

void randomInitSurfaceScalar(surfaceScalarField& field) {
      // random init field value to (-0.5, 0.5)
      // internal
      double *field_internal_ptr = &field[0];
      std::vector<double> init_field_internal;
      init_field_internal.resize(dfDataBase.num_surfaces);
      for (int i = 0; i < dfDataBase.num_surfaces; i++) {
          init_field_internal[i] = (rand() % 10000 - 5000) / 10000.0;
      }
      memcpy(field_internal_ptr, init_field_internal.data(), dfDataBase.surface_value_bytes);
      // boundary
      int offset = 0;
      forAll(field.boundaryField(), patchi)
      {
          auto& patchField = field.boundaryFieldRef()[patchi];
          int patchsize = patchField.size();
          double *field_boundary_ptr = &patchField[0];
          std::vector<double> init_field_boundary;
          init_field_boundary.resize(patchsize);
          for (int i = 0; i < patchsize; i++) {
              init_field_boundary[i] = (rand() % 10000 - 5000) / 10000.0;
          }
          memcpy(field_boundary_ptr, init_field_boundary.data(), patchsize * sizeof(double));
          offset += patchsize;
      }
}

void randomInitVolScalar(volScalarField& field) {
      // random init field value to (-0.5, 0.5)
      // internal
      double *field_internal_ptr = &field[0];
      std::vector<double> init_field_internal;
      init_field_internal.resize(dfDataBase.num_cells);
      for (int i = 0; i < dfDataBase.num_cells; i++) {
          init_field_internal[i] = (rand() % 10000 - 5000) / 10000.0;
      }
      memcpy(field_internal_ptr, init_field_internal.data(), dfDataBase.cell_value_bytes);
      // boundary
      int offset = 0;
      forAll(field.boundaryField(), patchi)
      {
          auto& patchField = field.boundaryFieldRef()[patchi];
          int patchsize = patchField.size();
          double *field_boundary_ptr = &patchField[0];
          std::vector<double> init_field_boundary;
          init_field_boundary.resize(patchsize);
          for (int i = 0; i < patchsize; i++) {
              init_field_boundary[i] = (rand() % 10000 - 5000) / 10000.0;
          }
          memcpy(field_boundary_ptr, init_field_boundary.data(), patchsize * sizeof(double));
          offset += patchsize;
      }
}

// rho_old need special treatment: it use h_xxx of rho
void uploadRegisteredRhoOld(dfMatrixDataBase& dfDataBase, const volScalarField& field) {
    double *h_internal_field = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
    double *h_boundary_field = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);
    double *d_internal_field = dfDataBase.getFieldPointer("rho_old", location::gpu, position::internal);
    double *d_boundary_field = dfDataBase.getFieldPointer("rho_old", location::gpu, position::boundary);
    // internal
    memcpy(h_internal_field, &field[0], dfDataBase.cell_value_bytes);
    // boundary
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field + offset, &patchField[0], patchsize * sizeof(double));
        offset += patchsize;
    }
    // transfer
    checkCudaErrors(cudaMemcpyAsync(d_internal_field, h_internal_field, dfDataBase.cell_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field, dfDataBase.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

void uploadRegisteredVolScalar(dfMatrixDataBase& dfDataBase, const volScalarField& field, const char* fieldAlias) {
    double *h_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::internal);
    double *h_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::boundary);
    double *d_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::internal);
    double *d_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::boundary);
    // internal
    memcpy(h_internal_field, &field[0], dfDataBase.cell_value_bytes);
    // boundary
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field + offset, &patchField[0], patchsize * sizeof(double));
        offset += patchsize;
    }
    // transfer
    checkCudaErrors(cudaMemcpyAsync(d_internal_field, h_internal_field, dfDataBase.cell_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field, dfDataBase.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

void uploadRegisteredVolVector(dfMatrixDataBase& dfDataBase, const volVectorField& field, const char* fieldAlias) {
    double *h_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::internal);
    double *h_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::boundary);
    double *d_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::internal);
    double *d_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::boundary);
    // internal
    memcpy(h_internal_field, &field[0], dfDataBase.cell_value_vec_bytes);
    // boundary
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field + offset * 3, &patchField[0], patchsize * 3 * sizeof(double));
        offset += patchsize;
    }
    // transfer
    checkCudaErrors(cudaMemcpyAsync(d_internal_field, h_internal_field, dfDataBase.cell_value_vec_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field, dfDataBase.boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

void uploadRegisteredSurfaceScalar(dfMatrixDataBase& dfDataBase, const surfaceScalarField& field, const char* fieldAlias) {
    double *h_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::internal);
    double *h_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::boundary);
    double *d_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::internal);
    double *d_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::boundary);
    // internal
    memcpy(h_internal_field, &field[0], dfDataBase.surface_value_bytes);
    // boundary
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field + offset, &patchField[0], patchsize * sizeof(double));
        offset += patchsize;
    }
    // transfer
    checkCudaErrors(cudaMemcpyAsync(d_internal_field, h_internal_field, dfDataBase.surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field, dfDataBase.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

void uploadVolScalar(dfMatrixDataBase& dfDataBase, const volScalarField& field, double *d_field, double *d_boundary_field) {
    std::vector<double> h_boundary_field;
    h_boundary_field.resize(dfDataBase.num_boundary_surfaces);
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field.data() + offset, &patchField[0], patchsize * sizeof(double));
        offset += patchsize;
    }
    checkCudaErrors(cudaMemcpyAsync(d_field, &field[0], dfDataBase.cell_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field.data(), dfDataBase.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

void buildTestGPUDataBaseScalar(const dfMatrixDataBase& dfDataBase, testGPUDataBase& testData, const volScalarField& field,
        bool lowerFlag, bool upperFlag, bool diagFlag, bool sourceFlag, bool internalCoeffsFlag, bool boundaryCoeffsFlag,
        bool valueInternalCoeffsFlag, bool valueBoundaryCoeffsFlag, bool gradientInternalCoeffsFlag, bool gradientBoundaryCoeffsFlag) {
    // ldu
    if (lowerFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_lower, dfDataBase.surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_lower, 0, dfDataBase.surface_value_bytes));
    }
    if (upperFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_upper, dfDataBase.surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_upper, 0, dfDataBase.surface_value_bytes));
    }
    if (diagFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_diag, dfDataBase.cell_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_diag, 0, dfDataBase.cell_value_bytes));
    }
    if (sourceFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_source, dfDataBase.cell_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_source, 0, dfDataBase.cell_value_bytes));
    }
    if (internalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_internal_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    if (boundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_boundary_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    // boundary coeffs
    if (valueInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_internal_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_value_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    if (valueBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_boundary_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_value_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    if (gradientInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_internal_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_gradient_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    if (gradientBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_boundary_coeffs, dfDataBase.boundary_surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_gradient_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes));
    }
    // patch type
    testData.patch_type.resize(dfDataBase.num_patches);
    forAll(field.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(testData.patch_type[patchi]), field.boundaryField()[patchi].type());
    }
}

void buildTestGPUDataBaseVector(const dfMatrixDataBase& dfDataBase, testGPUDataBase& testData, const volVectorField& field,
        bool lowerFlag, bool upperFlag, bool diagFlag, bool sourceFlag, bool internalCoeffsFlag, bool boundaryCoeffsFlag,
        bool valueInternalCoeffsFlag, bool valueBoundaryCoeffsFlag, bool gradientInternalCoeffsFlag, bool gradientBoundaryCoeffsFlag) {
    // ldu
    if (lowerFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_lower, dfDataBase.surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_lower, 0, dfDataBase.surface_value_bytes));
    }
    if (upperFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_upper, dfDataBase.surface_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_upper, 0, dfDataBase.surface_value_bytes));
    }
    if (diagFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_diag, dfDataBase.cell_value_bytes));
        checkCudaErrors(cudaMemset(testData.d_diag, 0, dfDataBase.cell_value_bytes));
    }
    if (sourceFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_source, dfDataBase.cell_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_source, 0, dfDataBase.cell_value_vec_bytes));
    }
    if (internalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_internal_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    if (boundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_boundary_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    // boundary coeffs
    if (valueInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_value_internal_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    if (valueBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_value_boundary_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    if (gradientInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_gradient_internal_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    if (gradientBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes));
        checkCudaErrors(cudaMemset(testData.d_gradient_boundary_coeffs, 0, dfDataBase.boundary_surface_value_vec_bytes));
    }
    // patch type
    testData.patch_type.resize(dfDataBase.num_patches);
    forAll(field.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(testData.patch_type[patchi]), field.boundaryField()[patchi].type());
    }
}

void updateBoundaryCoeffsVector(const dfMatrixDataBase& dfDataBase, testGPUDataBase& testData) {
    update_boundary_coeffs_vector(dfDataBase.stream, dfDataBase.num_patches,
            dfDataBase.patch_size.data(), testData.patch_type.data(),
            testData.d_value_internal_coeffs, testData.d_value_boundary_coeffs,
            testData.d_gradient_internal_coeffs, testData.d_gradient_boundary_coeffs);
}

void compareResultVector(const dfMatrixDataBase& dfDataBase, const testGPUDataBase& testData, fvVectorMatrix& dfMatrix, bool printFlag) {
    if (testData.d_lower) {
        std::vector<double> h_lower;
        h_lower.resize(dfDataBase.num_surfaces);
        checkCudaErrors(cudaMemcpy(h_lower.data(), testData.d_lower, dfDataBase.surface_value_bytes, cudaMemcpyDeviceToHost));
        checkVectorEqual(dfDataBase.num_surfaces, &dfMatrix.lower()[0], h_lower.data(), 1e-14, printFlag);
    }
    if (testData.d_upper) {
        std::vector<double> h_upper;
        h_upper.resize(dfDataBase.num_surfaces);
        checkCudaErrors(cudaMemcpy(h_upper.data(), testData.d_upper, dfDataBase.surface_value_bytes, cudaMemcpyDeviceToHost));
        checkVectorEqual(dfDataBase.num_surfaces, &dfMatrix.upper()[0], h_upper.data(), 1e-14, printFlag);
    }
    if (testData.d_diag) {
        std::vector<double> h_diag;
        h_diag.resize(dfDataBase.num_cells);
        checkCudaErrors(cudaMemcpy(h_diag.data(), testData.d_diag, dfDataBase.cell_value_bytes, cudaMemcpyDeviceToHost));
        checkVectorEqual(dfDataBase.num_cells, &dfMatrix.diag()[0], h_diag.data(), 1e-14, printFlag);
    }
    if (testData.d_source) {
        std::vector<double> h_source;
        h_source.resize(dfDataBase.num_cells * 3);
        checkCudaErrors(cudaMemcpy(h_source.data(), testData.d_source, dfDataBase.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
        checkVectorEqual(dfDataBase.num_cells * 3, &dfMatrix.source()[0][0], h_source.data(), 1e-14, printFlag);
    }
    if (testData.d_internal_coeffs) {
        std::vector<double> h_internal_coeffs;
        h_internal_coeffs.resize(dfDataBase.num_boundary_surfaces * 3);
        checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), testData.d_internal_coeffs, dfDataBase.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
        std::vector<double> cpu_internal_coeffs(dfDataBase.num_boundary_surfaces * 3);
        int offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            int patchsize = dfDataBase.patch_size[patchi];
            const double* internal_coeff_ptr = &dfMatrix.internalCoeffs()[patchi][0][0];
            memcpy(cpu_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchsize * 3 * sizeof(double));
            offset += patchsize;
        }
        checkVectorEqual(dfDataBase.num_boundary_surfaces * 3, cpu_internal_coeffs.data(), h_internal_coeffs.data(), 1e-14, printFlag);
    }
    if (testData.d_boundary_coeffs) {
        std::vector<double> h_boundary_coeffs;
        h_boundary_coeffs.resize(dfDataBase.num_boundary_surfaces * 3);
        checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), testData.d_boundary_coeffs, dfDataBase.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
        std::vector<double> cpu_boundary_coeffs(dfDataBase.num_boundary_surfaces * 3);
        int offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            int patchsize = dfDataBase.patch_size[patchi];
            const double* boundary_coeff_ptr = &dfMatrix.boundaryCoeffs()[patchi][0][0];
            memcpy(cpu_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchsize * 3 * sizeof(double));
            offset += patchsize;
        }
        checkVectorEqual(dfDataBase.num_boundary_surfaces * 3, cpu_boundary_coeffs.data(), h_boundary_coeffs.data(), 1e-14, printFlag);
    }
}

// unittest of fvm::ddt(rho, U)
void test_fvm_ddt_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volScalarField& rho, volVectorField& U, initType type) {

    if (type == initType::randomInit) {
      // random init rho and rho.old
      rho.oldTime();
      randomInitVolScalar(rho);
    }

    // run CPU
    fvVectorMatrix dfMatrix = fvm::ddt(rho, U);

    // prepare for run GPU
    // prepare rho, rho.old, U
    uploadRegisteredVolScalar(dfDataBase, rho, "rho");
    uploadRegisteredRhoOld(dfDataBase, rho.oldTime());
    uploadRegisteredVolVector(dfDataBase, U.oldTime(), "u");
    // prepare testData
    testGPUDataBase testData;
    // only use diag and source
    buildTestGPUDataBaseVector(dfDataBase, testData, U, false, false, true, true, false, false, false, false, false, false);
    // run GPU
    fvm_ddt_vector(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.rdelta_t,
            dfDataBase.d_rho, dfDataBase.d_rho_old, dfDataBase.d_u, dfDataBase.d_volume,
            testData.d_diag, testData.d_source);

    // compare result
    bool printFlag = false;
    compareResultVector(dfDataBase, testData, dfMatrix, printFlag);
}

// unittest of fvm::div(phi, U)
void test_fvm_div_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, surfaceScalarField& phi, volVectorField& U, initType type) {
    if (type == initType::randomInit) {
      randomInitSurfaceScalar(phi);
      // TODO: random init weight failed, weight is const.
    }

    // run CPU
    fvVectorMatrix dfMatrix = fvm::div(phi, U);

    // prepare for run GPU
    // prepare phi field
    uploadRegisteredSurfaceScalar(dfDataBase, phi, "phi");
    // prepare testData
    testGPUDataBase testData;
    // not use source
    // gradient_internal_coeffs, gradient_boundary_coeffs are not needed actually, but updateBoundaryCoeffsVector will access them
    buildTestGPUDataBaseVector(dfDataBase, testData, U, true, true, true, false, true, true, true, true, true, true);
    // prepare boundary coeffs
    // TODO: updating boundary coeffs should be complemented later
    updateBoundaryCoeffsVector(dfDataBase, testData);

    // run GPU
    fvm_div_vector(dfDataBase.stream, dfDataBase.num_surfaces, dfDataBase.d_owner, dfDataBase.d_neighbor,
            dfDataBase.d_phi, dfDataBase.d_weight,
            testData.d_lower, testData.d_upper, testData.d_diag, // end for internal
            dfDataBase.num_patches, dfDataBase.patch_size.data(), testData.patch_type.data(),
            dfDataBase.d_boundary_phi, testData.d_value_internal_coeffs, testData.d_value_boundary_coeffs,
            testData.d_internal_coeffs, testData.d_boundary_coeffs);

    // compare result
    bool printFlag = false;
    compareResultVector(dfDataBase, testData, dfMatrix, printFlag);
}

// unittest of fvm::laplacian(gamma, vf)
void test_fvm_laplacian_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh,
        volScalarField& gamma, volVectorField& U, initType type)
{
    if (type == initType::randomInit) {
      randomInitVolScalar(gamma);
    }

    // run CPU
    fvVectorMatrix dfMatrix = fvm::laplacian(gamma, U);

    // prepare for run GPU
    // prepare gamma on GPU
    double *d_gamma = nullptr;
    double *d_boundary_gamma = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_gamma, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_gamma, dfDataBase.boundary_surface_value_bytes));
    uploadVolScalar(dfDataBase, gamma, d_gamma, d_boundary_gamma);
    // prepare testData
    testGPUDataBase testData;
    // not use source
    // value_internal_coeffs, value_boundary_coeffs are not needed actually, but updateBoundaryCoeffsVector will access them
    buildTestGPUDataBaseVector(dfDataBase, testData, U, true, true, true, false, true, true, true, true, true, true);
    // prepare boundary coeffs
    // TODO: updating boundary coeffs should be complemented later
    updateBoundaryCoeffsVector(dfDataBase, testData);

    // run GPU
    fvm_laplacian_vector(dfDataBase.stream, dfDataBase.num_surfaces,
            dfDataBase.d_owner, dfDataBase.d_neighbor,
            dfDataBase.d_weight, dfDataBase.d_mag_sf, dfDataBase.d_delta_coeffs, d_gamma,
            testData.d_lower, testData.d_upper, testData.d_diag, // end for internal
            dfDataBase.num_patches, dfDataBase.patch_size.data(), testData.patch_type.data(),
            dfDataBase.d_boundary_mag_sf, d_boundary_gamma,
            testData.d_gradient_internal_coeffs, testData.d_gradient_boundary_coeffs,
            testData.d_internal_coeffs, testData.d_boundary_coeffs);

    // compare result
    bool printFlag = false;
    compareResultVector(dfDataBase, testData, dfMatrix, printFlag);

    // free resources
    checkCudaErrors(cudaFree(d_gamma));
    checkCudaErrors(cudaFree(d_boundary_gamma));
}

// unittest of fvc::ddt(rho, K)
void test_fvc_ddt_scalar(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volScalarField& rho, volScalarField& K, initType type) {

    if (type == initType::randomInit) {
      // random init rho and rho.old
      rho.oldTime();
      randomInitVolScalar(rho);
      K.oldTime();
      randomInitVolScalar(K);
    }

    // run CPU
    volScalarField fvc_ouput_scalar = fvc::ddt(rho, K);

    // prepare for run GPU
    // prepare rho, rho.old on GPU
    uploadRegisteredVolScalar(dfDataBase, rho, "rho");
    uploadRegisteredRhoOld(dfDataBase, rho.oldTime());
    // prepare K, K_old on GPU
    double *d_K = nullptr;
    double *d_K_old = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_K, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_K_old, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_K, &K[0], dfDataBase.cell_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_K_old, &K.oldTime()[0], dfDataBase.cell_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    // there is no need for fvc ops to build testGPUDataBase, just build d_fvc_ouput_scalar directly.
    double *d_fvc_ouput_scalar = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_scalar, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_scalar, 0, dfDataBase.cell_value_bytes));
    // run GPU
    // fvc_ddt doesn't consider to add fvc_output to source yet, which needs (fvc_output * volume * sign).
    fvc_ddt_scalar(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.rdelta_t,
            dfDataBase.d_rho, dfDataBase.d_rho_old, d_K, d_K_old,
            d_fvc_ouput_scalar);

    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_scalar;
    h_fvc_ouput_scalar.resize(dfDataBase.num_cells);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_scalar.data(), d_fvc_ouput_scalar, dfDataBase.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells, &fvc_ouput_scalar[0], h_fvc_ouput_scalar.data(), 1e-12, printFlag);

    // free resources
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_K_old));
    checkCudaErrors(cudaFree(d_fvc_ouput_scalar));
}



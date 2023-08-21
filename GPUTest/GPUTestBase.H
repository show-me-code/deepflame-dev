
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

template <typename T>
void getTypeInfo(size_t *stride, size_t *internal_size, size_t *boundary_size) {
    size_t s = 1;
    bool isVol = false;
    if (typeid(T) == typeid(surfaceScalarField)) {
        s = 1;
        isVol = false;
    } else if (typeid(T) == typeid(surfaceVectorField)) {
        s = 3;
        isVol = false;
    } else if (typeid(T) == typeid(surfaceTensorField)) {
        s = 9;
        isVol = false;
    } else if (typeid(T) == typeid(volScalarField)) {
        s = 1;
        isVol = true;
    } else if (typeid(T) == typeid(volVectorField)) {
        s = 3;
        isVol = true;
    } else if (typeid(T) == typeid(volTensorField)) {
        s = 9;
        isVol = true;
    } else {
        fprintf(stderr, "ERROR! Unsupported field type()!\n");
        exit(EXIT_FAILURE);
    }
    *stride = s;
    *internal_size = (isVol ? dfDataBase.num_cells : dfDataBase.num_surfaces) * s;
    *boundary_size = dfDataBase.num_boundary_surfaces * s;
}


template <typename T>
void getFieldPtr(std::queue<double*>& fieldPtrQue, T& field){
    fieldPtrQue.push(&field[0]);
    forAll(field.boundaryField(), patchi){
        auto& patchField = field.boundaryFieldRef()[patchi];
        fieldPtrQue.push(&patchField[0]);
    }
};

// template <typename T>
// void getFieldPtr(std::vector<double*>& fieldPtrQue, T& field){
//     fieldPtrQue.push_back(&field[0]);
//     forAll(field.boundaryField(), patchi){
//         auto& patchField = field.boundaryFieldRef()[patchi];
//         fieldPtrQue.push_back(&patchField[0]);
//         Info << "patchi " << patchi << endl;
//     }
// };


template <typename T>
void randomInitField(T& field) {
    size_t stride = 0;
    size_t internal_size = 0;
    size_t boundary_size = 0;
    getTypeInfo<T>(&stride, &internal_size, &boundary_size);
    size_t internal_value_bytes = internal_size * sizeof(double) * stride;
    std::queue<double*> fieldPtrQue;
    // std::vector<double*> fieldPtrQue;
    getFieldPtr(fieldPtrQue, field);

    // random init field value to (-0.5, 0.5)
    // internal
    double *&field_internal_ptr = fieldPtrQue.front(); fieldPtrQue.pop();
    // double *field_internal_ptr = fieldPtrQue[0];
    std::vector<double> init_field_internal;
    init_field_internal.resize(internal_size * stride);
    for (size_t i = 0; i < internal_size * stride; i++) {
        init_field_internal[i] = (rand() % 10000 - 5000) / 10000.0;
    }
    memcpy(field_internal_ptr, init_field_internal.data(), internal_value_bytes);
    // boundary
    int ptrIndex = 1; 
    forAll(field.boundaryField(), patchi)
    {
        auto& patchField = field.boundaryFieldRef()[patchi];
        size_t patchsize = patchField.size();
        double *&field_boundary_ptr = fieldPtrQue.front(); fieldPtrQue.pop();
        // double *field_boundary_ptr = fieldPtrQue[ptrIndex];
        // ptrIndex ++;
        std::vector<double> init_field_boundary;
        init_field_boundary.resize(patchsize * stride);
        for (size_t i = 0; i < patchsize * stride; i++) {
            init_field_boundary[i] = (rand() % 10000 - 5000) / 10000.0;
        }
        memcpy(field_boundary_ptr, init_field_boundary.data(), patchsize * stride * sizeof(double));
    }
}

template <typename T>
void uploadRegisteredField(dfMatrixDataBase& dfDataBase, const T& field, const char* fieldAlias) {
    size_t stride = 0;
    size_t internal_size = 0;
    size_t boundary_size = 0;
    getTypeInfo<T>(&stride, &internal_size, &boundary_size);
    size_t internal_value_bytes = internal_size * sizeof(double);
    size_t boundary_value_bytes = boundary_size * sizeof(double);

    double *h_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::internal);
    double *h_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::cpu, position::boundary);
    double *d_internal_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::internal);
    double *d_boundary_field = dfDataBase.getFieldPointer(fieldAlias, location::gpu, position::boundary);

    // internal
    memcpy(h_internal_field, &field[0], internal_value_bytes);
    // boundary
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field + offset * stride, &patchField[0], patchsize * stride * sizeof(double));
        offset += patchsize;
    }
    // transfer
    checkCudaErrors(cudaMemcpyAsync(d_internal_field, h_internal_field, internal_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field, boundary_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

template <typename T>
void uploadField(dfMatrixDataBase& dfDataBase, const T& field, double *d_field, double *d_boundary_field) {
    size_t stride = 0;
    size_t internal_size = 0;
    size_t boundary_size = 0;
    getTypeInfo<T>(&stride, &internal_size, &boundary_size);
    size_t internal_value_bytes = internal_size * sizeof(double);
    size_t boundary_value_bytes = boundary_size * sizeof(double);

    std::vector<double> h_boundary_field;
    h_boundary_field.resize(boundary_size);
    int offset = 0;
    forAll(field.boundaryField(), patchi)
    {
        const auto& patchField = field.boundaryField()[patchi];
        int patchsize = patchField.size();
        memcpy(h_boundary_field.data() + offset * stride, &patchField[0], patchsize * stride * sizeof(double));
        offset += patchsize;
    }
    checkCudaErrors(cudaMemcpyAsync(d_field, &field[0], internal_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_field, h_boundary_field.data(), boundary_value_bytes, cudaMemcpyHostToDevice, dfDataBase.stream));
}

template <typename T>
void buildTestGPUDataBase(const dfMatrixDataBase& dfDataBase, testGPUDataBase& testData, const T& field,
        bool lowerFlag, bool upperFlag, bool diagFlag, bool sourceFlag, bool internalCoeffsFlag, bool boundaryCoeffsFlag,
        bool valueInternalCoeffsFlag, bool valueBoundaryCoeffsFlag, bool gradientInternalCoeffsFlag, bool gradientBoundaryCoeffsFlag) {
    if ((typeid(T) != typeid(volScalarField)) && (typeid(T) != typeid(volVectorField))) {
        fprintf(stderr, "ERROR! Unsupported field type()!\n");
        exit(EXIT_FAILURE);
    }
    bool isVec = (typeid(T) == typeid(volVectorField));
    size_t stride = isVec ? 3 : 1;

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
        checkCudaErrors(cudaMalloc((void**)&testData.d_source, dfDataBase.cell_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_source, 0, dfDataBase.cell_value_bytes * stride));
    }
    if (internalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_internal_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    if (boundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_boundary_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    // boundary coeffs
    if (valueInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_internal_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_value_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    if (valueBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_value_boundary_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_value_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    if (gradientInternalCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_internal_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_gradient_internal_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    if (gradientBoundaryCoeffsFlag) {
        checkCudaErrors(cudaMalloc((void**)&testData.d_gradient_boundary_coeffs, dfDataBase.boundary_surface_value_bytes * stride));
        checkCudaErrors(cudaMemset(testData.d_gradient_boundary_coeffs, 0, dfDataBase.boundary_surface_value_bytes * stride));
    }
    // patch type
    testData.patch_type.resize(dfDataBase.num_patches);
    forAll(field.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(testData.patch_type[patchi]), field.boundaryField()[patchi].type());
    }
}

// TODO: It seems that compareResult of scalar and vector can't be merged
void compareResultVector(const dfMatrixDataBase& dfDataBase, const testGPUDataBase& testData, fvVectorMatrix& dfMatrix, bool printFlag) {
    //if ((typeid(T) != typeid(fvScalarMatrix)) && (typeid(T) != typeid(fvVectorMatrix))) {
    //    fprintf(stderr, "ERROR! Unsupported field type()!\n");
    //    exit(EXIT_FAILURE);
    //}
    //bool isVec = (typeid(T) == typeid(fvVectorMatrix));
    //size_t stride = isVec ? 3 : 1;

    size_t stride = 3;
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
        h_source.resize(dfDataBase.num_cells * stride);
        checkCudaErrors(cudaMemcpy(h_source.data(), testData.d_source, dfDataBase.cell_value_bytes * stride, cudaMemcpyDeviceToHost));
        //void *source_ptr = isVec ? (&dfMatrix.source()[0][0]) : (&dfMatrix.source()[0]);
        double *source_ptr = &dfMatrix.source()[0][0];
        checkVectorEqual(dfDataBase.num_cells * stride, source_ptr, h_source.data(), 1e-14, printFlag);
    }
    if (testData.d_internal_coeffs) {
        std::vector<double> h_internal_coeffs;
        h_internal_coeffs.resize(dfDataBase.num_boundary_surfaces * stride);
        checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), testData.d_internal_coeffs, dfDataBase.boundary_surface_value_bytes * stride, cudaMemcpyDeviceToHost));
        std::vector<double> cpu_internal_coeffs(dfDataBase.num_boundary_surfaces * stride);
        int offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            int patchsize = dfDataBase.patch_size[patchi];
            //const void* internal_coeff_ptr = isVec ? (&dfMatrix.internalCoeffs()[patchi][0][0]) : (&dfMatrix.internalCoeffs()[patchi][0]);
            const void* internal_coeff_ptr = &dfMatrix.internalCoeffs()[patchi][0][0];
            memcpy(cpu_internal_coeffs.data() + offset * stride, internal_coeff_ptr, patchsize * stride * sizeof(double));
            offset += patchsize;
        }
        checkVectorEqual(dfDataBase.num_boundary_surfaces * stride, cpu_internal_coeffs.data(), h_internal_coeffs.data(), 1e-14, printFlag);
    }
    if (testData.d_boundary_coeffs) {
        std::vector<double> h_boundary_coeffs;
        h_boundary_coeffs.resize(dfDataBase.num_boundary_surfaces * stride);
        checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), testData.d_boundary_coeffs, dfDataBase.boundary_surface_value_bytes * stride, cudaMemcpyDeviceToHost));
        std::vector<double> cpu_boundary_coeffs(dfDataBase.num_boundary_surfaces * stride);
        int offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            int patchsize = dfDataBase.patch_size[patchi];
            //const void* boundary_coeff_ptr = isVec ? (&dfMatrix.boundaryCoeffs()[patchi][0][0]) : (&dfMatrix.boundaryCoeffs()[patchi][0]);
            const void* boundary_coeff_ptr = &dfMatrix.boundaryCoeffs()[patchi][0][0];
            memcpy(cpu_boundary_coeffs.data() + offset * stride, boundary_coeff_ptr, patchsize * stride * sizeof(double));
            offset += patchsize;
        }
        checkVectorEqual(dfDataBase.num_boundary_surfaces * stride, cpu_boundary_coeffs.data(), h_boundary_coeffs.data(), 1e-14, printFlag);
    }
}

// unittest of fvm::ddt(rho, U)
void test_fvm_ddt_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volScalarField& rho, volVectorField& U, initType type) {
    if (type == initType::randomInit) {
        rho.oldTime();
        randomInitField<volScalarField>(rho);
    }

    // run CPU
    // fvVectorMatrix dfMatrix = fvm::ddt(rho, U);
    fvVectorMatrix dfMatrix = EulerDdtSchemeFvmDdt(rho, U);

    // prepare for run GPU
    // prepare rho, rho.old, U
    uploadRegisteredField<volScalarField>(dfDataBase, rho, "rho");
    uploadRegisteredField<volScalarField>(dfDataBase, rho.oldTime(), "rho_old");
    uploadRegisteredField<volVectorField>(dfDataBase, U.oldTime(), "u");
    // prepare testData
    testGPUDataBase testData;
    // only use diag and source
    buildTestGPUDataBase<volVectorField>(dfDataBase, testData, U, false, false, true, true, false, false, false, false, false, false);
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
        phi.oldTime();
        randomInitField<surfaceScalarField>(phi);
    }

    // run CPU
    // fvVectorMatrix dfMatrix = fvm::div(phi, U);
    fvVectorMatrix dfMatrix = gaussConvectionSchemeFvmDiv(phi, U);

    // prepare for run GPU
    // prepare phi field
    uploadRegisteredField<surfaceScalarField>(dfDataBase, phi, "phi");
    // prepare testData
    testGPUDataBase testData;
    // not use source
    // gradient_internal_coeffs, gradient_boundary_coeffs are not needed actually, but update_boundary_coeffs_vector will access them
    buildTestGPUDataBase<volVectorField>(dfDataBase, testData, U, true, true, true, false, true, true, true, true, true, true);
    // prepare boundary coeffs
    // TODO: updating boundary coeffs should be complemented later
    update_boundary_coeffs_vector(dfDataBase.stream, dfDataBase.num_patches,
            dfDataBase.patch_size.data(), testData.patch_type.data(),
            testData.d_value_internal_coeffs, testData.d_value_boundary_coeffs,
            testData.d_gradient_internal_coeffs, testData.d_gradient_boundary_coeffs);

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
        gamma.oldTime();
        randomInitField<volScalarField>(gamma);
    }

    // run CPU
    // fvVectorMatrix dfMatrix = fvm::laplacian(gamma, U);
    fvVectorMatrix dfMatrix = gaussLaplacianSchemeFvmLaplacian(gamma, U);

    // prepare for run GPU
    // prepare gamma on GPU
    double *d_gamma = nullptr;
    double *d_boundary_gamma = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_gamma, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_gamma, dfDataBase.boundary_surface_value_bytes));
    uploadField<volScalarField>(dfDataBase, gamma, d_gamma, d_boundary_gamma);
    // prepare testData
    testGPUDataBase testData;
    // not use source
    // value_internal_coeffs, value_boundary_coeffs are not needed actually, but update_boundary_coeffs_vector will access them
    buildTestGPUDataBase<volVectorField>(dfDataBase, testData, U, true, true, true, false, true, true, true, true, true, true);
    // prepare boundary coeffs
    // TODO: updating boundary coeffs should be complemented later
    update_boundary_coeffs_vector(dfDataBase.stream, dfDataBase.num_patches,
            dfDataBase.patch_size.data(), testData.patch_type.data(),
            testData.d_value_internal_coeffs, testData.d_value_boundary_coeffs,
            testData.d_gradient_internal_coeffs, testData.d_gradient_boundary_coeffs);

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
        rho.oldTime();
        randomInitField<volScalarField>(rho);
        K.oldTime();
        randomInitField<volScalarField>(K);
    }

    // run CPU
    // volScalarField fvc_ouput_scalar = fvc::ddt(rho, K);
    volScalarField fvc_ouput_scalar = EulerDdtSchemeFvcDdt(rho, K);

    // prepare for run GPU
    // prepare rho, rho.old on GPU
    uploadRegisteredField<volScalarField>(dfDataBase, rho, "rho");
    uploadRegisteredField<volScalarField>(dfDataBase, rho.oldTime(), "rho_old");
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
    checkVectorEqual(dfDataBase.num_cells, &fvc_ouput_scalar[0], h_fvc_ouput_scalar.data(), 1e-14, printFlag);

    // free resources
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_K_old));
    checkCudaErrors(cudaFree(d_fvc_ouput_scalar));
}

// unittest of fvc::grad(U)
void test_fvc_grad_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volVectorField& U, initType type) {
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
    
    double *d_fvc_ouput_tensor = nullptr, *d_fvc_ouput_boundary_tensor = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_tensor, dfDataBase.cell_value_tsr_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_boundary_tensor, dfDataBase.boundary_surface_value_tsr_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_tensor, 0, dfDataBase.cell_value_tsr_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_boundary_tensor, 0, dfDataBase.boundary_surface_value_tsr_bytes));
    
    // only need patch_type
    testGPUDataBase testData;
    buildTestGPUDataBase<volVectorField>(dfDataBase, testData, U, false, false, false, false, false, false, false, false, false, false);

    fvc_grad_vector(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.num_surfaces,
            dfDataBase.d_owner, dfDataBase.d_neighbor, 
            dfDataBase.d_weight, dfDataBase.d_sf, dfDataBase.d_u, d_fvc_ouput_tensor,
            dfDataBase.num_patches, dfDataBase.patch_size.data(), testData.patch_type.data(), 
            dfDataBase.d_boundary_face_cell, dfDataBase.d_boundary_u, dfDataBase.d_boundary_sf,
            dfDataBase.d_volume, dfDataBase.d_boundary_mag_sf, d_fvc_ouput_boundary_tensor, dfDataBase.d_boundary_delta_coeffs);

    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_tensor(dfDataBase.num_cells * 9);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_tensor.data(), d_fvc_ouput_tensor, dfDataBase.cell_value_tsr_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells * 9, &fvc_ouput_tensor[0][0], h_fvc_ouput_tensor.data(), 1e-14, printFlag);
}

// unittest of fvc::div(phi)
void test_fvc_div_scalar(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, surfaceScalarField& phi, initType type) {
    if (type == initType::randomInit) {
        phi.oldTime();
        randomInitField<surfaceScalarField>(phi);
    }

    // run CPU
    volScalarField fvc_ouput_scalar = fvc::div(phi);
    // volScalarField fvc_ouput_scalar = gaussConvectionSchemeFvcDiv(phi);

    // prepare for run GPU
    // prepare phi on GPU
    uploadRegisteredField<surfaceScalarField>(dfDataBase, phi, "phi");

    double *d_fvc_ouput_scalar = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_scalar, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_scalar, 0, dfDataBase.cell_value_bytes));

    fvc_div_surface_scalar(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.num_surfaces, dfDataBase.num_boundary_surfaces,
            dfDataBase.d_owner, dfDataBase.d_neighbor,
            dfDataBase.d_phi, dfDataBase.d_boundary_face_cell, dfDataBase.d_boundary_phi, dfDataBase.d_volume, d_fvc_ouput_scalar);
    
    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_scalar(dfDataBase.num_cells);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_scalar.data(), d_fvc_ouput_scalar, dfDataBase.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells, &fvc_ouput_scalar[0], h_fvc_ouput_scalar.data(), 1e-14, printFlag);
}

// unittest of fvc::div(U)
void test_fvc_div_vector(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volVectorField& U, initType type) {
    if (type == initType::randomInit) {
        U.oldTime();
        randomInitField<volVectorField>(U);
    }

    // run CPU
    // volScalarField fvc_ouput_scalar = fvc::div(U);
    volScalarField fvc_ouput_scalar = gaussDivFvcdiv(U);

    // prepare for run GPU
    // prepare phi on GPU
    uploadRegisteredField<volVectorField>(dfDataBase, U, "u");

    double *d_fvc_ouput_scalar = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_fvc_ouput_scalar, dfDataBase.cell_value_bytes));
    checkCudaErrors(cudaMemset(d_fvc_ouput_scalar, 0, dfDataBase.cell_value_bytes));

    // only need patch_type
    testGPUDataBase testData;
    buildTestGPUDataBase<volVectorField>(dfDataBase, testData, U, false, false, false, false, false, false, false, false, false, false);

    fvc_div_cell_vector(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.num_surfaces, 
            dfDataBase.d_owner, dfDataBase.d_neighbor,
            dfDataBase.d_weight, dfDataBase.d_sf, dfDataBase.d_u, d_fvc_ouput_scalar,
            dfDataBase.num_patches, dfDataBase.patch_size.data(), testData.patch_type.data(), 
            dfDataBase.d_boundary_face_cell, dfDataBase.d_boundary_u, dfDataBase.d_boundary_sf,
            dfDataBase.d_volume);
    
    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_scalar(dfDataBase.num_cells);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_scalar.data(), d_fvc_ouput_scalar, dfDataBase.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells, &fvc_ouput_scalar[0], h_fvc_ouput_scalar.data(), 1e-14, printFlag);
}

// unittest of fvc::grad(p)
void test_fvc_grad_scalar(dfMatrixDataBase& dfDataBase, Foam::fvMesh& mesh, volScalarField& p, initType type) {
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

    // only need patch_type
    testGPUDataBase testData;
    buildTestGPUDataBase<volScalarField>(dfDataBase, testData, p, false, false, false, false, false, false, false, false, false, false);

    fvc_grad_cell_scalar(dfDataBase.stream, dfDataBase.num_cells, dfDataBase.num_surfaces,
            dfDataBase.d_owner, dfDataBase.d_neighbor, 
            dfDataBase.d_weight, dfDataBase.d_sf, dfDataBase.d_p, d_fvc_ouput_vector,
            dfDataBase.num_patches, dfDataBase.patch_size.data(), testData.patch_type.data(), 
            dfDataBase.d_boundary_face_cell, dfDataBase.d_boundary_p, dfDataBase.d_boundary_sf, dfDataBase.d_volume);

    // compare result
    bool printFlag = false;
    std::vector<double> h_fvc_ouput_vector(dfDataBase.num_cells * 3);
    checkCudaErrors(cudaMemcpy(h_fvc_ouput_vector.data(), d_fvc_ouput_vector, dfDataBase.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dfDataBase.num_cells * 3, &fvc_ouput_vector[0][0], h_fvc_ouput_vector.data(), 1e-14, printFlag);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
template <>
void getFieldPtr<volVectorField>(std::queue<double*>& fieldPtrQue, volVectorField& field) {
    fieldPtrQue.push(&field[0][0]);
    forAll(field.boundaryField(), patchi){
        auto& patchField = field.boundaryFieldRef()[patchi];
        fieldPtrQue.push(&patchField[0][0]);
    }
};

template <>
void getFieldPtr<volTensorField>(std::queue<double*>& fieldPtrQue, volTensorField& field) {
    fieldPtrQue.push(&field[0][0]);
    forAll(field.boundaryField(), patchi){
        auto& patchField = field.boundaryFieldRef()[patchi];
        fieldPtrQue.push(&patchField[0][0]);
    }
};
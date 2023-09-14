#include <queue>

ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
int myRank, nRanks, localRank = 0;

dfMatrixDataBase dfDataBase;
dfRhoEqn rhoEqn_GPU(dfDataBase);
dfUEqn UEqn_GPU(dfDataBase);
dfYEqn YEqn_GPU(dfDataBase);
dfEEqn EEqn_GPU(dfDataBase);

#if defined(DEBUG_)
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
#endif

void initNccl() {
    ncclInit(PstreamGlobals::MPI_COMM_FOAM, nccl_comm, nccl_id, &myRank, &nRanks, &localRank);
}

void createGPUBase(fvMesh& mesh, PtrList<volScalarField>& Y) {
    // prepare mpi and nccl info
    dfDataBase.setCommInfo(PstreamGlobals::MPI_COMM_FOAM, nccl_comm, nccl_id, myRank, nRanks, localRank);
    // TODO: test for temp
    ncclTest(nccl_comm);
    ncclDestroy(nccl_comm);

    // prepare constant values: num_cells, num_surfaces, num_boundary_surfaces, num_patches, patch_size, num_species, rdelta_t
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_surfaces = neighbour.size();
    int num_boundary_surfaces = 0;
    int num_patches = 0;
    std::vector<int> patch_size;
    forAll(mesh.boundary(), patchi) {
        labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
        int patchsize = sub_boundary.size();
        patch_size.push_back(patchsize);
        num_boundary_surfaces += patchsize;
        num_patches++;
    }
    // TODO: get deltaT fomr time API
    double rDeltaT = 1 / 1e-6;
    dfDataBase.setConstantValues(num_cells, num_surfaces, num_boundary_surfaces, num_patches, patch_size, Y.size(), rDeltaT);
    
    // prepare constant indexes: owner, neighbor
    dfDataBase.setConstantIndexes(&owner[0], &neighbour[0]);

    // prepare internal and boundary of sf, mag_sf, weights, delta_coeffs, volume
    double *boundary_sf = new double[3 * num_boundary_surfaces];
    double *boundary_mag_sf = new double[num_boundary_surfaces];
    double *boundary_delta_coeffs = new double[num_boundary_surfaces];
    int *boundary_face_cell = new int[num_boundary_surfaces];
    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.nonOrthDeltaCoeffs().boundaryField()[patchi];
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        int patchsize = pMagSf.size();

        memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
        memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
        memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
        memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));
        offset += patchsize;
    }

    dfDataBase.createConstantFieldsInternal();
    dfDataBase.createConstantFieldsBoundary();
    dfDataBase.initConstantFieldsInternal(&mesh.Sf()[0][0], &mesh.magSf()[0], &mesh.surfaceInterpolation::weights()[0], &mesh.nonOrthDeltaCoeffs()[0], &mesh.V()[0]);
    dfDataBase.initConstantFieldsBoundary(boundary_sf, boundary_mag_sf, boundary_delta_coeffs, boundary_face_cell);
    
    // prepare internal and boundary of Y
    dfDataBase.createNonConstantFieldsInternal();
    dfDataBase.createNonConstantFieldsBoundary();
    forAll(Y, speciesI) {
        volScalarField& Yi = Y[speciesI];
        memcpy(dfDataBase.h_y + speciesI * num_cells, &Yi[0], num_cells * sizeof(double));
        offset = 0;
        forAll(Yi.boundaryField(), patchi) {
            const scalarField& patchYi = Yi.boundaryField()[patchi];
            int patchsize = patchYi.size();
            memcpy(dfDataBase.h_boundary_y + speciesI * num_boundary_surfaces + offset, &patchYi[0], patchsize*sizeof(double));
            offset += patchsize;
        }
    }
    dfDataBase.initNonConstantFieldsInternal(dfDataBase.h_y);
    dfDataBase.initNonConstantFieldsBoundary(dfDataBase.h_boundary_y);
}

void createGPUUEqn(const IOdictionary& CanteraTorchProperties, const volVectorField& U) {
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    UEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(dfDataBase.num_patches);
    forAll(U.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), U.boundaryField()[patchi].type());
    }
    forAll(patch_type, patchi){
        Info << "patch_type_U " << patchi << " = " << patch_type[patchi] << endl;
    }
    UEqn_GPU.setConstantFields(patch_type);

    // prepare internal and boundary of xxx
    UEqn_GPU.createNonConstantFieldsInternal();
    UEqn_GPU.createNonConstantFieldsBoundary();
    UEqn_GPU.createNonConstantLduAndCsrFields();
    // UEqn_GPU has no internal non-constant fields to be init
    // UEqn_GPU.initNonConstantFieldsInternal();
    UEqn_GPU.initNonConstantFieldsBoundary();
}

void createGPUYEqn(const IOdictionary& CanteraTorchProperties, PtrList<volScalarField>& Y, const int inertIndex) {
    DEBUG_TRACE;
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    YEqn_GPU.setConstantValues(mode_string, settingPath, inertIndex);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(dfDataBase.num_patches);
    fprintf(stderr, "num_species: %d\n", dfDataBase.num_species);
    forAll(Y[0].boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), Y[0].boundaryField()[patchi].type());
    }
    YEqn_GPU.setConstantFields(patch_type);

    // prepare internal and boundary of xxx
    YEqn_GPU.createNonConstantFieldsInternal();
    YEqn_GPU.createNonConstantFieldsBoundary();
    YEqn_GPU.createNonConstantLduAndCsrFields();
    YEqn_GPU.initNonConstantFieldsInternal();
    YEqn_GPU.initNonConstantFieldsBoundary();
}

void createGPUEEqn(const IOdictionary& CanteraTorchProperties, volScalarField& he, volScalarField& K) {
    DEBUG_TRACE;
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    EEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type_he(dfDataBase.num_patches), patch_type_k(dfDataBase.num_patches);
    fprintf(stderr, "num_species: %d\n", dfDataBase.num_species);
    forAll(he.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_he[patchi]), he.boundaryField()[patchi].type());
        constructBoundarySelectorPerPatch(&(patch_type_k[patchi]), K.boundaryField()[patchi].type());
    }
    EEqn_GPU.setConstantFields(patch_type_he, patch_type_k);

    // prepare internal and boundary of xxx
    EEqn_GPU.createNonConstantFieldsInternal();
    EEqn_GPU.createNonConstantFieldsBoundary();
    EEqn_GPU.createNonConstantLduAndCsrFields();
}

dfMatrixDataBase dfDataBase;
//dfRhoEqn rhoEqn_GPU;
dfUEqn UEqn_GPU(dfDataBase);
//dfYEqn YEqn_GPU;
//dfEEqn EEqn_GPU;

void createGPUBase(fvMesh& mesh, PtrList<volScalarField>& Y) {
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
    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.nonOrthDeltaCoeffs().boundaryField()[patchi];

        int patchsize = pMagSf.size();

        memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
        memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
        memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
        offset += patchsize;
    }

    dfDataBase.createConstantFieldsInternal();
    dfDataBase.createConstantFieldsBoundary();
    dfDataBase.initConstantFieldsInternal(&mesh.Sf()[0][0], &mesh.magSf()[0], &mesh.surfaceInterpolation::weights()[0], &mesh.nonOrthDeltaCoeffs()[0], &mesh.V()[0]);
    dfDataBase.initConstantFieldsBoundary(boundary_sf, boundary_mag_sf, boundary_delta_coeffs);
    
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
    UEqn_GPU.setConstantFields(patch_type);

    // prepare internal and boundary of xxx
    UEqn_GPU.createNonConstantFieldsInternal();
    UEqn_GPU.createNonConstantFieldsBoundary();
    UEqn_GPU.createNonConstantLduAndCsrFields();
    // UEqn_GPU has no internal non-constant fields to be init
    // UEqn_GPU.initNonConstantFieldsInternal();
    UEqn_GPU.initNonConstantFieldsBoundary();
}

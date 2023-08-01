dfMatrixDataBase dfDataBase;

void createGPUBase(fvMesh& mesh, PtrList<volScalarField>& Y) {
    // obtain variables from fvMesh
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_surfaces = neighbour.size();

    
    // prepare num_cells, num_surfaces, num_boundary_surfaces, num_patches, patch_sizes, num_species, rdelta_t
    // - obtain boundary size info from mesh
    int patchSize = 0, num_patches = 0, num_boundary_surfaces = 0;
    std::vector<int> patch_sizes;
    forAll(mesh.boundary(), patchi) {
        labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
        patchSize = sub_boundary.size();

        patch_sizes.push_back(patchSize);
        num_boundary_surfaces += patchSize;
        num_patches ++;
    }
    dfDataBase.setConstantValues(num_cells, num_surfaces, num_boundary_surfaces, num_patches, patch_sizes, Y.size(), 1e-6); // TODO: get deltaT fomr time API
    
    // prepare owner, neighbor
    dfDataBase.setConstantIndexes(&owner[0], &neighbour[0]);
    
    // prepare internal and boundary of sf, mag_sf, weights, delta_coeffs, volume
    // - obtain boundary field info from mesh
    double *boundary_sf = new double[3 * num_boundary_surfaces];
    double *boundary_mag_sf = new double[num_boundary_surfaces];
    double *boundary_delta_coeffs = new double[num_boundary_surfaces];
    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.nonOrthDeltaCoeffs().boundaryField()[patchi];

        patchSize = pMagSf.size();

        memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchSize*sizeof(double));
        memcpy(boundary_mag_sf + offset, &pMagSf[0], patchSize*sizeof(double));
        memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchSize*sizeof(double));
        offset += patchSize;
    }

    dfDataBase.createConstantFieldsInternal();
    dfDataBase.createConstantFieldsBoundary();
    dfDataBase.initConstantFieldsInternal(&mesh.Sf()[0][0], &mesh.magSf()[0], &mesh.surfaceInterpolation::weights()[0], &mesh.nonOrthDeltaCoeffs()[0], &mesh.V()[0]);
    dfDataBase.initConstantFieldsBoundary(boundary_sf, boundary_mag_sf, boundary_delta_coeffs);
    
    // prepare internal and boundary of xxx
    // - obtain init_Y
    double *h_Y = new double[Y.size() * num_cells];
    double *boundary_Y = new double[Y.size() * num_boundary_surfaces];
    forAll(Y, speciesI) {
        volScalarField& Yi = Y[speciesI];
        memcpy(h_Y + speciesI * num_cells, &Yi[0], num_cells * sizeof(double));
        offset = 0;
        forAll(Yi.boundaryField(), patchi) {
            const scalarField& patchYi = Yi.boundaryField()[patchi];
            patchSize = patchYi.size();
            memcpy(boundary_Y + speciesI * num_boundary_surfaces + offset, &patchYi[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    dfDataBase.createNonConstantFieldsInternal();
    dfDataBase.createNonConstantFieldsBoundary();
    dfDataBase.initNonConstantFieldsInternal(h_Y);
    dfDataBase.initNonConstantFieldsBoundary(boundary_Y);
};
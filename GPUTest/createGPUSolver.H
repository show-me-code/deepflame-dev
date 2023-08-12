
dfMatrixDataBase dfDataBase;

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
};


dfMatrixDataBaseOrig* createGPUBaseOrig(fvMesh& mesh, PtrList<volScalarField>& Y, volVectorField& U) {
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_surfaces = neighbour.size();

    std::vector<int> boundaryCellIndex;
    std::vector<double> boundary_face_vector_init;
    std::vector<double> boundary_face_init;
    std::vector<double> boundary_deltaCoeffs_init;
    std::vector<std::vector<int>> patchTypes;
    std::vector<int> patchTypeU, patchTypeY;
    int num_boundary_faces = 0;
    int patchSize;
    forAll(mesh.boundary(), patchi)
    {
        labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
        patchSize = sub_boundary.size();
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.nonOrthDeltaCoeffs().boundaryField()[patchi];

        boundaryCellIndex.insert(boundaryCellIndex.end(), &sub_boundary[0], &sub_boundary[0]+patchSize);
        boundary_face_vector_init.insert(boundary_face_vector_init.end(), &pSf[0][0], &pSf[0][0]+3*patchSize);
        boundary_face_init.insert(boundary_face_init.end(), &pMagSf[0], &pMagSf[0]+patchSize);
        boundary_deltaCoeffs_init.insert(boundary_deltaCoeffs_init.end(), &pDeltaCoeffs[0], &pDeltaCoeffs[0]+patchSize);
        num_boundary_faces += patchSize;

        constructBoundarySelector(patchTypeU, U.boundaryField()[patchi].type(), patchSize);
        constructBoundarySelector(patchTypeY, Y[0].boundaryField()[patchi].type(), patchSize);
    }
    patchTypes.emplace_back(patchTypeU);
    patchTypes.emplace_back(patchTypeY);

    int num_boundary_cells;

    dfMatrixDataBaseOrig* dfDataBase = new dfMatrixDataBaseOrig(num_surfaces, num_cells, num_boundary_faces, Y.size(), num_boundary_cells, 
            &neighbour[0], &owner[0], &mesh.V()[0], &mesh.surfaceInterpolation::weights()[0], 
            &mesh.Sf()[0][0], &mesh.magSf()[0], &mesh.nonOrthDeltaCoeffs()[0], boundary_face_vector_init, 
            boundary_face_init, boundary_deltaCoeffs_init, boundaryCellIndex, patchTypes);
    
    return dfDataBase;
}
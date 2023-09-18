#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfNcclBase.H"

void constructBoundarySelectorPerPatch(int *patchTypeSelector, const std::string& patchTypeStr)
{
    boundaryConditions patchCondition;
    std::vector<int> tmpSelector;
    static std::map<std::string, boundaryConditions> BCMap = {
        {"zeroGradient", zeroGradient},
        {"fixedValue", fixedValue},
        {"empty", empty},
        {"gradientEnergy", gradientEnergy},
        {"calculated", calculated},
        {"coupled", coupled},
        {"cyclic", cyclic},
        {"processor", processor}
    };
    auto iter = BCMap.find(patchTypeStr);
    if (iter != BCMap.end()) {
        patchCondition = iter->second;
    } else {
        throw std::runtime_error("Unknown boundary condition: " + patchTypeStr);
    }
    // zeroGradient labeled as 0, fixedValue labeled as 1, coupled labeled as 2
    switch (patchCondition){
        case zeroGradient:
        {
            *patchTypeSelector = 0;
            break;
        }
        case fixedValue:
        {
            *patchTypeSelector = 1;
            break;
        }
        case coupled:
        {
            *patchTypeSelector = 2;
            break;
        }
        case empty:
        {
            *patchTypeSelector = 3;
            break;
        }
        case gradientEnergy:
        {
            *patchTypeSelector = 4;
            break;
        }
        case calculated:
        {
            *patchTypeSelector = 5;
            break;
        }
        case cyclic:
        {
            *patchTypeSelector = 6;
            break;
        }
        case processor:
        {
            *patchTypeSelector = 7;
            break;
        }
    }
}

dfMatrixDataBase::dfMatrixDataBase() {}

dfMatrixDataBase::~dfMatrixDataBase() {}

void dfMatrixDataBase::setCommInfo(MPI_Comm mpi_comm, ncclComm_t nccl_comm, ncclUniqueId nccl_id,
        int nRanks, int myRank, int localRank, std::vector<int> &neighbProcNo) {
    this->mpi_comm = mpi_comm;
    this->nccl_comm = nccl_comm;
    this->nccl_id = nccl_id;
    this->nRanks = nRanks;
    this->myRank = myRank;
    this->localRank = localRank;
    this->neighbProcNo = neighbProcNo;
    //fprintf(stderr, "myRank: %d, nRanks: %d, localRank: %d, neighbProcNo: %d\n", myRank, nRanks, localRank, neighbProcNo);
}
 
void dfMatrixDataBase::prepareCudaResources() {
    checkCudaErrors(cudaStreamCreate(&stream));
}

void dfMatrixDataBase::cleanCudaResources() {
    // destroy cuda resources
    checkCudaErrors(cudaStreamDestroy(stream));
    //ncclDestroy(nccl_comm);
    // TODO: free pointers
}

void dfMatrixDataBase::setConstantValues(int num_cells, int num_surfaces, int num_boundary_surfaces,
                   int num_patches, std::vector<int> patch_size,
                   int num_species, double rdelta_t) {
    // constant values -- basic
    this->num_cells = num_cells;
    this->num_surfaces = num_surfaces;
    this->num_boundary_surfaces = num_boundary_surfaces;
    this->num_patches = num_patches;
    this->patch_size = patch_size;
    this->num_species = num_species;
    this->rdelta_t = rdelta_t;

    // constant values -- ldu bytesize
    cell_value_bytes = num_cells * sizeof(double);
    cell_value_vec_bytes = num_cells * 3 * sizeof(double);
    cell_value_tsr_bytes = num_cells * 9 * sizeof(double);
    cell_index_bytes = num_cells * sizeof(int);
    surface_value_bytes = num_surfaces * sizeof(double);
    surface_index_bytes = num_surfaces * sizeof(int);
    surface_value_vec_bytes = num_surfaces * 3 * sizeof(double);
    boundary_surface_value_bytes = num_boundary_surfaces * sizeof(double);
    boundary_surface_value_vec_bytes = num_boundary_surfaces * 3 * sizeof(double);
    boundary_surface_value_tsr_bytes = num_boundary_surfaces * 9 * sizeof(double);
    boundary_surface_index_bytes = num_boundary_surfaces * sizeof(int);

    // constant values -- csr bytesize
    csr_row_index_bytes = (num_cells + 1) * sizeof(int);
    csr_col_index_bytes = (num_cells + num_surfaces * 2) * sizeof(int);
    csr_value_bytes = (num_cells + num_surfaces * 2) * sizeof(double);
    csr_value_vec_bytes = (num_cells + num_surfaces * 2) * 3 * sizeof(double);
}

void dfMatrixDataBase::setConstantIndexes(const int *owner, const int *neighbor) {
    // build d_owner, d_neighbor
    checkCudaErrors(cudaMalloc((void**)&d_owner, surface_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_neighbor, surface_index_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_owner, owner, surface_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_neighbor, neighbor, surface_index_bytes, cudaMemcpyHostToDevice, stream));


    // build d_lower_to_csr_index, d_diag_to_csr_index, d_upper_to_csr_index
    std::vector<int> upperNum(num_cells, 0);
    std::vector<int> lowerNum(num_cells, 0);
    std::vector<int> lowerPermListInit(num_surfaces);

    int *upperOffset = (int*)calloc(num_cells + 1, sizeof(int));
    int *lowerOffset = (int*)calloc(num_cells + 1, sizeof(int));

    for(int faceI = 0; faceI < num_surfaces; ++faceI){
        upperNum[owner[faceI]] ++;
        lowerNum[neighbor[faceI]] ++;
    }
    std::partial_sum(upperNum.begin(), upperNum.end(), 
        upperOffset+1);
    std::partial_sum(lowerNum.begin(), lowerNum.end(), 
        lowerOffset+1);

    std::iota(lowerPermListInit.begin(), lowerPermListInit.end(), 0);

    std::multimap<int,int> permutation;
    for (int i = 0; i < num_surfaces; ++i){
        permutation.insert(std::make_pair(neighbor[i], lowerPermListInit[i]));
    }
    std::vector<std::pair<int, int>> permPair(permutation.begin(), permutation.end());
    std::sort(permPair.begin(), permPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });

    std::vector<int> lowerPermList;
    std::transform(permPair.begin(), permPair.end(), std::back_inserter(lowerPermList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    }); 

    std::vector<int> lowCSRIndex, uppCSRIndex, diagCSRIndex, CSRRowIndex, CSRColIndex;
    int uppIndexInCSR = 0, uppIndexInLdu = 0, lowIndexInCSR = 0, lowIndexInLdu = 0, lowNumInLdu = 0;
    CSRColIndex.resize(2 * num_surfaces + num_cells);
    lowCSRIndex.resize(num_surfaces);
    for (int i = 0; i < num_cells; ++i) {
        int numUppPerRow = upperOffset[i + 1] - upperOffset[i];
        int numLowPerRow = lowerOffset[i + 1] - lowerOffset[i];
        int numNZBefore = upperOffset[i] + lowerOffset[i] + i; // add diag
        // csr row index
        CSRRowIndex.push_back(numNZBefore);
        // upper
        for (int j = 0; j < numUppPerRow; ++j) {
            uppIndexInCSR = numNZBefore + numLowPerRow + 1 + j; // 1 means diag
            uppCSRIndex.push_back(uppIndexInCSR);
            CSRColIndex[uppIndexInCSR] = neighbor[uppIndexInLdu]; // fill upper entry in CSRColIndex
            uppIndexInLdu ++;
        }
        // lower
        for (int j = 0; j < numLowPerRow; ++j) {
            lowIndexInCSR = numNZBefore + j;
            lowIndexInLdu = lowerPermList[lowNumInLdu];
            lowCSRIndex[lowIndexInLdu] = lowIndexInCSR;
            CSRColIndex[lowIndexInCSR] = owner[lowIndexInLdu]; // fill lower entry in CSRColIndex
            lowNumInLdu ++;
        }
        // diag
        int diagIndexInCSR = numNZBefore + numLowPerRow;
        diagCSRIndex.push_back(diagIndexInCSR);
        CSRColIndex[diagIndexInCSR] = i; // fill diag entry in CSRColIndex
    }
    int nNz = 2 * num_surfaces + num_cells;
    CSRRowIndex.push_back(nNz);

    // get reverseIndex from csr to ldu (low + diag + upp)
    std::vector<int> CSRIndex;
    CSRIndex.insert(CSRIndex.end(), lowCSRIndex.begin(), lowCSRIndex.end());
    CSRIndex.insert(CSRIndex.end(), diagCSRIndex.begin(), diagCSRIndex.end());
    CSRIndex.insert(CSRIndex.end(), uppCSRIndex.begin(), uppCSRIndex.end());

    std::vector<int> lduCSRIndexPermInit(nNz);
    std::iota(lduCSRIndexPermInit.begin(), lduCSRIndexPermInit.end(), 0);

    std::multimap<int,int> IndexPermutation;
    for (int i = 0; i < nNz; ++i){
        IndexPermutation.insert(std::make_pair(CSRIndex[i], lduCSRIndexPermInit[i]));
    }
    std::vector<std::pair<int, int>> IndexPermPair(IndexPermutation.begin(), IndexPermutation.end());
    std::sort(IndexPermPair.begin(), IndexPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        return pair1.first < pair2.first;});

    std::vector<int> lduCSRIndex;
    std::transform(IndexPermPair.begin(), IndexPermPair.end(), std::back_inserter(lduCSRIndex), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    std::vector<int> test;
    std::transform(IndexPermPair.begin(), IndexPermPair.end(), std::back_inserter(test), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    

    checkCudaErrors(cudaMalloc((void**)&d_lower_to_csr_index, surface_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag_to_csr_index, cell_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_upper_to_csr_index, surface_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_ldu_to_csr_index, csr_col_index_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_lower_to_csr_index, lowCSRIndex.data(), surface_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_diag_to_csr_index, diagCSRIndex.data(), cell_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_upper_to_csr_index, uppCSRIndex.data(), surface_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_ldu_to_csr_index, lduCSRIndex.data(), csr_col_index_bytes, cudaMemcpyHostToDevice, stream));


    // build d_csr_row_index, d_csr_col_index
    checkCudaErrors(cudaMalloc((void**)&d_csr_row_index, csr_row_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_csr_col_index, csr_col_index_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_csr_row_index, CSRRowIndex.data(), csr_row_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_csr_col_index, CSRColIndex.data(), csr_col_index_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::createConstantFieldsInternal() {
    checkCudaErrors(cudaMalloc((void**)&d_sf, surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_mag_sf, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_weight, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_weight, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_delta_coeffs, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_volume, cell_value_bytes));
    fieldPointerMap["d_sf"] = d_sf;
    fieldPointerMap["d_mag_sf"] = d_mag_sf;
    fieldPointerMap["d_weight"] = d_weight;
    fieldPointerMap["d_phi_weight"] = d_phi_weight;
    fieldPointerMap["d_delta_coeffs"] = d_delta_coeffs;
    fieldPointerMap["d_volume"] = d_volume;

    checkCudaErrors(cudaMallocHost((void**)&h_sf, surface_value_vec_bytes));
    fieldPointerMap["h_sf"] = h_sf;
}

void dfMatrixDataBase::createConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_sf, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mag_sf, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_delta_coeffs, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_face_cell, boundary_surface_index_bytes));
    fieldPointerMap["d_boundary_sf"] = d_boundary_sf;
    fieldPointerMap["d_boundary_mag_sf"] = d_boundary_mag_sf;
    fieldPointerMap["d_boundary_delta_coeffs"] = d_boundary_delta_coeffs;

    checkCudaErrors(cudaMallocHost((void**)&h_boundary_sf, boundary_surface_value_vec_bytes));
    fieldPointerMap["h_boundary_sf"] = h_boundary_sf;
}

void dfMatrixDataBase::initConstantFieldsInternal(const double *sf, const double *mag_sf, 
        const double *weight, const double *delta_coeffs, const double *volume) {
    // permute sf
    for (int i = 0; i < num_surfaces; i++) {
        h_sf[num_surfaces * 0 + i] = sf[i * 3 + 0];
        h_sf[num_surfaces * 1 + i] = sf[i * 3 + 1];
        h_sf[num_surfaces * 2 + i] = sf[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpyAsync(d_sf, h_sf, surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mag_sf, mag_sf, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_weight, weight, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_delta_coeffs, delta_coeffs, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_volume, volume, cell_value_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::initConstantFieldsBoundary(const double *boundary_sf, const double *boundary_mag_sf, 
        const double *boundary_delta_coeffs, const int *boundary_face_cell) {
    // permute bouSf
    for (int i = 0; i < num_boundary_surfaces; i++) {
        h_boundary_sf[num_boundary_surfaces * 0 + i] = boundary_sf[i * 3 + 0];
        h_boundary_sf[num_boundary_surfaces * 1 + i] = boundary_sf[i * 3 + 1];
        h_boundary_sf[num_boundary_surfaces * 2 + i] = boundary_sf[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpyAsync(d_boundary_sf, h_boundary_sf, boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_mag_sf, boundary_mag_sf, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_delta_coeffs, boundary_delta_coeffs, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_face_cell, boundary_face_cell, boundary_surface_index_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::createNonConstantFieldsInternal() {
    checkCudaErrors(cudaMalloc((void**)&d_rho, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u_old, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u_old_host_order, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_he, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_p, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_k, cell_value_bytes));
    fieldPointerMap["d_rho"] = d_rho;
    fieldPointerMap["d_u"] = d_u;
    fieldPointerMap["d_u_old"] = d_u_old;
    fieldPointerMap["d_y"] = d_y;
    fieldPointerMap["d_he"] = d_he;
    fieldPointerMap["d_p"] = d_p;
    fieldPointerMap["d_k"] = d_k;
    
    checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_k_old, cell_value_bytes));
    fieldPointerMap["d_rho_old"] = d_rho_old;
    fieldPointerMap["d_k_old"] = d_k_old;
    // checkCudaErrors(cudaMalloc((void**)&d_u_old, cell_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_y_old, cell_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_he_old, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_p_old, cell_value_bytes));
    fieldPointerMap["d_p_old"] = d_p_old;
    
    checkCudaErrors(cudaMalloc((void**)&d_phi, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_old, surface_value_bytes));
    fieldPointerMap["d_phi"] = d_phi;
    fieldPointerMap["d_phi_old"] = d_phi_old;

    // thermophysical fields
    checkCudaErrors(cudaMalloc((void**)&d_thermo_psi, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_thermo_alpha, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_hDiff_corr_flux, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diff_alphaD, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_dpdt, cell_value_bytes));

    // internal fields used between eqns
    checkCudaErrors(cudaMalloc((void**)&d_rAU, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_HbyA, cell_value_vec_bytes));

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_rho, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_rho_old, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_u_old, cell_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_he, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_k, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_k_old, cell_value_bytes));
    fieldPointerMap["h_rho"] = h_rho;
    fieldPointerMap["h_rho_old"] = h_rho_old;
    fieldPointerMap["h_u"] = h_u;
    fieldPointerMap["h_u_old"] = h_u_old;
    fieldPointerMap["h_y"] = h_y;
    fieldPointerMap["h_he"] = h_he;
    fieldPointerMap["h_k"] = h_k;
    fieldPointerMap["h_k_old"] = h_k_old;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_p, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_p_old, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_phi, surface_value_bytes));
    fieldPointerMap["h_p"] = h_p;
    fieldPointerMap["h_p_old"] = h_p_old;
    fieldPointerMap["h_phi"] = h_phi;
}

void dfMatrixDataBase::createNonConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old_host_order, boundary_surface_value_vec_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_he, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_p, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_k, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho"] = d_boundary_rho;
    fieldPointerMap["d_boundary_u"] = d_boundary_u;
    fieldPointerMap["d_boundary_u_old"] = d_boundary_u_old;
    fieldPointerMap["d_boundary_y"] = d_boundary_y;
    fieldPointerMap["d_boundary_he"] = d_boundary_he;
    fieldPointerMap["d_boundary_p"] = d_boundary_p;
    fieldPointerMap["d_boundary_k"] = d_boundary_k;

    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_old, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho_old"] = d_boundary_rho_old;
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old, boundary_surface_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_y_old, boundary_surface_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_he_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_p_old, boundary_surface_value_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_phi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_phi_old, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_phi"] = d_boundary_phi;
    fieldPointerMap["d_boundary_phi_old"] = d_boundary_phi_old;

    // thermophysical fields
    checkCudaErrors(cudaMalloc((void**)&d_boundary_thermo_psi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_thermo_alpha, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_hDiff_corr_flux, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_diff_alphaD, boundary_surface_value_bytes));

    // internal fields used between eqns
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rAU, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_HbyA, boundary_surface_value_vec_bytes));

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_u_old, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_he, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_k, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_rho"] = h_boundary_rho;
    fieldPointerMap["h_boundary_rho_old"] = h_boundary_rho_old;
    fieldPointerMap["h_boundary_u"] = h_boundary_u;
    fieldPointerMap["h_boundary_u_old"] = h_boundary_u_old;
    fieldPointerMap["h_boundary_y"] = h_boundary_y;
    fieldPointerMap["h_boundary_he"] = h_boundary_he;
    fieldPointerMap["h_boundary_k"] = h_boundary_k;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_p, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_p_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_phi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_phi_old, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_p"] = h_boundary_p;
    fieldPointerMap["h_boundary_p_old"] = h_boundary_p_old;
    fieldPointerMap["h_boundary_phi"] = h_boundary_phi;
    fieldPointerMap["h_boundary_phi_old"] = h_boundary_phi_old;
}

void dfMatrixDataBase::initNonConstantFieldsInternal(const double *y) {
    checkCudaErrors(cudaMemcpyAsync(d_y, y, cell_value_bytes * num_species, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::initNonConstantFieldsBoundary(const double *boundary_y) {
    checkCudaErrors(cudaMemcpyAsync(d_boundary_y, boundary_y, boundary_surface_value_bytes* num_species, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::preTimeStep(const double *rho_old, const double *phi_old, const double *boundary_phi_old, const double *u_old, 
        const double *boundary_u_old, const double *p_old, const double *boundary_p_old) {
    checkCudaErrors(cudaMemcpyAsync(d_rho_old, rho_old, cell_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_phi_old, phi_old, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_phi_old, boundary_phi_old, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    
    checkCudaErrors(cudaMemcpyAsync(d_u_old_host_order, u_old, cell_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_u_old_host_order, boundary_u_old, boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    
    permute_vector_h2d(stream, num_cells, d_u_old_host_order, d_u_old);
    permute_vector_h2d(stream, num_boundary_surfaces, d_boundary_u_old_host_order, d_boundary_u_old);

    checkCudaErrors(cudaMemcpyAsync(d_p_old, p_old, cell_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_p_old, boundary_p_old, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::postTimeStep() {}

double* dfMatrixDataBase::getFieldPointer(const char* fieldAlias, location loc, position pos) {
    char mergedName[256];
    if (pos == position::internal) {
        sprintf(mergedName, "%s_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    } else if (pos == position::boundary) {
        sprintf(mergedName, "%s_boundary_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    }

    double *pointer = nullptr;
    if (fieldPointerMap.find(std::string(mergedName)) != fieldPointerMap.end()) {
        pointer = fieldPointerMap[std::string(mergedName)];
    }
    if (pointer == nullptr) {
        fprintf(stderr, "Warning! getFieldPointer of %s returns nullptr!\n", mergedName);
    }
    //fprintf(stderr, "fieldAlias: %s, mergedName: %s, pointer: %p\n", fieldAlias, mergedName, pointer);

    return pointer;
}

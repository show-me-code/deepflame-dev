#include "dfMatrixDataBase.H"

void constructBoundarySelectorPerPatch(int *patchTypeSelector, const std::string& patchTypeStr)
{
    boundaryConditions patchCondition;
    std::vector<int> tmpSelector;
    static std::map<std::string, boundaryConditions> BCMap = {
        {"zeroGradient", zeroGradient},
        {"fixedValue", fixedValue},
        {"empty", empty},
        {"coupled", coupled}
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
        case empty:
        {
            *patchTypeSelector = 2;
            break;
        }
        case coupled:
        {
            *patchTypeSelector = 3;
            break;
        }
    }
}

dfMatrixDataBase::dfMatrixDataBase() {
    checkCudaErrors(cudaStreamCreate(&stream));
}

dfMatrixDataBase::~dfMatrixDataBase() {
    // destroy cuda resources
    checkCudaErrors(cudaStreamDestroy(stream));
    if (graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance));
        checkCudaErrors(cudaGraphDestroy(graph));
    }
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
    CSRRowIndex.push_back(0);
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

    checkCudaErrors(cudaMalloc((void**)&d_lower_to_csr_index, surface_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag_to_csr_index, cell_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_upper_to_csr_index, surface_index_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_lower_to_csr_index, lowCSRIndex.data(), surface_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_diag_to_csr_index, diagCSRIndex.data(), cell_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_upper_to_csr_index, uppCSRIndex.data(), surface_index_bytes, cudaMemcpyHostToDevice, stream));


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
    checkCudaErrors(cudaMalloc((void**)&d_delta_coeffs, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_volume, cell_value_bytes));
    fieldPointerMap["d_sf"] = d_sf;
    fieldPointerMap["d_mag_sf"] = d_mag_sf;
    fieldPointerMap["d_weight"] = d_weight;
    fieldPointerMap["d_delta_coeffs"] = d_delta_coeffs;
    fieldPointerMap["d_volume"] = d_volume;
}

void dfMatrixDataBase::createConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_sf, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mag_sf, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_delta_coeffs, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_face_cell, boundary_surface_index_bytes));
    fieldPointerMap["d_boundary_sf"] = d_boundary_sf;
    fieldPointerMap["d_boundary_mag_sf"] = d_boundary_mag_sf;
    fieldPointerMap["d_boundary_delta_coeffs"] = d_boundary_delta_coeffs;
}

void dfMatrixDataBase::initConstantFieldsInternal(const double *sf, const double *mag_sf, 
        const double *weight, const double *delta_coeffs, const double *volume) {
    checkCudaErrors(cudaMemcpyAsync(d_sf, sf, surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mag_sf, mag_sf, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_weight, weight, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_delta_coeffs, delta_coeffs, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_volume, volume, cell_value_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::initConstantFieldsBoundary(const double *boundary_sf, const double *boundary_mag_sf, 
        const double *boundary_delta_coeffs, const int *boundary_face_cell) {
    checkCudaErrors(cudaMemcpyAsync(d_boundary_sf, boundary_sf, boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_mag_sf, boundary_mag_sf, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_delta_coeffs, boundary_delta_coeffs, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_face_cell, boundary_face_cell, boundary_surface_index_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::createNonConstantFieldsInternal() {
    checkCudaErrors(cudaMalloc((void**)&d_rho, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_he, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_p, cell_value_bytes));
    fieldPointerMap["d_rho"] = d_rho;
    fieldPointerMap["d_u"] = d_u;
    fieldPointerMap["d_y"] = d_y;
    fieldPointerMap["d_he"] = d_he;
    fieldPointerMap["d_p"] = d_p;
    
    checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_value_bytes));
    fieldPointerMap["d_rho_old"] = d_rho_old;
    // checkCudaErrors(cudaMalloc((void**)&d_u_old, cell_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_y_old, cell_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_he_old, cell_value_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_p_old, cell_value_bytes));
    
    checkCudaErrors(cudaMalloc((void**)&d_phi, surface_value_bytes));
    fieldPointerMap["d_phi"] = d_phi;

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_rho, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_rho_old, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_he, cell_value_bytes));
    fieldPointerMap["h_rho"] = h_rho;
    fieldPointerMap["h_rho_old"] = h_rho_old;
    fieldPointerMap["h_u"] = h_u;
    fieldPointerMap["h_y"] = h_y;
    fieldPointerMap["h_he"] = h_he;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_p, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_phi, surface_value_bytes));
    fieldPointerMap["h_p"] = h_p;
    fieldPointerMap["h_phi"] = h_phi;
}

void dfMatrixDataBase::createNonConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_he, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_p, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho"] = d_boundary_rho;
    fieldPointerMap["d_boundary_u"] = d_boundary_u;
    fieldPointerMap["d_boundary_y"] = d_boundary_y;
    fieldPointerMap["d_boundary_he"] = d_boundary_he;
    fieldPointerMap["d_boundary_p"] = d_boundary_p;

    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_old, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho_old"] = d_boundary_rho_old;
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old, boundary_surface_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_y_old, boundary_surface_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_he_old, boundary_surface_value_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_p_old, boundary_surface_value_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_phi, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_phi"] = d_boundary_phi;

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_he, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_rho"] = h_boundary_rho;
    fieldPointerMap["h_boundary_rho_old"] = h_boundary_rho_old;
    fieldPointerMap["h_boundary_u"] = h_boundary_u;
    fieldPointerMap["h_boundary_y"] = h_boundary_y;
    fieldPointerMap["h_boundary_he"] = h_boundary_he;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_p, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_phi, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_p"] = h_boundary_p;
    fieldPointerMap["h_boundary_phi"] = h_boundary_phi;
}

void dfMatrixDataBase::initNonConstantFieldsInternal(const double *y) {
    checkCudaErrors(cudaMemcpyAsync(d_y, y, cell_value_bytes * num_species, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::initNonConstantFieldsBoundary(const double *boundary_y) {
    checkCudaErrors(cudaMemcpyAsync(d_boundary_y, boundary_y, boundary_surface_value_bytes* num_species, cudaMemcpyHostToDevice, stream));
}

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

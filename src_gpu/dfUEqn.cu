#include "dfUEqn.H"

void dfUEqn::setConstantValues(const std::string &mode_string, const std::string &setting_path) {
  this->mode_string = mode_string;
  this->setting_path = setting_path;
  UxSolver = new AmgXSolver(mode_string, setting_path);
  UySolver = new AmgXSolver(mode_string, setting_path);
  UzSolver = new AmgXSolver(mode_string, setting_path);
}

void dfUEqn::setConstantFields(const std::vector<int> patch_type) {
  this->patch_type = patch_type;
}

void dfUEqn::createNonConstantFieldsInternal() {
  // thermophysical fields
  checkCudaErrors(cudaMalloc((void**)&d_nu_eff, dataBase_.cell_value_bytes));
  // computed on CPU, used on GPU, need memcpyh2d
  checkCudaErrors(cudaMallocHost((void**)&h_nu_eff , dataBase_.cell_value_bytes));
  // intermediate fields
  checkCudaErrors(cudaMalloc((void**)&d_grad_u, dataBase_.cell_value_tsr_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_rho_nueff, dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_permute, dataBase_.cell_value_vec_bytes));

  // getter for h_nu_eff
  fieldPointerMap["h_nu_eff"] = h_nu_eff;
}
        
void dfUEqn::createNonConstantFieldsBoundary() {
  // thermophysical fields
  checkCudaErrors(cudaMalloc((void**)&d_boundary_nu_eff, dataBase_.boundary_surface_value_bytes));
  // computed on CPU, used on GPU, need memcpyh2d
  checkCudaErrors(cudaMallocHost((void**)&h_boundary_nu_eff, dataBase_.boundary_surface_value_bytes));
  // intermediate fields
  checkCudaErrors(cudaMalloc((void**)&d_boundary_grad_u, dataBase_.boundary_surface_value_tsr_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_nueff, dataBase_.boundary_surface_value_bytes));
  // boundary coeff fields
  checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));

  // getter for h_boundary_nu_eff
  fieldPointerMap["h_boundary_nu_eff"] = h_boundary_nu_eff;
}

void dfUEqn::createNonConstantLduAndCsrFields() {
  checkCudaErrors(cudaMalloc((void**)&d_lower, dataBase_.surface_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_upper, dataBase_.surface_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_diag, dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_b, dataBase_.cell_value_vec_bytes));
}

void dfUEqn::initNonConstantFieldsBoundary() {
    update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type.data(),
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
}

void dfUEqn::preProcess(const double *h_u, const double *h_boundary_u, const double *h_p, const double *h_boundary_p, 
        const double *h_nu_eff, const double *h_boundary_nu_eff, const double *h_boundary_rho) {
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_u, h_u, dataBase_.cell_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_u, h_boundary_u, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_p, h_p, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_p, h_boundary_p, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(d_nu_eff, h_nu_eff, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(d_boundary_nu_eff, h_boundary_nu_eff, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_rho, h_boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

  checkCudaErrors(cudaMemsetAsync(d_lower, 0, dataBase_.surface_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_upper, 0, dataBase_.surface_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_diag, 0, dataBase_.cell_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
}

void dfUEqn::preProcessForRhoEqn(const double *h_phi, const double *h_boundary_phi) {
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

  checkCudaErrors(cudaMemsetAsync(d_lower, 0, dataBase_.surface_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_upper, 0, dataBase_.surface_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_diag, 0, dataBase_.cell_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));

}

void dfUEqn::process() {
  // run each fvc or fvm function
  fvm_div_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_owner, dataBase_.d_neighbor,
          dataBase_.d_phi, dataBase_.d_weight,
          d_lower, d_upper, d_diag, // end for internal
          dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
          dataBase_.d_boundary_phi, d_value_internal_coeffs, d_value_boundary_coeffs,
          d_internal_coeffs, d_boundary_coeffs);
  //ldu_to_csr(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces,
  //        dataBase_.d_lower_to_csr_index, dataBase_.d_upper_to_csr_index, dataBase_.d_diag_to_csr_index,
  //        d_lower, d_upper, d_diag, d_source, d_internal_coeffs, d_boundary_coeffs, d_A, d_b);
  //solve();
}

void dfUEqn::solve() {
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));

    int nNz = dataBase_.num_cells + dataBase_.num_surfaces * 2; // matrix entries
    if (num_iteration == 0)                                     // first interation
    {
        printf("Initializing AmgX Linear Solver\n");
        UxSolver->setOperator(dataBase_.num_cells, nNz, dataBase_.d_csr_row_index, dataBase_.d_csr_col_index, d_A);
        UySolver->setOperator(dataBase_.num_cells, nNz, dataBase_.d_csr_row_index, dataBase_.d_csr_col_index, d_A + nNz);
        UzSolver->setOperator(dataBase_.num_cells, nNz, dataBase_.d_csr_row_index, dataBase_.d_csr_col_index, d_A + 2 * nNz);
    }
    else
    {
        UxSolver->updateOperator(dataBase_.num_cells, nNz, d_A);
        UySolver->updateOperator(dataBase_.num_cells, nNz, d_A + nNz);
        UzSolver->updateOperator(dataBase_.num_cells, nNz, d_A + 2 * nNz);
    }
    UxSolver->solve(dataBase_.num_cells, dataBase_.d_u, d_b);
    UySolver->solve(dataBase_.num_cells, dataBase_.d_u + dataBase_.num_cells, d_b + dataBase_.num_cells);
    UzSolver->solve(dataBase_.num_cells, dataBase_.d_u + 2 * dataBase_.num_cells, d_b + 2 * dataBase_.num_cells);
    num_iteration++;
}

void dfUEqn::postProcess(double *h_u) {
    permute_vector_d2h(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, d_permute);
    checkCudaErrors(cudaMemcpyAsync(h_u, d_permute, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));

    // some boundary conditions may also need vf.boundary, deltaCoeffs.boundary, and weight.boundary
    update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type.data(),
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
}

double* dfUEqn::getFieldPointer(const char* fieldAlias, location loc, position pos) {
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

void dfUEqn::compareResult(const double *lower, const double *upper, const double *diag, const double *internal_coeffs, const double *boundary_coeffs, bool printFlag)
{
    std::vector<double> h_lower;
    h_lower.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_surfaces, lower, h_lower.data(), 1e-14, printFlag);

    std::vector<double> h_upper;
    h_upper.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_surfaces, upper, h_upper.data(), 1e-14, printFlag);

    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);

    //std::vector<double> h_source;
    //h_source.resize(dataBase_.num_cells * 3);
    //checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    //checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);

    std::vector<double> h_internal_coeffs;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, internal_coeffs, h_internal_coeffs.data(), 1e-14, printFlag);

    std::vector<double> h_boundary_coeffs;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, boundary_coeffs, h_boundary_coeffs.data(), 1e-14, printFlag);
}


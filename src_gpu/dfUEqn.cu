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
  checkCudaErrors(cudaMallocHost((void**)&h_A_pEqn , dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMallocHost((void**)&h_H_pEqn , dataBase_.cell_value_vec_bytes));
  // intermediate fields
  checkCudaErrors(cudaMalloc((void**)&d_grad_u, dataBase_.cell_value_tsr_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_rho_nueff, dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_fvc_output, dataBase_.cell_value_vec_bytes));
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
  checkCudaErrors(cudaMalloc((void**)&d_ldu, dataBase_.csr_value_bytes));
  d_lower = d_ldu;
  d_diag = d_ldu + dataBase_.num_surfaces;
  d_upper = d_ldu + dataBase_.num_cells + dataBase_.num_surfaces;
  checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_b, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_A_pEqn, dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_H_pEqn, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_H_pEqn_perm, dataBase_.cell_value_vec_bytes));
}

void dfUEqn::initNonConstantFieldsBoundary() {
    update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_boundary_surfaces, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type.data(),
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
}

void dfUEqn::preProcessForRhoEqn(const double *h_rho, const double *h_phi, const double *h_boundary_phi) {
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_rho, h_rho, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
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

  checkCudaErrors(cudaMemsetAsync(d_ldu, 0, dataBase_.csr_value_bytes, dataBase_.stream)); // d_ldu contains d_lower, d_diag, and d_upper
  checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_A_pEqn, 0, dataBase_.cell_value_bytes, dataBase_.stream));
  checkCudaErrors(cudaMemsetAsync(d_H_pEqn, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
}

void dfUEqn::process() {
    //使用event计算时间
    float time_elapsed=0;
    cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start,0));

#ifndef TIME_GPU
    if(!graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

        permute_vector_h2d(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, d_permute);
        fvm_ddt_vector(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t,
                dataBase_.d_rho, dataBase_.d_rho_old, d_permute, dataBase_.d_volume,
                d_diag, d_source, 1.);
        fvm_div_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_phi, dataBase_.d_weight,
                d_lower, d_upper, d_diag, // end for internal
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_phi, d_value_internal_coeffs, d_value_boundary_coeffs,
                d_internal_coeffs, d_boundary_coeffs, 1.);
        field_multiply_scalar(dataBase_.stream,
               dataBase_.num_cells, dataBase_.d_rho, d_nu_eff, d_rho_nueff, // end for internal
               dataBase_.num_boundary_surfaces, dataBase_.d_boundary_rho, d_boundary_nu_eff, d_boundary_rho_nueff);
        fvm_laplacian_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
               dataBase_.d_owner, dataBase_.d_neighbor,
               dataBase_.d_weight, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs, d_rho_nueff,
               d_lower, d_upper, d_diag, // end for internal
               dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
               dataBase_.d_boundary_mag_sf, d_boundary_rho_nueff,
               d_gradient_internal_coeffs, d_gradient_boundary_coeffs,
               d_internal_coeffs, d_boundary_coeffs, -1);
        fvc_grad_vector(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
               dataBase_.d_owner, dataBase_.d_neighbor,
               dataBase_.d_weight, dataBase_.d_sf, d_permute, d_grad_u,
               dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
               dataBase_.d_boundary_face_cell, dataBase_.d_boundary_u, dataBase_.d_boundary_sf,
               dataBase_.d_volume, dataBase_.d_boundary_mag_sf, d_boundary_grad_u, dataBase_.d_boundary_delta_coeffs);
        scale_dev2T_tensor(dataBase_.stream, dataBase_.num_cells, d_rho_nueff, d_grad_u, // end for internal
               dataBase_.num_boundary_surfaces, d_boundary_rho_nueff, d_boundary_grad_u);
        fvc_div_cell_tensor(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
               dataBase_.d_owner, dataBase_.d_neighbor,
               dataBase_.d_weight, dataBase_.d_sf, d_grad_u, d_source, // end for internal
               dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
               dataBase_.d_boundary_face_cell, d_boundary_grad_u, dataBase_.d_boundary_sf, dataBase_.d_volume);
        fvc_grad_cell_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_p, d_source,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_p, dataBase_.d_boundary_sf, dataBase_.d_volume, -1.);
        ldu_to_csr(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.d_boundary_face_cell, dataBase_.d_ldu_to_csr_index, dataBase_.d_diag_to_csr_index, 
            d_ldu, d_source, d_internal_coeffs, d_boundary_coeffs, d_A, d_b);

#ifndef TIME_GPU
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));
        graph_created = true;
    }
    DEBUG_TRACE;
    checkCudaErrors(cudaGraphLaunch(graph_instance, dataBase_.stream));
#endif

    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed,start,stop));
    fprintf(stderr, "ueqn process time:%f(ms)\n",time_elapsed);

    //solve();
}

void dfUEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfUEqn::solve() {

    int nNz = dataBase_.num_cells + dataBase_.num_surfaces * 2; // matrix entries
    sync();

    // double *h_A_csr = new double[nNz * 3];
    // checkCudaErrors(cudaMemcpy(h_A_csr, d_A, dataBase_.csr_value_vec_bytes, cudaMemcpyDeviceToHost));
    // for (int i = 0; i < nNz; i++)
    //     fprintf(stderr, "h_A_csr[%d]: (%.10lf, %.10lf, %.10lf)\n", i, h_A_csr[i], h_A_csr[i + nNz], h_A_csr[i + 2 * nNz]);

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
    UxSolver->solve(dataBase_.num_cells, d_permute, d_b);
    UySolver->solve(dataBase_.num_cells, d_permute + dataBase_.num_cells, d_b + dataBase_.num_cells);
    UzSolver->solve(dataBase_.num_cells, d_permute + 2 * dataBase_.num_cells, d_b + 2 * dataBase_.num_cells);
    num_iteration++;
}

void dfUEqn::postProcess(double *h_u) {
    permute_vector_d2h(dataBase_.stream, dataBase_.num_cells, d_permute, dataBase_.d_u);
    checkCudaErrors(cudaMemcpyAsync(h_u, dataBase_.d_u, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));

    // some boundary conditions may also need vf.boundary, deltaCoeffs.boundary, and weight.boundary
    update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_boundary_surfaces, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type.data(),
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
}

void dfUEqn::A(double *Psi) {
    fvMtx_A(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.d_boundary_face_cell, d_internal_coeffs, dataBase_.d_volume, d_diag, d_A_pEqn);
    checkCudaErrors(cudaMemcpyAsync(h_A_pEqn, d_A_pEqn, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
    // TODO: correct Boundary conditions
    memcpy(Psi, h_A_pEqn, dataBase_.cell_value_bytes);
}

void dfUEqn::H(double *Psi) {
    fvMtx_H(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_volume,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
            dataBase_.d_boundary_face_cell, d_internal_coeffs, d_boundary_coeffs,
            d_lower, d_upper, d_source, d_permute, d_H_pEqn, d_H_pEqn_perm);
    checkCudaErrors(cudaMemcpyAsync(h_H_pEqn, d_H_pEqn_perm, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
    // TODO: correct Boundary conditions
    memcpy(Psi, h_H_pEqn, dataBase_.cell_value_vec_bytes);
}

void dfUEqn::correctPsi(double *Psi) {
    checkCudaErrors(cudaMemcpy(dataBase_.d_u, Psi, dataBase_.cell_value_vec_bytes, cudaMemcpyHostToDevice));
    permute_vector_h2d(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, d_permute);
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

void dfUEqn::compareResult(const double *lower, const double *upper, const double *diag, 
        const double *source, const double *internal_coeffs, const double *boundary_coeffs, 
        // const double *tmpVal, 
        bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_lower;
    h_lower.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_lower");
    checkVectorEqual(dataBase_.num_surfaces, lower, h_lower.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_upper;
    h_upper.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_upper");
    checkVectorEqual(dataBase_.num_surfaces, upper, h_upper.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_diag");
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source, h_source_ref;
    h_source.resize(dataBase_.num_cells * 3);
    h_source_ref.resize(dataBase_.num_cells * 3);
    for (int i = 0; i < dataBase_.num_cells; i++) {
        h_source_ref[0 * dataBase_.num_cells + i] = source[i * 3 + 0];
        h_source_ref[1 * dataBase_.num_cells + i] = source[i * 3 + 1];
        h_source_ref[2 * dataBase_.num_cells + i] = source[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source");
    checkVectorEqual(dataBase_.num_cells * 3, h_source_ref.data(), h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_internal_coeffs, h_internal_coeffs_ref;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    h_internal_coeffs_ref.resize(dataBase_.num_boundary_surfaces * 3);
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++) {
        h_internal_coeffs_ref[0 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 0];
        h_internal_coeffs_ref[1 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 1];
        h_internal_coeffs_ref[2 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_internal_coeffs_ref.data(), h_internal_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_coeffs, h_boundary_coeffs_ref;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    h_boundary_coeffs_ref.resize(dataBase_.num_boundary_surfaces * 3);
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++) {
        h_boundary_coeffs_ref[0 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 0];
        h_boundary_coeffs_ref[1 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 1];
        h_boundary_coeffs_ref[2 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_boundary_coeffs_ref.data(), h_boundary_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    // std::vector<double> h_tmpVal;
    // h_tmpVal.resize(dataBase_.num_cells * 3);
    // checkCudaErrors(cudaMemcpy(h_tmpVal.data(), d_fvc_output, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    // checkVectorEqual(dataBase_.num_cells * 3, tmpVal, h_tmpVal.data(), 1e-14, printFlag);
    // DEBUG_TRACE;
}


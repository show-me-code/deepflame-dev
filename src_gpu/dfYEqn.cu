#include "dfYEqn.H"

__global__ void yeqn_compute_phiUc_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *weight, const double *sf, const double *sumY_diff_error, double *phiUc)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double sfx = sf[index * 3 + 0];
    double sfy = sf[index * 3 + 1];
    double sfz = sf[index * 3 + 2];

    double w = weight[index]; 
    double ssfx = (w * (sumY_diff_error[owner * 3 + 0] - sumY_diff_error[neighbor * 3 + 0]) + sumY_diff_error[neighbor * 3 + 0]);
    double ssfy = (w * (sumY_diff_error[owner * 3 + 1] - sumY_diff_error[neighbor * 3 + 1]) + sumY_diff_error[neighbor * 3 + 1]);
    double ssfz = (w * (sumY_diff_error[owner * 3 + 2] - sumY_diff_error[neighbor * 3 + 2]) + sumY_diff_error[neighbor * 3 + 2]);

    phiUc[index] = sfx * ssfx + sfy * ssfy + sfz * ssfz;
}
 
__global__ void yeqn_compute_phiUc_boundary(int num_boundary_surfaces,
        const double *boundary_sf, const double *boundary_sumY_diff_error, double *boundary_phiUc)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_surfaces)
        return;

    double boundary_sfx = boundary_sf[index * 3 + 0];
    double boundary_sfy = boundary_sf[index * 3 + 1];
    double boundary_sfz = boundary_sf[index * 3 + 2];

    double boundary_ssfx = boundary_sumY_diff_error[index * 3 + 0];
    double boundary_ssfy = boundary_sumY_diff_error[index * 3 + 1];
    double boundary_ssfz = boundary_sumY_diff_error[index * 3 + 2];

    boundary_phiUc[index] = boundary_sfx * boundary_ssfx + boundary_sfy * boundary_ssfy + boundary_sfz * boundary_ssfz;
}
 
__global__ void yeqn_sumError_and_compute_hDiffCorrFlux(int num_species, int num,
        const double *rhoD, const double *hai, const double *y, const double *grady,
        double *sum_rhoD_grady, double *hDiffCorrFlux)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    double sum_hai_rhoD_grady_x = 0;
    double sum_hai_rhoD_grady_y = 0;
    double sum_hai_rhoD_grady_z = 0;
    double sum_rhoD_grady_x = 0;
    double sum_rhoD_grady_y = 0;
    double sum_rhoD_grady_z = 0;
    double sum_hai_y = 0;
    for (int s = 0; s < num_species; s++) {
        double hai_value = hai[num * s + index];
        double rhoD_value = rhoD[num * s + index];
        double y_value = y[num * s + index];
        double grady_x = grady[num * s * 3 + index * 3 + 0];
        double grady_y = grady[num * s * 3 + index * 3 + 1];
        double grady_z = grady[num * s * 3 + index * 3 + 2];
        sum_hai_rhoD_grady_x += hai_value * rhoD_value * grady_x;
        sum_hai_rhoD_grady_y += hai_value * rhoD_value * grady_y;
        sum_hai_rhoD_grady_z += hai_value * rhoD_value * grady_z;
        sum_rhoD_grady_x += rhoD_value * grady_x;
        sum_rhoD_grady_y += rhoD_value * grady_y;
        sum_rhoD_grady_z += rhoD_value * grady_z;
        sum_hai_y += hai_value * y_value;
    }
    sum_rhoD_grady[index * 3 + 0] = sum_rhoD_grady_x;
    sum_rhoD_grady[index * 3 + 1] = sum_rhoD_grady_y;
    sum_rhoD_grady[index * 3 + 2] = sum_rhoD_grady_z;
    hDiffCorrFlux[index * 3 + 0] = (sum_hai_rhoD_grady_x - sum_hai_y * sum_rhoD_grady_x);
    hDiffCorrFlux[index * 3 + 1] = (sum_hai_rhoD_grady_y - sum_hai_y * sum_rhoD_grady_y);
    hDiffCorrFlux[index * 3 + 2] = (sum_hai_rhoD_grady_z - sum_hai_y * sum_rhoD_grady_z);
}

__global__ void yeqn_fvc_laplacian_scalar_internal(int num_species, int num_cells, int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *mag_sf, const double *delta_coeffs, const double *weight,
        const double *rhoD, const double *hai, const double *vf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double w = weight[index];
    double magsf = mag_sf[index];
    double delta_coeff = delta_coeffs[index];

    // compute thermo_alpha
    // UnityLewis
    // alpha = nu * rho / 0.7
    // rhoD[i] = alpha
    double thermo_alpha_owner = rhoD[owner];
    double thermo_alpha_neighbor = rhoD[neighbor];

    //if (owner == 21 || neighbor == 21)
    //   printf("input index: %d, thermo: %.16lf, %.16lf\n", index, thermo_alpha_owner, thermo_alpha_neighbor);
    double sum_ssf = 0;
    for (int s = 0; s < num_species; s++) {
        double haii_owner = hai[num_cells * s + owner];
        double haii_neighbor = hai[num_cells * s + neighbor];
        double gamma = w * (thermo_alpha_owner * haii_owner) + (1 - w) * (thermo_alpha_neighbor * haii_neighbor);
        double sngrad = delta_coeff * (vf[num_cells * s + neighbor] - vf[num_cells * s + owner]);
        double ssf = gamma * sngrad * magsf;
        sum_ssf += ssf;
        //if (owner == 21 || neighbor == 21)
        //    printf("hai: %.16lf, %.16lf, gamma: %.16lf, sngrad: %.16lf, ssf: %.16lf\n", haii_owner, haii_neighbor, gamma, sngrad, ssf);
    }

    // owner
    atomicAdd(&(output[owner]), sum_ssf);
    // neighbor
    atomicAdd(&(output[neighbor]), -sum_ssf);
}

__global__ void yeqn_fvc_laplacian_scalar_boundary_fixedValue(int num_species, int num_cells, int num_boundary_surfaces,
        int num, int offset, const int *face2Cells,
        const double *boundary_mag_sf, const double *boundary_delta_coeffs,
        const double *boundary_rhoD, const double *boundary_hai,
        const double *vf, const double *boundary_vf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    int cellIndex = face2Cells[index];

    double boundary_delta_coeff = boundary_delta_coeffs[start_index];
    double boundary_magsf = boundary_mag_sf[start_index];
    // compute thermo_alpha
    // UnityLewis
    // alpha = nu * rho / 0.7
    // rhoD[i] = alpha
    double boundary_thermo_alpha = boundary_rhoD[start_index];

    double sum_boundary_ssf = 0;
    for (int s = 0; s < num_species; s++) {
        // sn_grad: solving according to fixedValue BC
        double boundary_sngrad = boundary_delta_coeff * (boundary_vf[num_boundary_surfaces * s + start_index] - vf[num_cells * s + cellIndex]);
        double boundary_gamma = boundary_thermo_alpha * boundary_hai[num_boundary_surfaces * s + start_index];
        double boundary_ssf = boundary_gamma * boundary_sngrad * boundary_magsf;
        sum_boundary_ssf += boundary_ssf;
    }

    atomicAdd(&(output[cellIndex]), sum_boundary_ssf);
}

__global__ void yeqn_buildBC_scalar(int num_boundary_surfaces,
        const int *face2Cells, const double *output, double *boundary_output)

{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_surfaces)
        return;

    int cellIndex = face2Cells[index];
    boundary_output[index] = output[cellIndex];
}

__global__ void yeqn_divide_cell_volume_scalar(int num_cells, const double* volume, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double vol = volume[index];

    output[index] = output[index] / vol * sign;
}

__global__ void yeqn_compute_y_inertIndex_kernel(int num_species, int inertIndex, int num_cells, double *y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    double sum_y = 0;
    for (int s = 0; s < num_species; s++) {
        if (s == inertIndex) continue;

        double yi = y[num_cells * s + index];

        y[num_cells * s + index] = yi > 0 ? yi : 0;
        sum_y += yi > 0 ? yi : 0;
    }
    sum_y = 1 - sum_y;
    y[num_cells * inertIndex + index] = (sum_y > 0 ? sum_y : 0);
}

double* dfYEqn::getFieldPointer(const char* fieldAlias, location loc, position pos) {
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

void dfYEqn::setConstantValues(const std::string &mode_string, const std::string &setting_path, const int inertIndex) {
    this->mode_string = mode_string;
    this->setting_path = setting_path;
    this->inertIndex = inertIndex;
    YSolverSet.resize(dataBase_.num_species - 1); // consider inert species
    for (auto &solver : YSolverSet)
        solver = new AmgXSolver(mode_string, setting_path);
}

void dfYEqn::setConstantFields(const std::vector<int> patch_type) {
    this->patch_type = patch_type;
}

void dfYEqn::createNonConstantFieldsInternal() {
    // thermophysical fields - read
    checkCudaErrors(cudaMalloc((void**)&d_rhoD, dataBase_.cell_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_hai, dataBase_.cell_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_mut_sct, dataBase_.cell_value_bytes));
    // thermophysical fields - write
    checkCudaErrors(cudaMalloc((void**)&d_hDiff_corr_flux, dataBase_.cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diff_alphaD, dataBase_.cell_value_bytes));
    // intermediate fields
    checkCudaErrors(cudaMalloc((void**)&d_grad_y, dataBase_.cell_value_vec_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_sumY_diff_error, dataBase_.cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phiUc, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_special_weight, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phiUc_special_weight, dataBase_.surface_value_bytes));
    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_rhoD, dataBase_.cell_value_bytes * dataBase_.num_species));
    // UnityLewis
    checkCudaErrors(cudaMallocHost((void**)&h_hai, dataBase_.cell_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_mut_sct, dataBase_.cell_value_bytes));
    // getter for h_xxx
    // UnityLewis
    fieldPointerMap["h_rhoD"] = h_rhoD;
    fieldPointerMap["h_hai"] = h_hai;
    fieldPointerMap["h_mut_sct"] = h_mut_sct;
}

void dfYEqn::createNonConstantFieldsBoundary() {
    // thermophysical fields - read
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rhoD, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_hai, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mut_sct, dataBase_.boundary_surface_value_bytes));
    // thermophysical fields - write
    checkCudaErrors(cudaMalloc((void**)&d_boundary_hDiff_corr_flux, dataBase_.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_diff_alphaD, dataBase_.boundary_surface_value_bytes));
    // intermediate fields
    checkCudaErrors(cudaMalloc((void**)&d_boundary_grad_y, dataBase_.boundary_surface_value_vec_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_sumY_diff_error, dataBase_.boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_phiUc, dataBase_.boundary_surface_value_bytes));
    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rhoD, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    // UnityLewis
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_hai, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_mut_sct, dataBase_.boundary_surface_value_bytes));
    // boundary coeff fields
    checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_bytes * dataBase_.num_species));
    // getter for h_boundary_xxx
    fieldPointerMap["h_boundary_rhoD"] = h_boundary_rhoD;
    // UnityLewis
    fieldPointerMap["h_boundary_hai"] = h_boundary_hai;
    fieldPointerMap["h_boundary_mut_sct"] = h_boundary_mut_sct;
}

void dfYEqn::createNonConstantLduAndCsrFields() {
    checkCudaErrors(cudaMalloc((void**)&d_lower, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_upper, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag, dataBase_.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_bytes));
    // use d_source as d_b
    //checkCudaErrors(cudaMalloc((void**)&d_b, dataBase_.cell_value_bytes));
}

void dfYEqn::initNonConstantFieldsInternal() {
    // UnityLewis
    //checkCudaErrors(cudaMemsetAsync(d_hai, 0, dataBase_.cell_value_bytes * dataBase_.num_species, dataBase_.stream));
    //checkCudaErrors(cudaMemsetAsync(d_boundary_hai, 0, dataBase_.boundary_surface_value_bytes * dataBase_.num_species, dataBase_.stream));
    //checkCudaErrors(cudaMemsetAsync(d_mut_sct, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    //checkCudaErrors(cudaMemsetAsync(d_boundary_mut_sct, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
}

void dfYEqn::initNonConstantFieldsBoundary() {
    for (int s = 0; s < dataBase_.num_species; s++) {
        update_boundary_coeffs_scalar(dataBase_.stream,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_y + dataBase_.num_boundary_surfaces * s,
                d_value_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                d_value_boundary_coeffs + dataBase_.num_boundary_surfaces * s,
                d_gradient_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                d_gradient_boundary_coeffs + dataBase_.num_boundary_surfaces * s);
    }
}

void dfYEqn::preProcess(const double *h_rhoD, const double *h_boundary_rhoD,
        const double *h_hai, const double *h_boundary_hai,
        const double *h_mut_sct, const double *h_boundary_mut_sct) {
    checkCudaErrors(cudaMemcpyAsync(d_rhoD, h_rhoD, dataBase_.cell_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_rhoD, h_boundary_rhoD, dataBase_.boundary_surface_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice, dataBase_.stream));
    // UnityLewis
    checkCudaErrors(cudaMemcpyAsync(d_hai, h_hai, dataBase_.cell_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_hai, h_boundary_hai, dataBase_.boundary_surface_value_bytes * dataBase_.num_species, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(d_mut_sct, h_mut_sct, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_mut_sct, h_boundary_mut_sct, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

    checkCudaErrors(cudaMemsetAsync(d_hDiff_corr_flux, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_hDiff_corr_flux, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_diff_alphaD, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_diff_alphaD, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_sumY_diff_error, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_sumY_diff_error, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));

    DEBUG_TRACE;
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
    DEBUG_TRACE;
}

void dfYEqn::process() {
    //使用event计算时间
    float time_elapsed=0;
    cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start,0));

    //if(!graph_created) {
    //    DEBUG_TRACE;
    //    checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));

        // compute diffAlphaD
        yeqn_fvc_laplacian_scalar(dataBase_.stream, dataBase_.num_species,
                dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs, dataBase_.d_volume,
                d_rhoD, d_hai, dataBase_.d_y, d_diff_alphaD, // end for internal
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_mag_sf, dataBase_.d_boundary_delta_coeffs,
                d_boundary_rhoD, d_boundary_hai, dataBase_.d_boundary_y, d_boundary_diff_alphaD, 1.);
        // fvc::grad(Yi)
        for (int s = 0; s < dataBase_.num_species; s++) {
            fvc_grad_cell_scalar_withBC(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                    dataBase_.d_owner, dataBase_.d_neighbor,
                    dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_y + dataBase_.num_cells * s, d_grad_y + dataBase_.num_cells * s * 3,
                    dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                    dataBase_.d_boundary_face_cell, dataBase_.d_boundary_y + dataBase_.num_boundary_surfaces * s, dataBase_.d_boundary_sf,
                    dataBase_.d_volume, dataBase_.d_boundary_mag_sf, d_boundary_grad_y + dataBase_.num_boundary_surfaces * s * 3,
                    dataBase_.d_boundary_delta_coeffs);
        }
        // compute sumYDiffError and hDiffCorrFlux
        yeqn_compute_sumYDiffError_and_hDiffCorrFlux(dataBase_.stream,
                dataBase_.num_species, dataBase_.num_cells, dataBase_.num_boundary_surfaces,
                d_rhoD, d_hai, dataBase_.d_y, d_grad_y, d_sumY_diff_error, d_hDiff_corr_flux,
                d_boundary_rhoD, d_boundary_hai, dataBase_.d_boundary_y, d_boundary_grad_y,
                d_boundary_sumY_diff_error, d_boundary_hDiff_corr_flux);
        // compute phiUc
        yeqn_compute_phiUc(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_sf, d_sumY_diff_error, d_phiUc,
                dataBase_.d_boundary_sf, d_boundary_sumY_diff_error, d_boundary_phiUc);
        // compute upwind weight of phi and phiUc: only need internal upwind-weight
        compute_upwind_weight(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_phi, d_phi_special_weight);
        compute_upwind_weight(dataBase_.stream, dataBase_.num_surfaces, d_phiUc, d_phiUc_special_weight);
        // compute DEff
        // UnityLewis
        // tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
        // turbulence->mut()/Sct = 0 when UnityLewis.
        //field_add(dataBase_.stream, dataBase_.num_cells, d_rhoD, d_mut_sct, d_DEff,
        //        dataBase_.num_surfaces, d_boundary_rhoD, d_boundary_mut_sct, d_boundary_DEff);
        double *d_DEff = d_rhoD;
        double *d_boundary_DEff = d_boundary_rhoD;

        // construct YiEqn and solve
#if defined DEBUG_
        for (int s = 0; s < 1; s++) {
#else
        for (int s = 0; s < dataBase_.num_species; s++) {
#endif
            if (s != this->inertIndex) {
                // reset ldu structures used cross YiEqn
                checkCudaErrors(cudaMemsetAsync(d_lower, 0, dataBase_.surface_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_upper, 0, dataBase_.surface_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_diag, 0, dataBase_.cell_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
                checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_bytes, dataBase_.stream));
                // use d_source as d_b
                //checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_bytes, dataBase_.stream));
                // fvm::ddt(rho, Yi)
                fvm_ddt_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t,
                        dataBase_.d_rho, dataBase_.d_rho_old, dataBase_.d_y + dataBase_.num_cells * s, dataBase_.d_volume,
                        d_diag, d_source, 1.);
                // fvmDiv(phi, Yi)
                fvm_div_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_owner, dataBase_.d_neighbor,
                        dataBase_.d_phi, d_phi_special_weight,
                        d_lower, d_upper, d_diag, // end for internal
                        dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                        dataBase_.d_boundary_phi,
                        d_value_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_value_boundary_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_internal_coeffs, d_boundary_coeffs, 1.);
                // fvmDiv(phiUc, Yi)
                fvm_div_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.d_owner, dataBase_.d_neighbor,
                        d_phiUc, d_phiUc_special_weight,
                        d_lower, d_upper, d_diag, // end for internal
                        dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                        d_boundary_phiUc,
                        d_value_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_value_boundary_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_internal_coeffs, d_boundary_coeffs, 1.);
                // fvm::laplacian(DEff(), Yi)
                fvm_laplacian_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                        dataBase_.d_owner, dataBase_.d_neighbor,
                        dataBase_.d_weight, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs,
                        d_DEff + dataBase_.num_cells * s,
                        d_lower, d_upper, d_diag, // end for internal
                        dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                        dataBase_.d_boundary_mag_sf, d_boundary_DEff + dataBase_.num_boundary_surfaces * s,
                        d_gradient_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_gradient_boundary_coeffs + dataBase_.num_boundary_surfaces * s,
                        d_internal_coeffs, d_boundary_coeffs, -1.);
#if defined DEBUG_
#else                
                // ldu to csr
                // use d_source as d_b
                ldu_to_csr_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                        dataBase_.d_boundary_face_cell,
                        dataBase_.d_lower_to_csr_index, dataBase_.d_upper_to_csr_index, dataBase_.d_diag_to_csr_index,
                        d_lower, d_upper, d_diag, d_source, d_internal_coeffs, d_boundary_coeffs, d_A);
                // not open solve yet
                //solve(s);
                //if (s == dataBase_.num_species - 1)
                //    num_iteration++;
#endif                
            }
        }
    //    checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph));
    //    checkCudaErrors(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));
    //    graph_created = true;
    //}
    //DEBUG_TRACE;
    //checkCudaErrors(cudaGraphLaunch(graph_instance, dataBase_.stream));

    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed,start,stop));
    fprintf(stderr, "yeqn process time：%f(ms)\n",time_elapsed);
}

void dfYEqn::solve(int solverIndex) {
    int nNz = dataBase_.num_cells + dataBase_.num_surfaces * 2; // matrix entries
    if (num_iteration == 0)                                     // first interation
    {
        fprintf(stderr, "Initializing AmgX Linear Solver\n");
        DEBUG_TRACE;
        YSolverSet[solverIndex]->setOperator(dataBase_.num_cells, nNz, dataBase_.d_csr_row_index, dataBase_.d_csr_col_index, d_A);
        DEBUG_TRACE;
    }
    else
    {
        DEBUG_TRACE;
        YSolverSet[solverIndex]->updateOperator(dataBase_.num_cells, nNz, d_A);
        DEBUG_TRACE;
    }
    // use d_source as d_b
    DEBUG_TRACE;
    YSolverSet[solverIndex]->solve(dataBase_.num_cells, dataBase_.d_y + dataBase_.num_cells * solverIndex, d_source);
    DEBUG_TRACE;
}

void dfYEqn::postProcess(double *h_y) {
    // compute y_inertIndex
    yeqn_compute_y_inertIndex(dataBase_.stream, dataBase_.num_species, inertIndex, dataBase_.num_cells, dataBase_.d_y);

    // update boundary coeffs
    for (int s = 0; s < dataBase_.num_species; s++) {
        update_boundary_coeffs_scalar(dataBase_.stream,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_y + dataBase_.num_boundary_surfaces * s,
                d_value_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                d_value_boundary_coeffs + dataBase_.num_boundary_surfaces * s,
                d_gradient_internal_coeffs + dataBase_.num_boundary_surfaces * s,
                d_gradient_boundary_coeffs + dataBase_.num_boundary_surfaces * s);
    }

    // copy y to host
    checkCudaErrors(cudaMemcpyAsync(h_y, dataBase_.d_y, dataBase_.cell_value_bytes * dataBase_.num_species, cudaMemcpyDeviceToHost, dataBase_.stream));
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfYEqn::yeqn_fvc_laplacian_scalar(cudaStream_t stream, int num_species, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
        const double *rhoD, const double *hai, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face,
        const double *boundary_mag_sf, const double *boundary_delta_coeffs,
        const double *boundary_rhoD, const double *boundary_hai, const double *boundary_vf,
        double *boundary_output, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * sizeof(double), stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    yeqn_fvc_laplacian_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_species, num_cells, num_surfaces,
            lowerAddr, upperAddr, mag_sf, delta_coeffs, weight, rhoD, hai, vf, output);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient) {
            //fprintf(stderr, "patch_type is zeroGradient\n");
        } else if (patch_type[i] == boundaryConditions::fixedValue) {
            //fprintf(stderr, "patch_type is fixedValue\n");
            // TODO: just vector version now
            yeqn_fvc_laplacian_scalar_boundary_fixedValue<<<blocks_per_grid, threads_per_block, 0, stream>>>(
                    num_species, num_cells, num_boundary_surfaces, patch_size[i], offset, boundary_cell_face,
                    boundary_mag_sf, boundary_delta_coeffs, boundary_rhoD, boundary_hai, vf, boundary_vf, output);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }

    // divide cell volume
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    yeqn_divide_cell_volume_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);

    // correct boundary condition
    yeqn_buildBC_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces,
            boundary_cell_face, output, boundary_output);
}

void dfYEqn::yeqn_compute_sumYDiffError_and_hDiffCorrFlux(cudaStream_t stream, int num_species, int num_cells, int num_boundary_surfaces,
        const double *rhoD, const double *hai, const double *y, const double *grad_y,
        double *sumY_diff_error, double *hDiff_corr_flux,
        const double *boundary_rhoD, const double *boundary_hai, const double *boundary_y, const double *boundary_grad_y,
        double *boundary_sumY_diff_error, double *boundary_hDiff_corr_flux)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    yeqn_sumError_and_compute_hDiffCorrFlux<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_species, num_cells,
            rhoD, hai, y, grad_y, sumY_diff_error, hDiff_corr_flux);
    blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    yeqn_sumError_and_compute_hDiffCorrFlux<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_species, num_boundary_surfaces,
            boundary_rhoD, boundary_hai, boundary_y, boundary_grad_y, boundary_sumY_diff_error, boundary_hDiff_corr_flux);
}

void dfYEqn::yeqn_compute_phiUc(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *sf, const double *sumY_diff_error, double *phiUc,
        const double *boundary_sf, const double *boundary_sumY_diff_error, double *boundary_phiUc)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    yeqn_compute_phiUc_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces,
            lowerAddr, upperAddr, weight, sf, sumY_diff_error, phiUc);
    blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    yeqn_compute_phiUc_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces,
            boundary_sf, boundary_sumY_diff_error, boundary_phiUc);
}

void dfYEqn::yeqn_compute_y_inertIndex(cudaStream_t stream, int num_species, int inertIndex, int num_cells, double *y)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    yeqn_compute_y_inertIndex_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_species, inertIndex, num_cells, y);
}

#if defined DEBUG_
void dfYEqn::comparediffAlphaD(const double *diffAlphaD, const double *boundary_diffAlphaD, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_diffAlphaD;
    h_diffAlphaD.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diffAlphaD.data(), d_diff_alphaD, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells, diffAlphaD, h_diffAlphaD.data(), 1e-10, printFlag);
    DEBUG_TRACE;
    std::vector<double> h_boundary_diffAlphaD;
    h_boundary_diffAlphaD.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_diffAlphaD.data(), d_boundary_diff_alphaD, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_diffAlphaD, h_boundary_diffAlphaD.data(), 1e-10, printFlag);
    DEBUG_TRACE;
}

void dfYEqn::comparegrady0(const double *grad_y0, const double *boundary_grad_y0, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_grad_y0;
    h_grad_y0.resize(dataBase_.num_cells * 3);
    checkCudaErrors(cudaMemcpy(h_grad_y0.data(), d_grad_y, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells * 3, grad_y0, h_grad_y0.data(), 1e-10, printFlag);
    DEBUG_TRACE;
    std::vector<double> h_boundary_grad_y0;
    h_boundary_grad_y0.resize(dataBase_.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_boundary_grad_y0.data(), d_boundary_grad_y, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, boundary_grad_y0, h_boundary_grad_y0.data(), 1e-10, printFlag);
    DEBUG_TRACE;

}

void dfYEqn::comparesumYDiffError(const double *sumYDiffError, const double *boundary_sumYDiffError, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_sumYDiffError;
    h_sumYDiffError.resize(dataBase_.num_cells * 3);
    checkCudaErrors(cudaMemcpy(h_sumYDiffError.data(), d_sumY_diff_error, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells * 3, sumYDiffError, h_sumYDiffError.data(), 1e-10, printFlag);
    DEBUG_TRACE;
    std::vector<double> h_boundary_sumYDiffError;
    h_boundary_sumYDiffError.resize(dataBase_.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_boundary_sumYDiffError.data(), d_boundary_sumY_diff_error, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, boundary_sumYDiffError, h_boundary_sumYDiffError.data(), 1e-10, printFlag);
    DEBUG_TRACE;
}

void dfYEqn::comparehDiffCorrFlux(const double *hDiffCorrFlux, const double *boundary_hDiffCorrFlux, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_hDiffCorrFlux;
    h_hDiffCorrFlux.resize(dataBase_.num_cells * 3);
    checkCudaErrors(cudaMemcpy(h_hDiffCorrFlux.data(), d_hDiff_corr_flux, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells * 3, hDiffCorrFlux, h_hDiffCorrFlux.data(), 1e-10, printFlag);
    DEBUG_TRACE;
    std::vector<double> h_boundary_hDiffCorrFlux;
    h_boundary_hDiffCorrFlux.resize(dataBase_.num_boundary_surfaces * 3);
    checkCudaErrors(cudaMemcpy(h_boundary_hDiffCorrFlux.data(), d_boundary_hDiff_corr_flux, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, boundary_hDiffCorrFlux, h_boundary_hDiffCorrFlux.data(), 1e-10, printFlag);
    DEBUG_TRACE;
}

void dfYEqn::comparephiUc(const double *phiUc, const double *boundary_phiUc,  bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_phiUc;
    h_phiUc.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_phiUc.data(), d_phiUc, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_surfaces, phiUc, h_phiUc.data(), 1e-10, printFlag);
    DEBUG_TRACE;
    std::vector<double> h_boundary_phiUc;
    h_boundary_phiUc.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_phiUc.data(), d_boundary_phiUc, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_phiUc, h_boundary_phiUc.data(), 1e-10, printFlag);
    DEBUG_TRACE;
}

void dfYEqn::compareResult(const double *lower, const double *upper, const double *diag, const double *source,
        const double *internal_coeffs, const double *boundary_coeffs, bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_lower;
    h_lower.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_surfaces, lower, h_lower.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_upper;
    h_upper.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_surfaces, upper, h_upper.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source;
    h_source.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_internal_coeffs;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces, internal_coeffs, h_internal_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_coeffs;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_coeffs, h_boundary_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}

#endif

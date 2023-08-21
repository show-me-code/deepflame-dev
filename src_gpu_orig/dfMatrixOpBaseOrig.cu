#include "dfMatrixOpBaseOrig.H"


__global__ void fvc_grad_vector_internal(int num_cells,
                                         const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                         const double *sf, const double *vf, const double *tlambdas, const double *volume,
                                         double *grad)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int row_elements = csr_row_index[index + 1] - row_index;
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_vf_x = vf[index * 3 + 0];
    double own_vf_y = vf[index * 3 + 1];
    double own_vf_z = vf[index * 3 + 2];
    double grad_xx = 0;
    double grad_xy = 0;
    double grad_xz = 0;
    double grad_yx = 0;
    double grad_yy = 0;
    double grad_yz = 0;
    double grad_zx = 0;
    double grad_zy = 0;
    double grad_zz = 0;
    // lower
    for (int i = 0; i < diag_index; i++)
    {
        int neighbor_index = neighbor_offset + i;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_x = vf[neighbor_cell_id * 3 + 0];
        double neighbor_vf_y = vf[neighbor_cell_id * 3 + 1];
        double neighbor_vf_z = vf[neighbor_cell_id * 3 + 2];
        double face_x = (1 - w) * own_vf_x + w * neighbor_vf_x;
        double face_y = (1 - w) * own_vf_y + w * neighbor_vf_y;
        double face_z = (1 - w) * own_vf_z + w * neighbor_vf_z;
        grad_xx -= sf_x * face_x;
        grad_xy -= sf_x * face_y;
        grad_xz -= sf_x * face_z;
        grad_yx -= sf_y * face_x;
        grad_yy -= sf_y * face_y;
        grad_yz -= sf_y * face_z;
        grad_zx -= sf_z * face_x;
        grad_zy -= sf_z * face_y;
        grad_zz -= sf_z * face_z;
    }
    // upper
    for (int i = diag_index + 1; i < row_elements; i++)
    {
        int neighbor_index = neighbor_offset + i - 1;
        int neighbor_cell_id = csr_col_index[row_index + i];
        double w = tlambdas[neighbor_index];
        double sf_x = sf[neighbor_index * 3 + 0];
        double sf_y = sf[neighbor_index * 3 + 1];
        double sf_z = sf[neighbor_index * 3 + 2];
        double neighbor_vf_x = vf[neighbor_cell_id * 3 + 0];
        double neighbor_vf_y = vf[neighbor_cell_id * 3 + 1];
        double neighbor_vf_z = vf[neighbor_cell_id * 3 + 2];
        double face_x = w * own_vf_x + (1 - w) * neighbor_vf_x;
        double face_y = w * own_vf_y + (1 - w) * neighbor_vf_y;
        double face_z = w * own_vf_z + (1 - w) * neighbor_vf_z;
        grad_xx += sf_x * face_x;
        grad_xy += sf_x * face_y;
        grad_xz += sf_x * face_z;
        grad_yx += sf_y * face_x;
        grad_yy += sf_y * face_y;
        grad_yz += sf_y * face_z;
        grad_zx += sf_z * face_x;
        grad_zy += sf_z * face_y;
        grad_zz += sf_z * face_z;
        // if (index == 0)
        // {
        //     printf("grad_xx = %.20lf\n", grad_xx);
        //     // printf("sf_x = %.20lf\n", sf_x);
        //     // printf("face_x = %.20lf\n", face_x);
        // }
    }
    double vol = volume[index];
    grad[index * 9 + 0] = grad_xx / vol;
    grad[index * 9 + 1] = grad_xy / vol;
    grad[index * 9 + 2] = grad_xz / vol;
    grad[index * 9 + 3] = grad_yx / vol;
    grad[index * 9 + 4] = grad_yy / vol;
    grad[index * 9 + 5] = grad_yz / vol;
    grad[index * 9 + 6] = grad_zx / vol;
    grad[index * 9 + 7] = grad_zy / vol;
    grad[index * 9 + 8] = grad_zz / vol;


    // if (index == 2257)
    // {
    //     printf("grad[2257] = (%.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e)\n", grad[index * 9 + 0], grad[index * 9 + 1], grad[index * 9 + 2],
    //             grad[index * 9 + 3], grad[index * 9 + 4], grad[index * 9 + 5], grad[index * 9 + 6], grad[index * 9 + 7], grad[index * 9 + 8]);
    // }
}

__global__ void fvc_grad_vector_boundary(int num_cells, int num_boundary_cells, const int *bouPermedIndex,
                                         const int *boundary_cell_offset, const int *boundary_cell_id,
                                         const double *boundary_sf, const double *boundary_vf, const double *volume,
                                         double *grad, double *grad_boundary_init)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    double grad_xx = 0;
    double grad_xy = 0;
    double grad_xz = 0;
    double grad_yx = 0;
    double grad_yy = 0;
    double grad_yz = 0;
    double grad_zx = 0;
    double grad_zy = 0;
    double grad_zz = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        int p = bouPermedIndex[i];
        double sf_x = boundary_sf[i * 3 + 0];
        double sf_y = boundary_sf[i * 3 + 1];
        double sf_z = boundary_sf[i * 3 + 2];
        double vf_x = boundary_vf[p * 3 + 0];
        double vf_y = boundary_vf[p * 3 + 1];
        double vf_z = boundary_vf[p * 3 + 2];
        grad_xx += sf_x * vf_x;
        grad_xy += sf_x * vf_y;
        grad_xz += sf_x * vf_z;
        grad_yx += sf_y * vf_x;
        grad_yy += sf_y * vf_y;
        grad_yz += sf_y * vf_z;
        grad_zx += sf_z * vf_x;
        grad_zy += sf_z * vf_y;
        grad_zz += sf_z * vf_z;
    }

    double vol = volume[cell_index];

    grad[cell_index * 9 + 0] += grad_xx / vol;
    grad[cell_index * 9 + 1] += grad_xy / vol;
    grad[cell_index * 9 + 2] += grad_xz / vol;
    grad[cell_index * 9 + 3] += grad_yx / vol;
    grad[cell_index * 9 + 4] += grad_yy / vol;
    grad[cell_index * 9 + 5] += grad_yz / vol;
    grad[cell_index * 9 + 6] += grad_zx / vol;
    grad[cell_index * 9 + 7] += grad_zy / vol;
    grad[cell_index * 9 + 8] += grad_zz / vol;

    grad_boundary_init[index * 9 + 0] = grad[cell_index * 9 + 0];
    grad_boundary_init[index * 9 + 1] = grad[cell_index * 9 + 1];
    grad_boundary_init[index * 9 + 2] = grad[cell_index * 9 + 2];
    grad_boundary_init[index * 9 + 3] = grad[cell_index * 9 + 3];
    grad_boundary_init[index * 9 + 4] = grad[cell_index * 9 + 4];
    grad_boundary_init[index * 9 + 5] = grad[cell_index * 9 + 5];
    grad_boundary_init[index * 9 + 6] = grad[cell_index * 9 + 6];
    grad_boundary_init[index * 9 + 7] = grad[cell_index * 9 + 7];
    grad_boundary_init[index * 9 + 8] = grad[cell_index * 9 + 8];

    // if (index == 0)
    // {
    //     printf("grad[0] = (%.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e, %.5e)\n", grad[index * 9 + 0], grad[index * 9 + 1], grad[index * 9 + 2],
    //             grad[index * 9 + 3], grad[index * 9 + 4], grad[index * 9 + 5], grad[index * 9 + 6], grad[index * 9 + 7], grad[index * 9 + 8]);
    // }
}

__global__ void correct_boundary_conditions(int num_boundary_cells, const int *bouPermedIndex,
                                            const int *boundary_cell_offset, const int *boundary_cell_id,
                                            const double *boundary_sf, const double *mag_sf,
                                            double *boundary_grad_init, double *boundary_grad, const double *boundary_deltaCoeffs,
                                            const double *internal_velocity, const double *boundary_velocity, const int *U_patch_type)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // initialize boundary_grad
    double grad_xx = boundary_grad_init[index * 9 + 0];
    double grad_xy = boundary_grad_init[index * 9 + 1];
    double grad_xz = boundary_grad_init[index * 9 + 2];
    double grad_yx = boundary_grad_init[index * 9 + 3];
    double grad_yy = boundary_grad_init[index * 9 + 4];
    double grad_yz = boundary_grad_init[index * 9 + 5];
    double grad_zx = boundary_grad_init[index * 9 + 6];
    double grad_zy = boundary_grad_init[index * 9 + 7];
    double grad_zz = boundary_grad_init[index * 9 + 8];

    double internal_U_x = internal_velocity[cell_index * 3 + 0];
    double internal_U_y = internal_velocity[cell_index * 3 + 1];
    double internal_U_z = internal_velocity[cell_index * 3 + 2];

    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        // OpenFoam code
        // const vectorField n
        //     (
        //      vsf.mesh().Sf().boundaryField()[patchi]
        //      / vsf.mesh().magSf().boundaryField()[patchi]
        //     );
        // gGradbf[patchi] += n *
        //     (
        //      vsf.boundaryField()[patchi].snGrad()
        //      - (n & gGradbf[patchi])
        //     );
        // template<class Type> // fixedValue
        // Foam::tmp<Foam::Field<Type>> Foam::fvPatchField<Type>::snGrad() const
        // {
        //     return patch_.deltaCoeffs()*(*this - patchInternalField());
        // }

        double n_x = boundary_sf[i * 3 + 0] / mag_sf[i];
        double n_y = boundary_sf[i * 3 + 1] / mag_sf[i];
        double n_z = boundary_sf[i * 3 + 2] / mag_sf[i];

        int p = bouPermedIndex[i];

        double sn_grad_x, sn_grad_y, sn_grad_z;
        int patchIndex = U_patch_type[i];
        if (patchIndex == 0) { // zeroGradient
            sn_grad_x = 0;
            sn_grad_y = 0;
            sn_grad_z = 0;
        } else if (patchIndex == 1) { // fixedValue
            sn_grad_x = boundary_deltaCoeffs[i] * (boundary_velocity[p * 3 + 0] - internal_velocity[cell_index * 3 + 0]);
            sn_grad_y = boundary_deltaCoeffs[i] * (boundary_velocity[p * 3 + 1] - internal_velocity[cell_index * 3 + 1]);
            sn_grad_z = boundary_deltaCoeffs[i] * (boundary_velocity[p * 3 + 2] - internal_velocity[cell_index * 3 + 2]);
            // if (index == 1)
            // {
            //     printf("cell_index = %d\n", cell_index);
            //     printf("boundary_velocity = %e\n", boundary_velocity[i * 3 + 1]);
            //     printf("internal_velocity = %e\n", internal_velocity[cell_index * 3 + 0]);
            // }
            
        }
        // TODO: implement other BCs
        double grad_correction_x = sn_grad_x - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx);
        double grad_correction_y = sn_grad_y - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
        double grad_correction_z = sn_grad_z - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);
        boundary_grad[i * 9 + 0] = grad_xx + n_x * grad_correction_x;
        boundary_grad[i * 9 + 1] = grad_xy + n_x * grad_correction_y;
        boundary_grad[i * 9 + 2] = grad_xz + n_x * grad_correction_z;
        boundary_grad[i * 9 + 3] = grad_yx + n_y * grad_correction_x;
        boundary_grad[i * 9 + 4] = grad_yy + n_y * grad_correction_y;
        boundary_grad[i * 9 + 5] = grad_yz + n_y * grad_correction_z;
        boundary_grad[i * 9 + 6] = grad_zx + n_z * grad_correction_x;
        boundary_grad[i * 9 + 7] = grad_zy + n_z * grad_correction_y;
        boundary_grad[i * 9 + 8] = grad_zz + n_z * grad_correction_z;
        
    }
}

__global__ void fvc_grad_scalar_internal(int num_cells,
                                       const int *csr_row_index, const int *csr_col_index, const int *csr_diag_index,
                                       const double *face_vector, const double *weight, const double *pressure, const double *volume,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;

    double own_cell_p = pressure[index];
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    double grad_bx_low = 0;
    double grad_bx_upp = 0;
    double grad_by_low = 0;
    double grad_by_upp = 0;
    double grad_bz_low = 0;
    double grad_bz_upp = 0;
    for (int i = row_index; i < next_row_index; i++)
    {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index;
            double w = weight[neighbor_index];
            double sfx = face_vector[neighbor_index * 3 + 0];
            double sfy = face_vector[neighbor_index * 3 + 1];
            double sfz = face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
            grad_bx_low -= face_p * sfx;
            grad_by_low -= face_p * sfy;
            grad_bz_low -= face_p * sfz;
        }
        // upper
        if (inner_index > diag_index)
        {
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = weight[neighbor_index];
            double sfx = face_vector[neighbor_index * 3 + 0];
            double sfy = face_vector[neighbor_index * 3 + 1];
            double sfz = face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = csr_col_index[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = w * own_cell_p + (1 - w) * neighbor_cell_p;
            grad_bx_upp += face_p * sfx;
            grad_by_upp += face_p * sfy;
            grad_bz_upp += face_p * sfz;
        }
    }
    double vol = volume[index];
    b_output[index * 3 + 0] = b_input[index * 3 + 0] + (grad_bx_low + grad_bx_upp) / vol;
    b_output[index * 3 + 1] = b_input[index * 3 + 1] + (grad_by_low + grad_by_upp) / vol;
    b_output[index * 3 + 2] = b_input[index * 3 + 2] + (grad_bz_low + grad_bz_upp) / vol;
    // b_output[index * 3 + 0] = b_input[index * 3 + 0] + grad_bx_low + grad_bx_upp;
    // b_output[index * 3 + 1] = b_input[index * 3 + 1] + grad_by_low + grad_by_upp;
    // b_output[index * 3 + 2] = b_input[index * 3 + 2] + grad_bz_low + grad_bz_upp;

}

__global__ void fvc_grad_scalar_boundary(int num_cells, int num_boundary_cells, const int *bouPermedIndex,
                                       const int *boundary_cell_offset, const int *boundary_cell_id,
                                       const double *boundary_face_vector, const double *boundary_pressure, const double *volume,
                                       const double *b_input, double *b_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells)
        return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // compute boundary gradient
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    for (int i = cell_offset; i < next_cell_offset; i++)
    {
        int p = bouPermedIndex[i];
        double sfx = boundary_face_vector[i * 3 + 0];
        double sfy = boundary_face_vector[i * 3 + 1];
        double sfz = boundary_face_vector[i * 3 + 2];
        double face_p = boundary_pressure[p];
        grad_bx += face_p * sfx;
        grad_by += face_p * sfy;
        grad_bz += face_p * sfz;
    }

    //// correct the boundary gradient
    // double nx = boundary_face_vector[face_index * 3 + 0] / magSf[face_index];
    // double ny = boundary_face_vector[face_index * 3 + 1] / magSf[face_index];
    // double nz = boundary_face_vector[face_index * 3 + 2] / magSf[face_index];
    // double sn_grad = 0;
    // double grad_correction = sn_grad * volume[cell_index] - (nx * grad_bx + ny * grad_by + nz * grad_bz);
    // grad_bx += nx * grad_correction;
    // grad_by += ny * grad_correction;
    // grad_bz += nz * grad_correction;

    double vol = volume[cell_index];
    b_output[cell_index * 3 + 0] = b_input[cell_index * 3 + 0] + grad_bx / vol;
    b_output[cell_index * 3 + 1] = b_input[cell_index * 3 + 1] + grad_by / vol;
    b_output[cell_index * 3 + 2] = b_input[cell_index * 3 + 2] + grad_bz / vol;
}


void fvc_grad_vector_orig(cudaStream_t stream, dfMatrixDataBaseOrig* dataBaseOrig, dfMatrixDataBase& dataBase, double *d_grad, 
        double *d_grad_boundary_init, double *d_grad_boundary)
{
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (dataBase.num_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase.num_cells,
                                                                                dataBaseOrig->d_A_csr_row_index, dataBaseOrig->d_A_csr_col_index, dataBaseOrig->d_A_csr_diag_index,
                                                                                dataBaseOrig->d_face_vector, dataBase.d_u, dataBaseOrig->d_weight, dataBaseOrig->d_volume, d_grad);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    printf("\nfvc_grad_vector_orig internal 执行时间：%f(ms)\n", time_elapsed);
    
    
    checkCudaErrors(cudaEventRecord(start, 0));
    blocks_per_grid = (dataBaseOrig->num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase.num_cells, dataBaseOrig->num_boundary_cells, dataBaseOrig->d_bouPermedIndex,
                                                                                dataBaseOrig->d_boundary_cell_offset, dataBaseOrig->d_boundary_cell_id, dataBaseOrig->d_boundary_face_vector, 
                                                                                dataBase.d_boundary_u, dataBase.d_volume, d_grad, d_grad_boundary_init);
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    printf("fvc_grad_vector_orig boundary1 执行时间：%f(ms)\n", time_elapsed);
    
    
    checkCudaErrors(cudaEventRecord(start, 0));
    correct_boundary_conditions<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBaseOrig->num_boundary_cells, dataBaseOrig->d_bouPermedIndex,
                                                                                   dataBaseOrig->d_boundary_cell_offset, dataBaseOrig->d_boundary_cell_id, dataBaseOrig->d_boundary_face_vector, 
                                                                                   dataBaseOrig->d_boundary_face, d_grad_boundary_init, d_grad_boundary, dataBaseOrig->d_boundary_deltaCoeffs, 
                                                                                   dataBase.d_u, dataBase.d_boundary_u, dataBaseOrig->d_boundary_UpatchType);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    printf("fvc_grad_vector_orig boundary2 执行时间：%f(ms)\n", time_elapsed);
}

void fvc_grad_scalar_orig(cudaStream_t stream, dfMatrixDataBaseOrig* dataBaseOrig, dfMatrixDataBase& dataBase, double *d_grad)
{
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (dataBase.num_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase.num_cells,
                                                                                dataBaseOrig->d_A_csr_row_index, dataBaseOrig->d_A_csr_col_index, dataBaseOrig->d_A_csr_diag_index,
                                                                                dataBaseOrig->d_face_vector, dataBaseOrig->d_weight, dataBase.d_p, dataBaseOrig->d_volume, d_grad, d_grad);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    printf("\nfvc_grad_scalar_orig internal 执行时间：%f(ms)\n", time_elapsed);

    checkCudaErrors(cudaEventRecord(start, 0));

    blocks_per_grid = (dataBaseOrig->num_boundary_cells + threads_per_block - 1) / threads_per_block;
    fvc_grad_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(dataBase.num_cells, dataBaseOrig->num_boundary_cells, dataBaseOrig->d_bouPermedIndex,
                                                                              dataBaseOrig->d_boundary_cell_offset, dataBaseOrig->d_boundary_cell_id,
                                                                              dataBaseOrig->d_boundary_face_vector, dataBase.d_boundary_p, dataBaseOrig->d_volume, d_grad, d_grad);
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed, start, stop));
    printf("fvc_grad_scalar_orig boundary 执行时间：%f(ms)\n", time_elapsed);
}
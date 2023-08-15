#include "dfMatrixOpBase.H"
#include "dfMatrixDataBase.H"

#include <cuda_runtime.h>
#include "cuda_profiler_api.h"

__global__ void permute_vector_d2h_kernel(int num_cells, const double *input, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    output[index * 3 + 0] = input[num_cells * 0 + index];
    output[index * 3 + 1] = input[num_cells * 1 + index];
    output[index * 3 + 2] = input[num_cells * 2 + index];
}

__global__ void permute_vector_h2d_kernel(int num_cells, const double *input, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    output[num_cells * 0 + index] = input[index * 3 + 0];
    output[num_cells * 1 + index] = input[index * 3 + 1];
    output[num_cells * 2 + index] = input[index * 3 + 2];
}

__global__ void field_multiply_scalar_kernel(int num_cells, int num_boundary_surfaces,
        const double *input1, const double *input2, double *output,
        const double *boundary_input1, const double *boundary_input2, double *boundary_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < num_cells) {
        output[index] = input1[index] * input2[index];
    }
    if (index < num_boundary_surfaces) {
        boundary_output[index] = boundary_input1[index] * boundary_input2[index];
    }
}

__global__ void fvc_to_source_vector_kernel(int num_cells, const double *volume, const double *fvc_output, double *source)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    source[index * 3 + 0] += fvc_output[index * 3 + 0] * volume[index];
    source[index * 3 + 1] += fvc_output[index * 3 + 1] * volume[index];
    source[index * 3 + 2] += fvc_output[index * 3 + 2] * volume[index];
}

__global__ void update_boundary_coeffs_zeroGradient_vector(int num, int offset,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    // valueInternalCoeffs = 1
    // valueBoundaryCoeffs = 0
    // gradientInternalCoeffs = 0
    // gradientBoundaryCoeffs = 0
    value_internal_coeffs[start_index * 3 + 0] = 1;
    value_internal_coeffs[start_index * 3 + 1] = 1;
    value_internal_coeffs[start_index * 3 + 2] = 1;
    value_boundary_coeffs[start_index * 3 + 0] = 0;
    value_boundary_coeffs[start_index * 3 + 1] = 0;
    value_boundary_coeffs[start_index * 3 + 2] = 0;
    gradient_internal_coeffs[start_index * 3 + 0] = 0;
    gradient_internal_coeffs[start_index * 3 + 1] = 0;
    gradient_internal_coeffs[start_index * 3 + 2] = 0;
    gradient_boundary_coeffs[start_index * 3 + 0] = 0;
    gradient_boundary_coeffs[start_index * 3 + 1] = 0;
    gradient_boundary_coeffs[start_index * 3 + 2] = 0;
}

__global__ void scale_dev2t_tensor_kernel(int num, const double *vf1, double *vf2)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    double scale = vf1[index];
    double val_xx = vf2[index * 9 + 0];
    double val_xy = vf2[index * 9 + 1];
    double val_xz = vf2[index * 9 + 2];
    double val_yx = vf2[index * 9 + 3];
    double val_yy = vf2[index * 9 + 4];
    double val_yz = vf2[index * 9 + 5];
    double val_zx = vf2[index * 9 + 6];
    double val_zy = vf2[index * 9 + 7];
    double val_zz = vf2[index * 9 + 8];
    double trace_coeff = (2. / 3.) * (val_xx + val_yy + val_zz);
    vf2[index * 9 + 0] = scale * (val_xx - trace_coeff);
    vf2[index * 9 + 1] = scale * val_yx;
    vf2[index * 9 + 2] = scale * val_zx;
    vf2[index * 9 + 3] = scale * val_xy;
    vf2[index * 9 + 4] = scale * (val_yy - trace_coeff);
    vf2[index * 9 + 5] = scale * val_zy;
    vf2[index * 9 + 6] = scale * val_xz;
    vf2[index * 9 + 7] = scale * val_yz;
    vf2[index * 9 + 8] = scale * (val_zz - trace_coeff);
}

__global__ void fvm_ddt_vector_kernel(int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    diag[index] += rDeltaT * rho[index] * volume[index] * sign;
    // TODO: skip moving
    source[index * 3 + 0] += rDeltaT * rho_old[index] * vf[index * 3 + 0] * volume[index] * sign;
    source[index * 3 + 1] += rDeltaT * rho_old[index] * vf[index * 3 + 1] * volume[index] * sign;
    source[index * 3 + 2] += rDeltaT * rho_old[index] * vf[index * 3 + 2] * volume[index] * sign;
}

__global__ void fvm_div_vector_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double f = phi[index];

    double lower_value = (-w) * f * sign;
    double upper_value = (1 - w) * f * sign;
    lower[index] += lower_value;
    upper[index] += upper_value;
    // if (index == 0) printf("index = 0, lower: %.16lf, upper:%.16lf\n", lower[index], upper[index]);

    int owner = lower_index[index];
    int neighbor = upper_index[index];
    atomicAdd(&(diag[owner]), -lower_value);
    atomicAdd(&(diag[neighbor]), -upper_value);
}

__global__ void fvm_div_vector_boundary(int num, int offset,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    double boundary_f = boundary_phi[start_index];
    internal_coeffs[start_index * 3 + 0] += boundary_f * value_internal_coeffs[start_index * 3 + 0] * sign;
    internal_coeffs[start_index * 3 + 1] += boundary_f * value_internal_coeffs[start_index * 3 + 1] * sign;
    internal_coeffs[start_index * 3 + 2] += boundary_f * value_internal_coeffs[start_index * 3 + 2] * sign;
    boundary_coeffs[start_index * 3 + 0] += boundary_f * value_boundary_coeffs[start_index * 3 + 0] * sign;
    boundary_coeffs[start_index * 3 + 1] += boundary_f * value_boundary_coeffs[start_index * 3 + 1] * sign;
    boundary_coeffs[start_index * 3 + 2] += boundary_f * value_boundary_coeffs[start_index * 3 + 2] * sign;
}

__global__ void fvm_laplacian_vector_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double w = weight[index];
    double upper_face_gamma = w * gamma[owner] + (1 - w) * gamma[neighbor];
    double upper_value = upper_face_gamma * mag_sf[index] * delta_coeffs[index];

    // laplacian doesn't use the original lower, but use lower = upper
    //double lower_face_gamma = w * gamma[neighbor] + (1 - w) * gamma[owner];
    //double lower_value = lower_face_gamma * mag_sf[index] * delta_coeffs[index];
    double lower_value = upper_value;

    lower_value = lower_value * sign;
    upper_value = upper_value * sign;

    lower[index] += lower_value;
    upper[index] += upper_value;

    atomicAdd(&(diag[owner]), -lower_value);
    atomicAdd(&(diag[neighbor]), -upper_value);
}

__global__ void fvm_laplacian_vector_boundary(int num, int offset,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    double boundary_value = boundary_gamma[start_index] * boundary_mag_sf[start_index];
    internal_coeffs[start_index * 3 + 0] += boundary_value * gradient_internal_coeffs[start_index * 3 + 0] * sign;
    internal_coeffs[start_index * 3 + 1] += boundary_value * gradient_internal_coeffs[start_index * 3 + 1] * sign;
    internal_coeffs[start_index * 3 + 2] += boundary_value * gradient_internal_coeffs[start_index * 3 + 2] * sign;
    boundary_coeffs[start_index * 3 + 0] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 0] * sign;
    boundary_coeffs[start_index * 3 + 1] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 1] * sign;
    boundary_coeffs[start_index * 3 + 2] += boundary_value * gradient_boundary_coeffs[start_index * 3 + 2] * sign;
}

__global__ void fvc_ddt_scalar_kernel(int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *vf_old,
        double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    /*
    // workaround way1 (use printf):
    double val_new = rho[index] * vf[index];
    double val_old = rho_old[index] * vf_old[index];
    // TODO: skip moving
    // TODO: wyr
    // for the case of rho = rho_old and vf = vf_old, the floating-point numerical problem will be exposed.
    // it expect zero as output, but the gpu result get a sub-normal minimal value for (val_new - val_old),
    // which smaller than 1e-16, and then enlarged by rDeltaT (1e6)
    // then the comparison of cpu result and gpu result will failed with relative error: inf,
    // e.g.:
    // cpu data: 0.0000000000000000, gpu data: 0.0000000000298050, relative error: inf
    // if I add the print line for intermediate variables of val_new and val_old, the problem disappears.
    // It seems that print line will change the compiler behavior, maybe avoiding the fma optimization of compiler.
    if (index == -1) printf("index = 0, val_new: %.40lf, val_old: %.40lf\n", val_new, val_old);
    output[index] += rDeltaT * (val_new - val_old);
    */
    /*
    // workaround way2 (use volatile):
    // volatile will change the compiler behavior, maybe avoiding the fma optimization of compiler.
    volatile double val_new = rho[index] * vf[index];
    volatile double val_old = rho_old[index] * vf_old[index];
    output[index] += rDeltaT * (val_new - val_old);
    */
    // workaround way3 (use nvcc option -fmad=false)
    output[index] += rDeltaT * (rho[index] * vf[index] - rho_old[index] * vf_old[index]) * sign;
}

__global__ void fvc_grad_vector_internal(int num_surfaces, 
        const int *lower_index, const int *upper_index, const double *face_vector,
        const double *weight, const double *field_vector, 
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double w = weight[index];
    double Sfx = face_vector[index * 3 + 0];
    double Sfy = face_vector[index * 3 + 1];
    double Sfz = face_vector[index * 3 + 2];

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double ssfx = (w * (field_vector[owner * 3 + 0] - field_vector[neighbor * 3 + 0]) + field_vector[neighbor * 3 + 0]);
    double ssfy = (w * (field_vector[owner * 3 + 1] - field_vector[neighbor * 3 + 1]) + field_vector[neighbor * 3 + 1]);
    double ssfz = (w * (field_vector[owner * 3 + 2] - field_vector[neighbor * 3 + 2]) + field_vector[neighbor * 3 + 2]);    

    double grad_xx = Sfx * ssfx;
    double grad_xy = Sfx * ssfy;
    double grad_xz = Sfx * ssfz;
    double grad_yx = Sfy * ssfx;
    double grad_yy = Sfy * ssfy;
    double grad_yz = Sfy * ssfz;
    double grad_zx = Sfz * ssfx;
    double grad_zy = Sfz * ssfy;
    double grad_zz = Sfz * ssfz;

    // owner
    atomicAdd(&(output[owner * 9 + 0]), grad_xx);
    atomicAdd(&(output[owner * 9 + 1]), grad_xy);
    atomicAdd(&(output[owner * 9 + 2]), grad_xz);
    atomicAdd(&(output[owner * 9 + 3]), grad_yx);
    atomicAdd(&(output[owner * 9 + 4]), grad_yy);
    atomicAdd(&(output[owner * 9 + 5]), grad_yz);
    atomicAdd(&(output[owner * 9 + 6]), grad_zx);
    atomicAdd(&(output[owner * 9 + 7]), grad_zy);
    atomicAdd(&(output[owner * 9 + 8]), grad_zz);

    // neighbour
    atomicAdd(&(output[neighbor * 9 + 0]), -grad_xx);
    atomicAdd(&(output[neighbor * 9 + 1]), -grad_xy);
    atomicAdd(&(output[neighbor * 9 + 2]), -grad_xz);
    atomicAdd(&(output[neighbor * 9 + 3]), -grad_yx);
    atomicAdd(&(output[neighbor * 9 + 4]), -grad_yy);
    atomicAdd(&(output[neighbor * 9 + 5]), -grad_yz);
    atomicAdd(&(output[neighbor * 9 + 6]), -grad_zx);
    atomicAdd(&(output[neighbor * 9 + 7]), -grad_zy);
    atomicAdd(&(output[neighbor * 9 + 8]), -grad_zz);
}

// update boundary of interpolation field
// calculate the grad field
// TODO: this function is implemented for uncoupled boundary conditions
//       so it should use the more specific func name
__global__ void fvc_grad_vector_boundary(int num, int offset, const int *face2Cells,
        const double *boundary_face_vector, const double *boundary_field_vector, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    double bouSfx = boundary_face_vector[start_index * 3 + 0];
    double bouSfy = boundary_face_vector[start_index * 3 + 1];
    double bouSfz = boundary_face_vector[start_index * 3 + 2];

    double boussfx = boundary_field_vector[start_index * 3 + 0];
    double boussfy = boundary_field_vector[start_index * 3 + 1];
    double boussfz = boundary_field_vector[start_index * 3 + 2];

    int cellIndex = face2Cells[start_index];

    double grad_xx = bouSfx * boussfx;
    double grad_xy = bouSfx * boussfy;
    double grad_xz = bouSfx * boussfz;
    double grad_yx = bouSfy * boussfx;
    double grad_yy = bouSfy * boussfy;
    double grad_yz = bouSfy * boussfz;
    double grad_zx = bouSfz * boussfx;
    double grad_zy = bouSfz * boussfy;
    double grad_zz = bouSfz * boussfz;

    atomicAdd(&(output[cellIndex * 9 + 0]), grad_xx);
    atomicAdd(&(output[cellIndex * 9 + 1]), grad_xy);
    atomicAdd(&(output[cellIndex * 9 + 2]), grad_xz);
    atomicAdd(&(output[cellIndex * 9 + 3]), grad_yx);
    atomicAdd(&(output[cellIndex * 9 + 4]), grad_yy);
    atomicAdd(&(output[cellIndex * 9 + 5]), grad_yz);
    atomicAdd(&(output[cellIndex * 9 + 6]), grad_zx);
    atomicAdd(&(output[cellIndex * 9 + 7]), grad_zy);
    atomicAdd(&(output[cellIndex * 9 + 8]), grad_zz);
}

__global__ void fvc_grad_scalar_internal(int num_cells, int num_surfaces,
        const int *lower_index, const int *upper_index, const double *face_vector, 
        const double *weight, const double *vf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double w = weight[index];
    double Sfx = face_vector[index * 3 + 0];
    double Sfy = face_vector[index * 3 + 1];
    double Sfz = face_vector[index * 3 + 2];

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double ssf = (w * (vf[owner] - vf[neighbor]) + vf[neighbor]);

    double grad_x = Sfx * ssf;
    double grad_y = Sfy * ssf;
    double grad_z = Sfz * ssf;

    // // owner
    // atomicAdd(&(output[num_cells * 0 + owner]), grad_x);
    // atomicAdd(&(output[num_cells * 1 + owner]), grad_y);
    // atomicAdd(&(output[num_cells * 2 + owner]), grad_z);

    // // neighbour
    // atomicAdd(&(output[num_cells * 0 + neighbor]), -grad_x);
    // atomicAdd(&(output[num_cells * 1 + neighbor]), -grad_y);
    // atomicAdd(&(output[num_cells * 2 + neighbor]), -grad_z);

    // owner
    atomicAdd(&(output[owner * 3 + 0]), grad_x);
    atomicAdd(&(output[owner * 3 + 1]), grad_y);
    atomicAdd(&(output[owner * 3 + 2]), grad_z);

    // neighbour
    atomicAdd(&(output[neighbor * 3 + 0]), -grad_x);
    atomicAdd(&(output[neighbor * 3 + 1]), -grad_y);
    atomicAdd(&(output[neighbor * 3 + 2]), -grad_z);
    
}

__global__ void fvc_grad_scalar_boundary(int num_cells, int num, int offset, const int *face2Cells,
        const double *boundary_face_vector, const double *boundary_vf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    double bouvf = boundary_vf[start_index];
    double bouSfx = boundary_face_vector[start_index * 3 + 0];
    double bouSfy = boundary_face_vector[start_index * 3 + 1];
    double bouSfz = boundary_face_vector[start_index * 3 + 2];

    int cellIndex = face2Cells[start_index];

    double grad_x = bouSfx * bouvf;
    double grad_y = bouSfy * bouvf;
    double grad_z = bouSfz * bouvf;

    atomicAdd(&(output[cellIndex * 3 + 0]), grad_x);
    atomicAdd(&(output[cellIndex * 3 + 1]), grad_y);
    atomicAdd(&(output[cellIndex * 3 + 2]), grad_z);

    // if (cellIndex == 5)
    // {
    //     printf("Sfx = %.10e, ssf = %.10e\n", bouSfx, bouvf);
    //     printf("gradx = %.10e, output = %.10e\n\n", grad_x, output[5]);
    // }
}

__global__ void divide_cell_volume_tsr(int num_cells, const double* volume, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];
    output[index * 9 + 0] = output[index * 9 + 0] / vol * sign;
    output[index * 9 + 1] = output[index * 9 + 1] / vol * sign;
    output[index * 9 + 2] = output[index * 9 + 2] / vol * sign;
    output[index * 9 + 3] = output[index * 9 + 3] / vol * sign;
    output[index * 9 + 4] = output[index * 9 + 4] / vol * sign;
    output[index * 9 + 5] = output[index * 9 + 5] / vol * sign;
    output[index * 9 + 6] = output[index * 9 + 6] / vol * sign;
    output[index * 9 + 7] = output[index * 9 + 7] / vol * sign;
    output[index * 9 + 8] = output[index * 9 + 8] / vol * sign;
}

__global__ void divide_cell_volume_vec(int num_cells, const double* volume, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];

    output[index * 3 + 0] = output[index * 3 + 0] / vol * sign;
    output[index * 3 + 1] = output[index * 3 + 1] / vol * sign;
    output[index * 3 + 2] = output[index * 3 + 2] / vol * sign;
}

__global__ void divide_cell_volume_scalar(int num_cells, const double* volume, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];

    output[index] = output[index] / vol * sign;
}

__global__ void fvc_grad_vector_correctBC_zeroGradient(int num, int offset, const int *face2Cells, 
        const double *internal_grad, const double *vf, const double *boundary_sf,
        const double *boundary_mag_sf, double *boundary_grad, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    int cellIndex = face2Cells[start_index];

    double grad_xx = internal_grad[cellIndex * 9 + 0];
    double grad_xy = internal_grad[cellIndex * 9 + 1];
    double grad_xz = internal_grad[cellIndex * 9 + 2];
    double grad_yx = internal_grad[cellIndex * 9 + 3];
    double grad_yy = internal_grad[cellIndex * 9 + 4];
    double grad_yz = internal_grad[cellIndex * 9 + 5];
    double grad_zx = internal_grad[cellIndex * 9 + 6];
    double grad_zy = internal_grad[cellIndex * 9 + 7];
    double grad_zz = internal_grad[cellIndex * 9 + 8];

    double vfx = vf[cellIndex * 3 + 0];
    double vfy = vf[cellIndex * 3 + 1];
    double vfz = vf[cellIndex * 3 + 2];

    double n_x = boundary_sf[cellIndex * 3 + 0] / boundary_mag_sf[cellIndex];
    double n_y = boundary_sf[cellIndex * 3 + 1] / boundary_mag_sf[cellIndex];
    double n_z = boundary_sf[cellIndex * 3 + 2] / boundary_mag_sf[cellIndex];
    
    double grad_correction_x = - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx); // sn_grad_x = 0
    double grad_correction_y = - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
    double grad_correction_z = - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);

    boundary_grad[start_index * 9 + 0] = (grad_xx + n_x * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 1] = (grad_xy + n_x * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 2] = (grad_xz + n_x * grad_correction_z) * sign;
    boundary_grad[start_index * 9 + 3] = (grad_yx + n_y * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 4] = (grad_yy + n_y * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 5] = (grad_yz + n_y * grad_correction_z) * sign;
    boundary_grad[start_index * 9 + 6] = (grad_zx + n_z * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 7] = (grad_zy + n_z * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 8] = (grad_zz + n_z * grad_correction_z) * sign;
}

__global__ void fvc_grad_vector_correctBC_fixedValue(int num, int offset, const int *face2Cells, 
        const double *internal_grad, const double *vf, const double *boundary_sf,
        const double *boundary_mag_sf, double *boundary_grad,
        const double *boundary_deltaCoeffs, const double *boundary_vf, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    int cellIndex = face2Cells[start_index];

    double grad_xx = internal_grad[cellIndex * 9 + 0];
    double grad_xy = internal_grad[cellIndex * 9 + 1];
    double grad_xz = internal_grad[cellIndex * 9 + 2];
    double grad_yx = internal_grad[cellIndex * 9 + 3];
    double grad_yy = internal_grad[cellIndex * 9 + 4];
    double grad_yz = internal_grad[cellIndex * 9 + 5];
    double grad_zx = internal_grad[cellIndex * 9 + 6];
    double grad_zy = internal_grad[cellIndex * 9 + 7];
    double grad_zz = internal_grad[cellIndex * 9 + 8];

    double vfx = vf[cellIndex * 3 + 0];
    double vfy = vf[cellIndex * 3 + 1];
    double vfz = vf[cellIndex * 3 + 2];

    double n_x = boundary_sf[start_index * 3 + 0] / boundary_mag_sf[start_index];
    double n_y = boundary_sf[start_index * 3 + 1] / boundary_mag_sf[start_index];
    double n_z = boundary_sf[start_index * 3 + 2] / boundary_mag_sf[start_index];
    
    // sn_grad: solving according to fixedValue BC
    double sn_grad_x = boundary_deltaCoeffs[start_index] * (boundary_vf[start_index * 3 + 0] - vf[cellIndex * 3 + 0]);
    double sn_grad_y = boundary_deltaCoeffs[start_index] * (boundary_vf[start_index * 3 + 1] - vf[cellIndex * 3 + 1]);
    double sn_grad_z = boundary_deltaCoeffs[start_index] * (boundary_vf[start_index * 3 + 2] - vf[cellIndex * 3 + 2]);

    double grad_correction_x = sn_grad_x - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx); // sn_grad_x = 0
    double grad_correction_y = sn_grad_y - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
    double grad_correction_z = sn_grad_z - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);

    boundary_grad[start_index * 9 + 0] = (grad_xx + n_x * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 1] = (grad_xy + n_x * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 2] = (grad_xz + n_x * grad_correction_z) * sign;
    boundary_grad[start_index * 9 + 3] = (grad_yx + n_y * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 4] = (grad_yy + n_y * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 5] = (grad_yz + n_y * grad_correction_z) * sign;
    boundary_grad[start_index * 9 + 6] = (grad_zx + n_z * grad_correction_x) * sign;
    boundary_grad[start_index * 9 + 7] = (grad_zy + n_z * grad_correction_y) * sign;
    boundary_grad[start_index * 9 + 8] = (grad_zz + n_z * grad_correction_z) * sign;
}

__global__ void fvc_div_surface_scalar_internal(int num_surfaces, 
        const int *lower_index, const int *upper_index, const double *ssf,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double issf = ssf[index];

    // owner
    atomicAdd(&(output[owner]), issf);

    // neighbor
    atomicAdd(&(output[neighbor]), -issf);
}

__global__ void fvc_div_surface_scalar_boundary(int num_boundary_face, const int *face2Cells,
        const double *boundary_ssf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_face)
        return;
    
    int cellIndex = face2Cells[index];

    atomicAdd(&(output[cellIndex]), boundary_ssf[index]);
}

__global__ void fvc_div_cell_vector_internal(int num_surfaces, 
        const int *lower_index, const int *upper_index,
        const double *field_vector, const double *weight, const double *face_vector,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double Sfx = face_vector[index * 3 + 0];
    double Sfy = face_vector[index * 3 + 1];
    double Sfz = face_vector[index * 3 + 2];

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double ssfx = (w * (field_vector[owner * 3 + 0] - field_vector[neighbor * 3 + 0]) + field_vector[neighbor * 3 + 0]);
    double ssfy = (w * (field_vector[owner * 3 + 1] - field_vector[neighbor * 3 + 1]) + field_vector[neighbor * 3 + 1]);
    double ssfz = (w * (field_vector[owner * 3 + 2] - field_vector[neighbor * 3 + 2]) + field_vector[neighbor * 3 + 2]);

    double div = Sfx * ssfx + Sfy * ssfy + Sfz * ssfz;

    // owner
    atomicAdd(&(output[owner]), div);

    // neighbour
    atomicAdd(&(output[neighbor]), -div);
}

__global__ void fvc_div_cell_vector_boundary(int num, int offset, const int *face2Cells,
        const double *boundary_face_vector, const double *boundary_field_vector, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    double bouSfx = boundary_face_vector[start_index * 3 + 0];
    double bouSfy = boundary_face_vector[start_index * 3 + 1];
    double bouSfz = boundary_face_vector[start_index * 3 + 2];

    double boussfx = boundary_field_vector[start_index * 3 + 0];
    double boussfy = boundary_field_vector[start_index * 3 + 1];
    double boussfz = boundary_field_vector[start_index * 3 + 2];

    int cellIndex = face2Cells[start_index];

    double bouDiv = bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz;

    atomicAdd(&(output[cellIndex]), bouDiv);

}

__global__ void fvc_div_cell_tensor_internal(int num_surfaces,
        const int *lower_index, const int *upper_index,
        const double *vf, const double *weight, const double *face_vector,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    double w = weight[index];
    double Sfx = face_vector[index * 3 + 0];
    double Sfy = face_vector[index * 3 + 1];
    double Sfz = face_vector[index * 3 + 2];
    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double ssf_xx = (w * (vf[owner * 9 + 0] - vf[neighbor * 9 + 0]) + vf[neighbor * 9 + 0]);
    double ssf_xy = (w * (vf[owner * 9 + 1] - vf[neighbor * 9 + 1]) + vf[neighbor * 9 + 1]);
    double ssf_xz = (w * (vf[owner * 9 + 2] - vf[neighbor * 9 + 2]) + vf[neighbor * 9 + 2]);
    double ssf_yx = (w * (vf[owner * 9 + 3] - vf[neighbor * 9 + 3]) + vf[neighbor * 9 + 3]);
    double ssf_yy = (w * (vf[owner * 9 + 4] - vf[neighbor * 9 + 4]) + vf[neighbor * 9 + 4]);
    double ssf_yz = (w * (vf[owner * 9 + 5] - vf[neighbor * 9 + 5]) + vf[neighbor * 9 + 5]);
    double ssf_zx = (w * (vf[owner * 9 + 6] - vf[neighbor * 9 + 6]) + vf[neighbor * 9 + 6]);
    double ssf_zy = (w * (vf[owner * 9 + 7] - vf[neighbor * 9 + 7]) + vf[neighbor * 9 + 7]);
    double ssf_zz = (w * (vf[owner * 9 + 8] - vf[neighbor * 9 + 8]) + vf[neighbor * 9 + 8]);
    double div_x = Sfx * ssf_xx + Sfy * ssf_xy + Sfz * ssf_xz;
    double div_y = Sfx * ssf_yx + Sfy * ssf_yy + Sfz * ssf_yz;
    double div_z = Sfx * ssf_zx + Sfy * ssf_zy + Sfz * ssf_zz;

    // owner
    atomicAdd(&(output[owner * 3 + 0]), div_x);
    atomicAdd(&(output[owner * 3 + 1]), div_y);
    atomicAdd(&(output[owner * 3 + 2]), div_z);

    // neighbour
    atomicAdd(&(output[neighbor * 3 + 0]), -div_x);
    atomicAdd(&(output[neighbor * 3 + 1]), -div_y);
    atomicAdd(&(output[neighbor * 3 + 2]), -div_z);
}

__global__ void fvc_div_cell_tensor_boundary(int num, int offset, const int *face2Cells,
        const double *boundary_face_vector, const double *boundary_vf, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    double bouSfx = boundary_face_vector[start_index * 3 + 0];
    double bouSfy = boundary_face_vector[start_index * 3 + 1];
    double bouSfz = boundary_face_vector[start_index * 3 + 2];

    double boussf_xx = boundary_vf[start_index * 9 + 0];
    double boussf_xy = boundary_vf[start_index * 9 + 1];
    double boussf_xz = boundary_vf[start_index * 9 + 2];
    double boussf_yx = boundary_vf[start_index * 9 + 3];
    double boussf_yy = boundary_vf[start_index * 9 + 4];
    double boussf_yz = boundary_vf[start_index * 9 + 5];
    double boussf_zx = boundary_vf[start_index * 9 + 6];
    double boussf_zy = boundary_vf[start_index * 9 + 7];
    double boussf_zz = boundary_vf[start_index * 9 + 8];
    int cellIndex = face2Cells[start_index];

    double bouDiv_x = bouSfx * boussf_xx + bouSfy * boussf_xy + bouSfz * boussf_xz;
    double bouDiv_y = bouSfx * boussf_yx + bouSfy * boussf_yy + bouSfz * boussf_yz;
    double bouDiv_z = bouSfx * boussf_zx + bouSfy * boussf_zy + bouSfz * boussf_zz;

    atomicAdd(&(output[cellIndex * 3 + 0]), bouDiv_x);
    atomicAdd(&(output[cellIndex * 3 + 1]), bouDiv_y);
    atomicAdd(&(output[cellIndex * 3 + 2]), bouDiv_z);
}

void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    permute_vector_d2h_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, input, output);
}

void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    permute_vector_h2d_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, input, output);
}

void field_multiply_scalar(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (std::max(num_cells, num_boundary_surfaces) + threads_per_block - 1) / threads_per_block;
    field_multiply_scalar_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_surfaces,
            input1, input2, output, boundary_input1, boundary_input2, boundary_output);
}

void fvc_to_source_vector(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source)
{
    size_t threads_per_block = 256;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_to_source_vector_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            volume, fvc_output, source);
}

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lower_to_csr_index, const int *upper_to_csr_index, const int *diag_to_csr_index,
        const double *lower, const double *upper, const double *diag, const double *source,
        const double *internal_coeffs, const double *boundary_coeffs,
        double *A, double *b)
{

}

void update_boundary_coeffs_vector(cudaStream_t stream, int num_patches,
        const int *patch_size, const int *patch_type,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        // TODO: just vector version now
        if (patch_type[i] == boundaryConditions::zeroGradient) {
            update_boundary_coeffs_zeroGradient_vector<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    value_internal_coeffs, value_boundary_coeffs, gradient_internal_coeffs, gradient_boundary_coeffs);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }
}

void fvm_ddt_vector(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvm_ddt_vector_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            rDeltaT, rho, rho_old, vf, volume, diag, source, sign);
}

void fvm_div_vector(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvm_div_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr,
            phi, weight, lower, upper, diag, sign);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvm_div_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_phi, value_internal_coeffs, value_boundary_coeffs,
                    internal_coeffs, boundary_coeffs, sign);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }
}

void fvm_laplacian_vector(cudaStream_t stream, int num_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = 1;

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvm_laplacian_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr,
            weight, mag_sf, delta_coeffs, gamma, lower, upper, diag, sign);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvm_laplacian_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_mag_sf, boundary_gamma, gradient_internal_coeffs, gradient_boundary_coeffs,
                    internal_coeffs, boundary_coeffs, sign);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }
}

void fvc_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *vf_old,
        double *output, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * sizeof(double), stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    fvc_ddt_scalar_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
            rDeltaT, rho, rho_old, vf, vf_old, output, sign);
}

void fvc_grad_vector(cudaStream_t stream, int num_cells, int num_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const double *boundary_deltaCoeffs, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * 9 * sizeof(double), stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_grad_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr,
            Sf, weight, vf, output);
    
    int offset = 0;
    // finish conctruct grad field except dividing cell volume
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvc_grad_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset, boundary_cell_face,
                    boundary_Sf, boundary_vf, output);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }

    // divide cell volume
    threads_per_block = 512;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_tsr<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);

    // correct boundary conditions
    offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient) {
            // TODO: just vector version now
            fvc_grad_vector_correctBC_zeroGradient<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset, boundary_cell_face,
                    output, boundary_vf, boundary_Sf, boundary_mag_Sf, boundary_output, sign);
        } else if (patch_type[i] == boundaryConditions::fixedValue) {
            fvc_grad_vector_correctBC_fixedValue<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset, boundary_cell_face,
                    output, boundary_vf, boundary_Sf, boundary_mag_Sf, boundary_output, boundary_deltaCoeffs, boundary_vf, sign);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }
}

void scale_dev2T_tensor(cudaStream_t stream, int num_cells, const double *vf1, double *vf2,
        int num_boundary_surfaces, const double *boundary_vf1, double *boundary_vf2)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    scale_dev2t_tensor_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, vf1, vf2);

    blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    scale_dev2t_tensor_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, boundary_vf1, boundary_vf2);
}

void fvc_div_surface_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, const double *ssf, const int *boundary_cell_face,
        const double *boundary_ssf, const double *volume, double *output, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * sizeof(double), stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_div_surface_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr, ssf, output);

    threads_per_block = 1024;
    blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_div_surface_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, boundary_cell_face, 
            boundary_ssf, output);

    // divide cell volume
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);
}

void fvc_div_cell_vector(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * sizeof(double), stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_div_cell_vector_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr, vf, weight, Sf, output);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvc_div_cell_vector_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset, boundary_cell_face,
                    boundary_Sf, boundary_vf, output);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }

    // divide cell volume
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_scalar<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);
}

void fvc_div_cell_tensor(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * 3 * sizeof(double), stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_div_cell_tensor_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces, lowerAddr, upperAddr, vf, weight, Sf, output);

    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just basic patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            // TODO: just vector version now
            fvc_div_cell_tensor_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset, boundary_cell_face,
                    boundary_Sf, boundary_vf, output);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }

    // divide cell volume
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_vec<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);
}

void fvc_grad_cell_scalar(cudaStream_t stream, int num_cells, int num_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *volume, double sign)
{
    checkCudaErrors(cudaMemsetAsync(output, 0, num_cells * 3 * sizeof(double), stream));
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_grad_scalar_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces, lowerAddr, upperAddr,
            Sf, weight, vf, output);
    
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just non-coupled patch type now
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue) {
            fvc_grad_scalar_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, patch_size[i], offset, boundary_cell_face,
                    boundary_Sf, boundary_vf, output);
        } else if (0) {
            // xxx
            fprintf(stderr, "boundaryConditions other than zeroGradient are not support yet!\n");
        }
        offset += patch_size[i];
    }

    // divide cell volume
    threads_per_block = 1024;
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_vec<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, output, sign);
}

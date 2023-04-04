#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
        int const line) {
  if (result) {
    fprintf(stderr, "cuda error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static void checkVectorEqual(int count, double* basevec, double* vec, double max_relative_error) {
  for (size_t i = 0; i < count; ++i)
  {
    double abs_diff = fabs(basevec[i] - vec[i]);
    double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
    if (abs_diff > 1e-16 && rel_diff > max_relative_error) {
      fprintf(stderr, "mismatch index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }
  }
}

__global__ void fvm_ddt(int num_cells, int num_faces, const double rdelta_t,
        const int* csr_row_index, const int* csr_diag_index,
        const double* rho_old, const double* rho_new, const double* volume, const double* velocity_old,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int diag_index = csr_diag_index[index];

    int csr_dim = num_cells + num_faces;
    int csr_index = row_index + diag_index;
    double ddt_diag = rdelta_t * rho_new[index] * volume[index];
    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + ddt_diag;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + ddt_diag;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + ddt_diag;

    double ddt_part_term = rdelta_t * rho_old[index] * volume[index];
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + ddt_part_term * velocity_old[index * 3 + 0];
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + ddt_part_term * velocity_old[index * 3 + 1];
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + ddt_part_term * velocity_old[index * 3 + 2];
}

__global__ void fvm_div_internal(int num_cells, int num_faces,
        const int* csr_row_index, const int* csr_diag_index,
        const double* weight, const double* phi,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;

    double div_diag = 0;
    for (int i = row_index; i < next_row_index; i++) {
      int inner_index = i - row_index;
      // lower
      if (inner_index < diag_index) {
        int neighbor_index = neighbor_offset + inner_index;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + (-w) * f;
        A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + (-w) * f;
        A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + (-w) * f;
        // lower neighbors contribute to sum of -1
        div_diag += (w - 1) * f;
      }
      // upper
      if (inner_index > diag_index) {
        // upper, index - 1, consider of diag
        int neighbor_index = neighbor_offset + inner_index - 1;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + (1 - w) * f;
        A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + (1 - w) * f;
        A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + (1 - w) * f;
        // upper neighbors contribute to sum of 1
        div_diag += w * f;
      }
    }
    A_csr_output[csr_dim * 0 + row_index + diag_index] = A_csr_input[csr_dim * 0 + row_index + diag_index] + div_diag; // diag
    A_csr_output[csr_dim * 1 + row_index + diag_index] = A_csr_input[csr_dim * 1 + row_index + diag_index] + div_diag; // diag
    A_csr_output[csr_dim * 2 + row_index + diag_index] = A_csr_input[csr_dim * 2 + row_index + diag_index] + div_diag; // diag
}

__global__ void fvm_div_boundary(int num_cells, int num_faces, int num_boundary_cells,
        const int* csr_row_index, const int* csr_diag_index,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* internal_coeffs, const double* boundary_coeffs,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index]; // face index
    int cell_index = boundary_cell_id[cell_offset];
    int loop_size = boundary_cell_offset[index + 1] - cell_offset;

    int row_index = csr_row_index[cell_index];
    int diag_index = csr_diag_index[cell_index];
    int csr_dim = num_cells + num_faces;
    int csr_index = row_index + diag_index;

    // construct internalCoeffs & boundaryCoeffs
    double internal_coeffs_x = 0;
    double internal_coeffs_y = 0;
    double internal_coeffs_z = 0;
    double boundary_coeffs_x = 0;
    double boundary_coeffs_y = 0;
    double boundary_coeffs_z = 0;
    for (int i = 0; i < loop_size; i++) {
        internal_coeffs_x += internal_coeffs[(cell_offset + i) * 3 + 0];
        internal_coeffs_y += internal_coeffs[(cell_offset + i) * 3 + 1];
        internal_coeffs_z += internal_coeffs[(cell_offset + i) * 3 + 2];
        boundary_coeffs_x += boundary_coeffs[(cell_offset + i) * 3 + 0];
        boundary_coeffs_y += boundary_coeffs[(cell_offset + i) * 3 + 1];
        boundary_coeffs_z += boundary_coeffs[(cell_offset + i) * 3 + 2];
    }

    A_csr_output[csr_dim * 0 + csr_index] = A_csr_input[csr_dim * 0 + csr_index] + internal_coeffs_x;
    A_csr_output[csr_dim * 1 + csr_index] = A_csr_input[csr_dim * 1 + csr_index] + internal_coeffs_y;
    A_csr_output[csr_dim * 2 + csr_index] = A_csr_input[csr_dim * 2 + csr_index] + internal_coeffs_z;
    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + boundary_coeffs_x;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + boundary_coeffs_y;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + boundary_coeffs_z;
}

__global__ void fvc_grad_internal_face(int num_cells,
        const int* csr_row_index, const int* csr_col_index, const int* csr_diag_index,
        const double* face_vector, const double* weight, const double* pressure,
        const double* b_input, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

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
    for (int i = row_index; i < next_row_index; i++) {
      int inner_index = i - row_index;
      // lower
      if (inner_index < diag_index) {
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
      if (inner_index > diag_index) {
        int neighbor_index = neighbor_offset + inner_index - 1;
        double w = weight[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        int neighbor_cell_id = csr_col_index[row_index + inner_index + 1];
        double neighbor_cell_p = pressure[neighbor_cell_id];
        double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
        grad_bx_upp += face_p * sfx;
        grad_by_upp += face_p * sfy;
        grad_bz_upp += face_p * sfz;
      }
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + grad_bx_low + grad_bx_upp;
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + grad_by_low + grad_by_upp;
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + grad_bz_low + grad_bz_upp;
}

__global__ void fvc_grad_boundary_face(int num_cells, int num_boundary_cells,
        const int* boundary_cell_offset, const int* boundary_cell_id, 
        const double* boundary_face_vector, const double* boundary_pressure,
        const double* b_input, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_cells) return;

    int cell_offset = boundary_cell_offset[index];
    int next_cell_offset = boundary_cell_offset[index + 1];
    int cell_index = boundary_cell_id[cell_offset];

    // compute boundary gradient
    double grad_bx = 0; 
    double grad_by = 0; 
    double grad_bz = 0; 
    for (int i = cell_offset; i < next_cell_offset; i++) {
      double sfx = boundary_face_vector[i * 3 + 0];
      double sfy = boundary_face_vector[i * 3 + 1];
      double sfz = boundary_face_vector[i * 3 + 2];
      double face_p = boundary_pressure[i];
      grad_bx += face_p * sfx;
      grad_by += face_p * sfy;
      grad_bz += face_p * sfz;
    }

    //// correct the boundary gradient
    //double nx = boundary_face_vector[face_index * 3 + 0] / magSf[face_index];
    //double ny = boundary_face_vector[face_index * 3 + 1] / magSf[face_index];
    //double nz = boundary_face_vector[face_index * 3 + 2] / magSf[face_index];
    //double sn_grad = 0;
    //double grad_correction = sn_grad * volume[cell_index] - (nx * grad_bx + ny * grad_by + nz * grad_bz);
    //grad_bx += nx * grad_correction; 
    //grad_by += ny * grad_correction; 
    //grad_bz += nz * grad_correction; 

    b_output[num_cells * 0 + cell_index] = b_input[num_cells * 0 + cell_index] + grad_bx;
    b_output[num_cells * 1 + cell_index] = b_input[num_cells * 1 + cell_index] + grad_by;
    b_output[num_cells * 2 + cell_index] = b_input[num_cells * 2 + cell_index] + grad_bz;
}

__global__ void add_fvMatrix(int num_cells, int num_faces,
        const int* csr_row_index,
        const double* turbSrc_A, const double* turbSrc_b,
        const double* A_csr_input, const double* b_input, double* A_csr_output, double* b_output) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int csr_dim = num_cells + num_faces;
    double A_entry;

    for (int i = row_index; i < next_row_index; i++)
    {
      A_entry = turbSrc_A[i];
      A_csr_output[csr_dim * 0 + i] = A_csr_input[csr_dim * 0 + i] + A_entry;
      A_csr_output[csr_dim * 1 + i] = A_csr_input[csr_dim * 1 + i] + A_entry;
      A_csr_output[csr_dim * 2 + i] = A_csr_input[csr_dim * 2 + i] + A_entry;
    }
    b_output[num_cells * 0 + index] = b_input[num_cells * 0 + index] + turbSrc_b[num_cells * 0 + index];
    b_output[num_cells * 1 + index] = b_input[num_cells * 1 + index] + turbSrc_b[num_cells * 1 + index];
    b_output[num_cells * 2 + index] = b_input[num_cells * 2 + index] + turbSrc_b[num_cells * 2 + index];
}

/*
// the implement fo assemble kernel is outdated.
__global__ void assemble(int num_cells, int num_faces, const double rdelta_t, const int* csr_row_index, const int* csr_diag_index,
        const double* rho_old, const double* rho_new, const double* volume, const double* velocity_old,
        const double* weight, const double* phi, const double* face_vector, const double* pressure,
        double* A_csr, double* b) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells) return;

    // A_csr has one more element in each row: itself
    int row_index = csr_row_index[index];
    int next_row_index = csr_row_index[index + 1];
    int diag_index = csr_diag_index[index];
    int neighbor_offset = csr_row_index[index] - index;
    int csr_dim = num_cells + num_faces;
 
    double div_diag = 0;
    double grad_bx = 0;
    double grad_by = 0;
    double grad_bz = 0;
    for (int i = row_index; i < next_row_index; i++) {
      int inner_index = i - row_index;
      // lower
      if (inner_index < diag_index) {
        int neighbor_index = neighbor_offset + inner_index;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        double p = pressure[neighbor_index];
        A_csr[csr_dim * 0 + i] += (-w) * f;
        A_csr[csr_dim * 1 + i] += (-w) * f;
        A_csr[csr_dim * 2 + i] += (-w) * f;
        // lower neighbors contribute to sum of -1
        div_diag += (w - 1) * f;
        grad_bx -= p * sfx;
        grad_by -= p * sfy;
        grad_bz -= p * sfz;
      }
      // upper
      if (inner_index > diag_index) {
        // upper, index - 1, consider of diag
        int neighbor_index = neighbor_offset + inner_index - 1;
        double w = weight[neighbor_index];
        double f = phi[neighbor_index];
        double sfx = face_vector[neighbor_index * 3 + 0];
        double sfy = face_vector[neighbor_index * 3 + 1];
        double sfz = face_vector[neighbor_index * 3 + 2];
        double p = pressure[neighbor_index];
        A_csr[csr_dim * 0 + i] += (1 - w) * f;
        A_csr[csr_dim * 1 + i] += (1 - w) * f;
        A_csr[csr_dim * 2 + i] += (1 - w) * f;
        // upper neighbors contribute to sum of 1
        div_diag += w * f;
        grad_bx += p * sfx;
        grad_by += p * sfy;
        grad_bz += p * sfz;
      }
    }

    double ddt_diag = rdelta_t * rho_new[index] * volume[index];
    A_csr[csr_dim * 0 + row_index + diag_index] = ddt_diag + div_diag; // diag of A
    A_csr[csr_dim * 1 + row_index + diag_index] = ddt_diag + div_diag; // diag of A
    A_csr[csr_dim * 2 + row_index + diag_index] = ddt_diag + div_diag; // diag of A

    double ddt_bx = rdelta_t * rho_old[index] * velocity_old[index * 3 + 0] * volume[index];
    double ddt_by = rdelta_t * rho_old[index] * velocity_old[index * 3 + 1] * volume[index];
    double ddt_bz = rdelta_t * rho_old[index] * velocity_old[index * 3 + 2] * volume[index];
    b[num_cells * 0 + index] = ddt_bx + grad_bx;
    b[num_cells * 1 + index] = ddt_by + grad_by;
    b[num_cells * 2 + index] = ddt_bz + grad_bz;
}
*/
int main(int argc, char* argv[]) {
  // Print the array length to be used, and compute its size
  bool validation_mode = false;
  int w = 128;
  int h = 128;
  int d = 128;
  char* input_file = NULL;

  if (argc > 1) {
    validation_mode = true;
    w = atoi(argv[1]);
    h = atoi(argv[2]);
    d = atoi(argv[3]);
    input_file = argv[4];
  }

  fprintf(stderr, "mesh(w, h, d) is (%d, %d, %d)\n", w, h, d);

  // Get num_cells, num_faces, num_boundary_cells, num_boundary_faces
  int num_cells = w * h * d;
  int num_faces = num_cells * 6;
  int num_boundary_cells = 0;
  int num_boundary_faces = 0;
  for (int i = 0; i < num_cells; ++i) {
    int ww = i % w;
    int hh = (i % (w * h)) / w;
    int dd = i / (w * h);

    bool boundary_flag = false;
    if (ww == 0) {
      num_boundary_faces++;
      boundary_flag = true;
    }
    if (ww == (w - 1)) {
      num_boundary_faces++;
      boundary_flag = true;
    }
    if (hh == 0) {
      num_boundary_faces++;
      boundary_flag = true;
    }
    if (hh == (h - 1)) {
      num_boundary_faces++;
      boundary_flag = true;
    }
    if (dd == 0) {
      num_boundary_faces++;
      boundary_flag = true;
    }
    if (dd == (d - 1)) {
      num_boundary_faces++;
      boundary_flag = true;
    }

    if (boundary_flag)
        num_boundary_cells++;
  }
  num_faces -= num_boundary_faces;

  // Load num_cells, num_faces, num_boundary_cells and num_boundary_faces
  FILE *fp = NULL;
  if (validation_mode) {
    fp = fopen(input_file, "rb+");
    if (fp == NULL) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }

    int readfile = 0;
    int input_num_cells = 0;
    int input_num_faces = 0;
    int input_num_boundary_cells = 0;
    int input_num_boundary_faces = 0;
    readfile = fread(&input_num_cells, sizeof(int), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    readfile = fread(&input_num_faces, sizeof(int), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    readfile = fread(&input_num_boundary_cells, sizeof(int), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    readfile = fread(&input_num_boundary_faces, sizeof(int), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //if ((input_num_cells != num_cells) || (input_num_faces != num_faces) || (input_num_boundary_cells != num_boundary_cells)) {
    // temp
    if ((input_num_cells != num_cells) || (input_num_faces * 2 != num_faces
                || (input_num_boundary_cells != num_boundary_cells) || (input_num_boundary_faces != num_boundary_faces))) {
      fprintf(stderr, "Input info mismatched:\n");
      fprintf(stderr, "input_num_cells: %d, num_cells: %d\n", input_num_cells, num_cells);
      fprintf(stderr, "input_num_faces: %d, num_faces: %d\n", input_num_faces, num_faces);
      fprintf(stderr, "input_num_boundary_cells: %d, num_boundary_cells: %d\n", input_num_boundary_cells, num_boundary_cells);
      fprintf(stderr, "input_num_boundary_faces: %d, num_boundary_faces: %d\n", input_num_boundary_faces, num_boundary_faces);
      exit(EXIT_FAILURE);
    }
    num_faces = input_num_faces * 2;
  }

  size_t cell_bytes = num_cells * sizeof(double);
  size_t cell_vec_bytes = num_cells * 3 * sizeof(double);
  size_t cell_index_bytes = num_cells * sizeof(int);

  size_t face_bytes = num_faces * sizeof(double);
  size_t face_vec_bytes = num_faces * 3 * sizeof(double);

  size_t boundary_cell_bytes = num_boundary_cells * sizeof(double);
  size_t boundary_cell_vec_bytes = num_boundary_cells * 3 * sizeof(double);
  size_t boundary_cell_index_bytes = num_boundary_cells * sizeof(int);

  size_t boundary_face_bytes = num_boundary_faces * sizeof(double);
  size_t boundary_face_vec_bytes = num_boundary_faces * 3 * sizeof(double);
  size_t boundary_face_index_bytes = num_boundary_faces * sizeof(int);

  // A_csr has one more element in each row: itself
  size_t csr_row_index_bytes = (num_cells + 1) * sizeof(int);
  size_t csr_col_index_bytes = (num_cells + num_faces) * sizeof(int);
  size_t csr_value_bytes = (num_cells + num_faces) * sizeof(double);
  size_t csr_value_vec_bytes = (num_cells + num_faces) * 3 * sizeof(double);

  // csr_row_index, csr_col_index, csr_diag_index are pre-computed on cpu, not computed on gpu.
  // values of each row are stored continued.
  // row_index stores the start offset of each row.
  // col_index stores the column number of each row
  // diag_index stores the serial number(in a row) of the diag element.
  int* h_A_csr_row_index = (int*)malloc(csr_row_index_bytes);
  int* h_A_csr_col_index = (int*)malloc(csr_col_index_bytes);
  int* h_A_csr_diag_index = (int*)malloc(cell_index_bytes);

  double* h_rho_old = (double*)malloc(cell_bytes);
  double* h_rho_new = (double*)malloc(cell_bytes);
  double* h_volume = (double*)malloc(cell_bytes);
  double* h_pressure = (double*)malloc(cell_bytes);
  double* h_velocity_old = (double*)malloc(cell_vec_bytes);
  double* h_weight = (double*)malloc(face_bytes);
  double* h_phi = (double*)malloc(face_bytes);
  double* h_face_vector = (double*)malloc(face_vec_bytes);

  int* h_boundary_cell_offset = (int*)malloc((num_boundary_cells+1) * sizeof(int));
  int* h_boundary_cell_id = (int*)malloc(boundary_face_bytes);

  double* h_boundary_pressure = (double*)malloc(boundary_face_bytes);
  double* h_boundary_face_vector = (double*)malloc(boundary_face_vec_bytes);
  double* h_internal_coeffs = (double*)malloc(boundary_face_vec_bytes);
  double* h_boundary_coeffs = (double*)malloc(boundary_face_vec_bytes);

  double* h_A_csr = (double*)malloc(csr_value_vec_bytes);
  double* h_b = (double*)malloc(cell_vec_bytes);
  double* h_A_csr_validation = (double*)malloc(csr_value_vec_bytes);
  double* h_b_validation = (double*)malloc(cell_vec_bytes);
  double* h_grad_validation = (double*)malloc(cell_vec_bytes);

  double* h_turbSrc_A = (double*)malloc(csr_value_bytes);
  double* h_turbSrc_b = (double*)malloc(cell_vec_bytes);

  double *b_openfoam = (double*)malloc(cell_vec_bytes);
  double *A_openfoam = (double*)malloc(csr_value_vec_bytes);

  // Verify that allocations succeeded
  if (h_A_csr_row_index == NULL || h_A_csr_col_index == NULL || h_A_csr_diag_index == NULL
          || h_rho_old == NULL || h_rho_new == NULL || h_volume == NULL || h_pressure == NULL || h_velocity_old == NULL
          || h_weight == NULL || h_phi == NULL || h_face_vector == NULL
          || h_boundary_cell_offset == NULL || h_boundary_cell_id == NULL
          || h_boundary_pressure == NULL || h_boundary_face_vector == NULL
          || h_internal_coeffs == NULL || h_boundary_coeffs == NULL
          || h_A_csr == NULL || h_b == NULL || h_A_csr_validation == NULL || h_b_validation == NULL) {
    fprintf(stderr, "Failed to allocate host arrays!\n");
    exit(EXIT_FAILURE);
  }

  // Load input data
  double rdelta_t = 1.0 / 0.0001;
  if (validation_mode) {
    int readfile = 0;
    readfile = fread((void*)&rdelta_t, sizeof(double), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    fprintf(stderr, "rdelta_t: %lf\n", rdelta_t);

    readfile = fread(h_A_csr_row_index, csr_row_index_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells + 1; i++)
    //   fprintf(stderr, "h_A_csr_row_index[%d]: %d\n", i, h_A_csr_row_index[i]);

    readfile = fread(h_A_csr_col_index, csr_col_index_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells + num_faces; i++)
    //   fprintf(stderr, "h_A_csr_col_index[%d]: %d\n", i, h_A_csr_col_index[i]);

    readfile = fread(h_A_csr_diag_index, cell_index_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_A_csr_diag_index[%d]: %d\n", i, h_A_csr_diag_index[i]);

    readfile = fread(h_rho_old, cell_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_rho_old[%d]: %lf\n", i, h_rho_old[i]);

    readfile = fread(h_rho_new, cell_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_rho_new[%d]: %lf\n", i, h_rho_new[i]);

    readfile = fread(h_volume, cell_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_volume[%d]: %.16lf\n", i, h_volume[i]);

    readfile = fread(h_pressure, cell_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_pressure[%d]: %.30lf\n", i, h_pressure[i]);

    readfile = fread(h_velocity_old, cell_vec_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_cells; i++)
    //  fprintf(stderr, "h_velocity[%d]: (%lf, %lf, %lf)\n", i, h_volume[i * 3 + 0], h_volume[i * 3 + 1], h_volume[i * 3 + 2]);

    readfile = fread(h_weight, face_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_faces; i++)
    //  fprintf(stderr, "h_weight[%d]: %.20lf\n", i, h_weight[i]);

    readfile = fread(h_phi, face_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_faces; i++)
    //  fprintf(stderr, "h_phi[%d]: %lf\n", i, h_phi[i]);

    readfile = fread(h_face_vector, face_vec_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_faces; i++)
    //  fprintf(stderr, "h_face_vector[%d]: (%.20lf, %.20lf, %.20lf)\n", i, h_face_vector[i * 3 + 0], h_face_vector[i * 3 + 1], h_face_vector[i * 3 + 2]);

    readfile = fread(h_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int), 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_boundary_cells; i++)
    //  fprintf(stderr, "h_boundary_cell_offset[%d]: %d\n", i, h_boundary_cell_offset[i]);

    readfile = fread(h_boundary_cell_id, boundary_face_index_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_boundary_faces; i++)
    //  fprintf(stderr, "h_boundary_cell_id[%d]: %d\n", i, h_boundary_cell_id[i]);

    readfile = fread(h_boundary_pressure, boundary_face_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_boundary_faces; i++)
    //  fprintf(stderr, "h_boundary_pressure[%d]: %lf\n", i, h_boundary_pressure[i]);

    readfile = fread(h_boundary_face_vector, boundary_face_vec_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_boundary_faces; i++)
    //  fprintf(stderr, "h_boundary_face_vector[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_boundary_face_vector[i * 3 + 0], h_boundary_face_vector[i * 3 + 1], h_boundary_face_vector[i * 3 + 2]);

    readfile = fread(h_internal_coeffs, boundary_face_vec_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_boundary_faces; i++)
    //  fprintf(stderr, "h_internal_coeffs[%d]: (%.15lf, %.15lf, %.15lf)\n", i, h_internal_coeffs[i * 3 + 0], h_internal_coeffs[i * 3 + 1], h_internal_coeffs[i * 3 + 2]);

    readfile = fread(h_boundary_coeffs, boundary_face_vec_bytes, 1, fp);
    if(readfile == 0) {
      fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
      exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < num_boundary_faces; i++)
    //  fprintf(stderr, "h_boundary_coeffs[%d]: (%lf, %lf, %lf)\n", i, h_boundary_coeffs[i * 3 + 0], h_boundary_coeffs[i * 3 + 1], h_boundary_coeffs[i * 3 + 2]);

    readfile = fread(h_A_csr_validation, csr_value_vec_bytes, 1, fp);
    if(readfile == 0) {
     fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
     exit(EXIT_FAILURE);
    }
    //for (int i = 0; i < (num_cells + num_faces) * 3; i++)
    //  fprintf(stderr, "h_A_csr_validation[%d]: %.30lf\n", i, h_A_csr_validation[i]);

    readfile = fread(h_b_validation, cell_vec_bytes, 1, fp);
    if(readfile == 0) {
     fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
     exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells * 3; i++)
    //  fprintf(stderr, "h_b_validation[%d]: %lf\n", i, h_b_validation[i]);

    readfile = fread(h_grad_validation, cell_vec_bytes, 1, fp);
    if(readfile == 0) {
     fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
     exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells * 3; i++)
    //  fprintf(stderr, "h_grad_validation[%d]: %lf\n", i, h_grad_validation[i]);

    readfile = fread(h_turbSrc_A, csr_value_bytes, 1, fp);
    if(readfile == 0) {
     fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
     exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < (num_cells + num_faces); i++)
    //  fprintf(stderr, "h_turbSrc_A[%d]: %.30lf\n", i, h_turbSrc_A[i]);

    readfile = fread(h_turbSrc_b, cell_vec_bytes, 1, fp);
    if(readfile == 0) {
     fprintf(stderr, "%s %d, Failed to open input file: %s!\n", __FILE__, __LINE__, input_file);
     exit(EXIT_FAILURE);
    }
    // for (int i = 0; i < num_cells * 3; i++)
    //  fprintf(stderr, "h_turbSrc_b[%d]: %lf\n", i, h_turbSrc_b[i]);

    // compare with of results
    FILE *fp1 = NULL;
    fp1 = fopen("of_finalresult.txt", "rb+");
    readfile = fread(A_openfoam, csr_value_vec_bytes, 1, fp1);
    readfile = fread(b_openfoam, cell_vec_bytes, 1, fp1);

  } else {
    // Initialize row index
    h_A_csr_row_index[0] = 0;
    for (int i = 0; i < num_cells; ++i) {
      int ww = i % w;
      int hh = (i % (w * h)) / w;
      int dd = i / (w * h);
      int neighbor_number = 6;

      if (ww == 0) {
        neighbor_number -= 1;
      }
      if (ww == (w - 1)) {
        neighbor_number -= 1;
      }
      if (hh == 0) {
        neighbor_number -= 1;
      }
      if (hh == (h - 1)) {
        neighbor_number -= 1;
      }
      if (dd == 0) {
        neighbor_number -= 1;
      }
      if (dd == (d - 1)) {
        neighbor_number -= 1;
      }

      h_A_csr_row_index[i + 1] = h_A_csr_row_index[i] + neighbor_number;
    }

    // Initialize cell fields
    for (int i = 0; i < num_cells; ++i) {
      h_rho_old[i] = rand() / (double)RAND_MAX;
      h_rho_new[i] = rand() / (double)RAND_MAX;
      h_volume[i] = rand() / (double)RAND_MAX;
      h_pressure[i] = rand() / (double)RAND_MAX;
      h_velocity_old[i * 3 + 0] = rand() / (double)RAND_MAX;
      h_velocity_old[i * 3 + 1] = rand() / (double)RAND_MAX;
      h_velocity_old[i * 3 + 2] = rand() / (double)RAND_MAX;
    }

    // Initialize A_csr_col_index, A_csr_diag_index, and the face fields.
    for (int i = 0; i < num_cells; ++i) {
      int row_offset =  h_A_csr_row_index[i];
      int neighbor_offset = row_offset - i;

      double weight = 0.5;

      // Make sure that the values are placed in each row with the following order.
      // (x, y, z - 1)--> column: (i - w * h)
      // (x, y - 1, z)--> column: (i - w)
      // (x - 1, y, z)--> column: (i - 1)
      // (x, y, z)    --> column: (i) //only in A_csr_index and A_csr_diag_index
      // (x + 1, y, z)--> column: (i + 1)
      // (x, y + 1, z)--> column: (i + w)
      // (x, y, z + 1)--> column: (i + w * h)

      // (x, y, z - 1)
      // (x, y - 1, z)
      // (x - 1, y, z)
      int num_lower = 0;
      if (i - w * h >= 0) {
        num_lower++;
        h_A_csr_col_index[row_offset++] = i - w * h;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_phi[face_index] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
      }
      if (i - w >= 0) {
        num_lower++;
        h_A_csr_col_index[row_offset++] = i - w;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_phi[face_index] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
      }
      if (i - 1 >= 0) {
        num_lower++;
        h_A_csr_col_index[row_offset++] = i - 1;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_phi[face_index] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
      }

      // (x, y, z)
      h_A_csr_col_index[row_offset++] = i;
      h_A_csr_diag_index[i] = num_lower;

      // (x + 1, y, z)
      // (x, y + 1, z)
      // (x, y, z + 1)
      if (i + 1 < num_cells) {
        h_A_csr_col_index[row_offset++] = i + 1;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
        h_face_vector[face_index] = rand() / (double)RAND_MAX;
      }
      if (i + w < num_cells) {
        h_A_csr_col_index[row_offset++] = i + w;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_phi[face_index] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
      }
      if (i + w * h < num_cells) {
        h_A_csr_col_index[row_offset++] = i + w * h;

        int face_index = neighbor_offset++;
        h_weight[face_index] = weight;
        h_phi[face_index] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 0] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 1] = rand() / (double)RAND_MAX;
        h_face_vector[face_index * 3 + 2] = rand() / (double)RAND_MAX;
      }
    }
  }

  if (fp) {
    fclose(fp);
  }

  int total_bytes = 0;
  int copy_bytes = 0;

  // Allocate the device arrays
  int* d_A_csr_row_index = NULL;
  int* d_A_csr_col_index = NULL;
  int* d_A_csr_diag_index = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_A_csr_row_index, csr_row_index_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_A_csr_col_index, csr_col_index_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_A_csr_diag_index, cell_index_bytes));
  total_bytes += (csr_row_index_bytes + csr_col_index_bytes + cell_index_bytes);

  double* d_rho_old = NULL;
  double* d_rho_new = NULL;
  double* d_volume = NULL;
  double* d_pressure = NULL;
  double* d_velocity_old = NULL;
  double* d_weight = NULL;
  double* d_phi = NULL;
  double* d_face_vector = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_rho_new, cell_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_volume, cell_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_pressure, cell_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_velocity_old, cell_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_weight, face_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_phi, face_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_face_vector, face_vec_bytes));
  total_bytes += (cell_bytes * 4 + face_bytes * 2 + cell_vec_bytes + face_vec_bytes);

  int* d_boundary_cell_offset = NULL;
  int* d_boundary_cell_id = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_cell_id, boundary_face_bytes));
  total_bytes += (boundary_cell_index_bytes + (num_boundary_cells+1) * sizeof(int));

  double* d_boundary_pressure = NULL;
  double* d_boundary_face_vector = NULL;
  double* d_internal_coeffs = NULL;
  double* d_boundary_coeffs = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_boundary_pressure, boundary_face_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_face_vector, boundary_face_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, boundary_face_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, boundary_face_vec_bytes));
  total_bytes += (boundary_face_bytes + boundary_face_vec_bytes * 3);

  double* d_A_csr = NULL;
  double* d_b = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_A_csr, csr_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_b, cell_vec_bytes));
  total_bytes += (csr_value_vec_bytes + cell_vec_bytes);

  double* d_turbSrc_A = NULL;
  double* d_turbSrc_b = NULL;
  checkCudaErrors(cudaMalloc((void**)&d_turbSrc_A, csr_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_turbSrc_b, cell_vec_bytes));
  total_bytes += (csr_value_bytes + cell_vec_bytes);

  fprintf(stderr, "Total bytes malloc on GPU: %.2fMB\n", total_bytes * 1.0 / 1024 / 1024);

  // Create cuda stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  struct timeval start, end;
  float duration = 0;
  gettimeofday(&start, NULL);

  // Copy the host input array in host memory to the device input array in device memory
  printf("Copy input arrays from the host memory to the CUDA device\n");
  checkCudaErrors(cudaMemcpyAsync(d_A_csr_row_index, h_A_csr_row_index, csr_row_index_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_A_csr_col_index, h_A_csr_col_index, csr_col_index_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_A_csr_diag_index, h_A_csr_diag_index, cell_index_bytes, cudaMemcpyHostToDevice, stream));
  copy_bytes += (csr_row_index_bytes + csr_col_index_bytes + cell_index_bytes);

  checkCudaErrors(cudaMemcpyAsync(d_rho_old, h_rho_old, cell_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_rho_new, h_rho_new, cell_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_volume, h_volume, cell_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_pressure, h_pressure, cell_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_velocity_old, h_velocity_old, cell_vec_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_weight, h_weight, face_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_phi, h_phi, face_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_face_vector, h_face_vector, face_vec_bytes, cudaMemcpyHostToDevice, stream));
  copy_bytes += (cell_bytes * 4 + cell_vec_bytes + face_bytes * 2 + face_vec_bytes);

  checkCudaErrors(cudaMemcpyAsync(d_boundary_cell_offset, h_boundary_cell_offset, (num_boundary_cells+1) * sizeof(int), cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_boundary_cell_id, h_boundary_cell_id, boundary_face_index_bytes, cudaMemcpyHostToDevice, stream));
  copy_bytes += ((num_boundary_cells+1) * sizeof(int) + boundary_face_index_bytes);

  checkCudaErrors(cudaMemcpyAsync(d_boundary_pressure, h_boundary_pressure, boundary_face_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_boundary_face_vector, h_boundary_face_vector, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_internal_coeffs, h_internal_coeffs, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_boundary_coeffs, h_boundary_coeffs, boundary_face_vec_bytes, cudaMemcpyHostToDevice, stream));
  copy_bytes += (boundary_face_bytes + boundary_face_vec_bytes * 3);

  checkCudaErrors(cudaMemcpyAsync(d_turbSrc_A, h_turbSrc_A, csr_value_bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_turbSrc_b, h_turbSrc_b, cell_vec_bytes, cudaMemcpyHostToDevice, stream));
  copy_bytes += (csr_value_bytes + cell_vec_bytes);

  // Synchronize stream
  checkCudaErrors(cudaStreamSynchronize(stream));
  gettimeofday(&end, NULL);
  duration = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
  fprintf(stderr, "memcpy time cost %lf ms.\n", duration);
  fprintf(stderr, "bandwidth %lf GB/s.\n", (copy_bytes * 1.0 / 1024 / 1024 / 1024) / (duration / 1000));

  size_t threads_per_block = 1024;
  size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocks_per_grid, threads_per_block);

  /**
   * start term-level
   */
  gettimeofday(&start, NULL);

  // Memset A and b to zero
  checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
  checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));

  // Launch the fvm_ddt Kernel
  fvm_ddt<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, rdelta_t,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_rho_old, d_rho_new, d_volume, d_velocity_old, d_A_csr, d_b, d_A_csr, d_b);
  // Launch the fvm_div Kernel
  fvm_div_internal<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_weight, d_phi, d_A_csr, d_b, d_A_csr, d_b);
  blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
  fvm_div_boundary<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, num_boundary_cells,
          d_A_csr_row_index, d_A_csr_diag_index,
          d_boundary_cell_offset, d_boundary_cell_id,
          d_internal_coeffs, d_boundary_coeffs, d_A_csr, d_b, d_A_csr, d_b);
  // Launch the fvc_grad Kernel
  blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
  fvc_grad_internal_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells,
          d_A_csr_row_index, d_A_csr_col_index, d_A_csr_diag_index,
          d_face_vector, d_weight, d_pressure, d_b, d_b);
  blocks_per_grid = (num_boundary_cells + threads_per_block - 1) / threads_per_block;
  fvc_grad_boundary_face<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_cells,
          d_boundary_cell_offset, d_boundary_cell_id,
          d_boundary_face_vector, d_boundary_pressure, d_b, d_b);
  blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
  add_fvMatrix<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces,
          d_A_csr_row_index, d_turbSrc_A, d_turbSrc_b, d_A_csr, d_b, d_A_csr, d_b);

  // Synchronize stream
  checkCudaErrors(cudaStreamSynchronize(stream));
  gettimeofday(&end, NULL);
  duration = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
  fprintf(stderr, "assemble on term level, time cost %lf ms.\n", duration);
  /**
   * end term-level
   */

  /**
   * start equation-level
   */
  //gettimeofday(&start, NULL);

  //// Memset A and b to zero
  //checkCudaErrors(cudaMemsetAsync(d_A_csr, 0, csr_value_vec_bytes, stream));
  //checkCudaErrors(cudaMemsetAsync(d_b, 0, cell_vec_bytes, stream));

  //// Note: the implement fo assemble kernel is outdated.
  //// Launch the assemble Kernel
  //assemble<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_faces, rdelta_t, d_A_csr_row_index, d_A_csr_diag_index,
  //        d_rho_old, d_rho_new, d_volume, d_velocity_old, d_weight, d_phi, d_face_vector, d_pressure, d_A_csr, d_b);

  //// Synchronize stream
  //checkCudaErrors(cudaStreamSynchronize(stream));
  //gettimeofday(&end, NULL);
  //duration = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
  //fprintf(stderr, "assemble on equation level, time cost %lf ms.\n", duration);
  /**
   * end equation-level
   */

  // Sleep 10 seconds, so that user can check gpu memory usage through nvidia-smi.
  //sleep(10);

  // Check result
  if (validation_mode) {
    checkCudaErrors(cudaMemcpyAsync(h_A_csr, d_A_csr, csr_value_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync(h_b, d_b, cell_vec_bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    // fprintf(stderr, "check of h_A_csr\n");
    // checkVectorEqual((num_cells + num_faces) * 3, h_A_csr_validation, h_A_csr, 1e-5);
    // fprintf(stderr, "check of h_b\n");
    // checkVectorEqual(num_cells * 3, h_b_validation, h_b, 1e-5);
    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual((num_cells + num_faces) * 3, A_openfoam, h_A_csr, 1e-5);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(num_cells * 3, b_openfoam, h_b, 1e-5);
  }

  // Free device global memory
  checkCudaErrors(cudaFree(d_A_csr_row_index));
  checkCudaErrors(cudaFree(d_A_csr_col_index));
  checkCudaErrors(cudaFree(d_A_csr_diag_index));
  checkCudaErrors(cudaFree(d_rho_old));
  checkCudaErrors(cudaFree(d_rho_new));
  checkCudaErrors(cudaFree(d_volume));
  checkCudaErrors(cudaFree(d_pressure));
  checkCudaErrors(cudaFree(d_velocity_old));
  checkCudaErrors(cudaFree(d_weight));
  checkCudaErrors(cudaFree(d_phi));
  checkCudaErrors(cudaFree(d_face_vector));
  checkCudaErrors(cudaFree(d_boundary_cell_offset));
  checkCudaErrors(cudaFree(d_boundary_cell_id));
  checkCudaErrors(cudaFree(d_boundary_pressure));
  checkCudaErrors(cudaFree(d_boundary_face_vector));
  checkCudaErrors(cudaFree(d_internal_coeffs));
  checkCudaErrors(cudaFree(d_boundary_coeffs));
  checkCudaErrors(cudaFree(d_A_csr));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaStreamDestroy(stream));

  // Free host memory
  free(h_A_csr_row_index);
  free(h_A_csr_col_index);
  free(h_A_csr_diag_index);
  free(h_rho_old);
  free(h_rho_new);
  free(h_volume);
  free(h_pressure);
  free(h_velocity_old);
  free(h_weight);
  free(h_phi);
  free(h_face_vector);
  free(h_boundary_cell_offset);
  free(h_boundary_cell_id);
  free(h_boundary_pressure);
  free(h_boundary_face_vector);
  free(h_internal_coeffs);
  free(h_boundary_coeffs);
  free(h_A_csr);
  free(h_b);
  free(h_A_csr_validation);
  free(h_b_validation);

  printf("Done\n");
  return 0;
}


/**
 * ============================================================================
 * Parallel DCT-Based Sparse Matrix Scaling using CUDA
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * Based on MatGen paper by Pamuk et al. (2025)
 * 
 * DCT Method:
 * 1. Convert sparse matrix to dense
 * 2. Apply 2D DCT using cuFFT
 * 3. Resize coefficient matrix
 * 4. Apply inverse 2D DCT
 * 5. Threshold and convert back to sparse
 * 
 * ============================================================================
 * Compilation:
 *   nvcc -O3 -arch=sm_89 -o parallel_dct parallel_dct.cu -lcufft
 * 
 * Usage:
 *   ./parallel_dct -s 1000        # 1000x1000 test matrix
 *   ./parallel_dct -i matrix.mtx  # Load from file
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

// M_PI definition for Windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT Error at %s:%d - code %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Timer
// ============================================================================
typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

void timer_init(GpuTimer* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

void timer_start(GpuTimer* t) {
    cudaEventRecord(t->start, 0);
}

float timer_stop(GpuTimer* t) {
    float ms;
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

void timer_destroy(GpuTimer* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel: Prepare data for DCT (reorder for FFT-based DCT)
 * DCT-II can be computed via FFT by mirroring the input
 */
__global__ void prepare_dct_kernel(
    const float* __restrict__ input,
    cufftComplex* __restrict__ output,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        
        output[idx].x = input[idx];
        output[idx].y = 0.0f;
    }
}

/**
 * Kernel: Apply DCT normalization factors after FFT
 */
__global__ void apply_dct_factors_kernel(
    cufftComplex* data,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        
        float factor_row = (row == 0) ? sqrtf(1.0f / rows) : sqrtf(2.0f / rows);
        float factor_col = (col == 0) ? sqrtf(1.0f / cols) : sqrtf(2.0f / cols);
        float factor = factor_row * factor_col;
        
        // Apply phase shift for DCT
        float phase_row = -M_PI * row / (2.0f * rows);
        float phase_col = -M_PI * col / (2.0f * cols);
        float phase = phase_row + phase_col;
        
        float cos_phase = cosf(phase);
        float sin_phase = sinf(phase);
        
        float real = data[idx].x * cos_phase - data[idx].y * sin_phase;
        float imag = data[idx].x * sin_phase + data[idx].y * cos_phase;
        
        data[idx].x = real * factor;
        data[idx].y = imag * factor;
    }
}

/**
 * Kernel: Resize DCT coefficients (zero-pad or truncate)
 */
__global__ void resize_coefficients_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    int in_rows, int in_cols,
    int out_rows, int out_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_rows * out_cols;
    
    if (idx < total) {
        int out_row = idx / out_cols;
        int out_col = idx % out_cols;
        
        if (out_row < in_rows && out_col < in_cols) {
            int in_idx = out_row * in_cols + out_col;
            output[idx] = input[in_idx];
        } else {
            output[idx].x = 0.0f;
            output[idx].y = 0.0f;
        }
    }
}

/**
 * Kernel: Apply IDCT normalization factors
 */
__global__ void apply_idct_factors_kernel(
    cufftComplex* data,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        
        float factor_row = (row == 0) ? sqrtf(1.0f / rows) : sqrtf(2.0f / rows);
        float factor_col = (col == 0) ? sqrtf(1.0f / cols) : sqrtf(2.0f / cols);
        
        // Phase shift for IDCT
        float phase_row = M_PI * row / (2.0f * rows);
        float phase_col = M_PI * col / (2.0f * cols);
        float phase = phase_row + phase_col;
        
        float cos_phase = cosf(phase);
        float sin_phase = sinf(phase);
        
        float real = data[idx].x * cos_phase - data[idx].y * sin_phase;
        float imag = data[idx].x * sin_phase + data[idx].y * cos_phase;
        
        data[idx].x = real * factor_row * factor_col;
        data[idx].y = imag * factor_row * factor_col;
    }
}

/**
 * Kernel: Extract real part and apply threshold
 */
__global__ void threshold_kernel(
    const cufftComplex* __restrict__ input,
    float* __restrict__ output,
    int* __restrict__ nnz_flags,
    int size,
    float threshold,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = input[idx].x * scale;
        
        if (fabsf(val) > threshold) {
            output[idx] = val;
            nnz_flags[idx] = 1;
        } else {
            output[idx] = 0.0f;
            nnz_flags[idx] = 0;
        }
    }
}

/**
 * Kernel: Convert sparse to dense (for DCT input)
 */
__global__ void sparse_to_dense_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    float* __restrict__ dense,
    int num_rows, int num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        
        for (int j = start; j < end; j++) {
            int col = col_idx[j];
            dense[row * num_cols + col] = values[j];
        }
    }
}

// ============================================================================
// Matrix Structures
// ============================================================================
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int* row_ptr;
    int* col_idx;
    float* values;
} CSRMatrix;

// ============================================================================
// Matrix I/O
// ============================================================================
CSRMatrix* generate_banded_matrix(int size, int bandwidth) {
    int est_nnz = size * (2 * bandwidth + 1);
    
    int* temp_rows = (int*)malloc(est_nnz * sizeof(int));
    int* temp_cols = (int*)malloc(est_nnz * sizeof(int));
    float* temp_vals = (float*)malloc(est_nnz * sizeof(float));
    
    int nnz = 0;
    for (int i = 0; i < size; i++) {
        int j_start = (i - bandwidth > 0) ? i - bandwidth : 0;
        int j_end = (i + bandwidth < size - 1) ? i + bandwidth : size - 1;
        
        for (int j = j_start; j <= j_end; j++) {
            temp_rows[nnz] = i;
            temp_cols[nnz] = j;
            temp_vals[nnz] = 1.0f + 0.1f * abs(i - j);
            nnz++;
        }
    }
    
    CSRMatrix* matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    matrix->num_rows = size;
    matrix->num_cols = size;
    matrix->nnz = nnz;
    matrix->row_ptr = (int*)calloc(size + 1, sizeof(int));
    matrix->col_idx = (int*)malloc(nnz * sizeof(int));
    matrix->values = (float*)malloc(nnz * sizeof(float));
    
    for (int i = 0; i < nnz; i++) {
        matrix->row_ptr[temp_rows[i] + 1]++;
    }
    for (int i = 0; i < size; i++) {
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];
    }
    
    memcpy(matrix->col_idx, temp_cols, nnz * sizeof(int));
    memcpy(matrix->values, temp_vals, nnz * sizeof(float));
    
    free(temp_rows);
    free(temp_cols);
    free(temp_vals);
    
    return matrix;
}

void free_csr_matrix(CSRMatrix* matrix) {
    if (matrix) {
        free(matrix->row_ptr);
        free(matrix->col_idx);
        free(matrix->values);
        free(matrix);
    }
}

// ============================================================================
// DCT Scaling Function
// ============================================================================
typedef struct {
    float gpu_time_ms;
    float cpu_time_ms;
    float speedup;
    int output_nnz;
} BenchmarkResult;

BenchmarkResult dct_scale_benchmark(CSRMatrix* input, int target_rows, int target_cols,
                                    float threshold, int num_runs) {
    BenchmarkResult result = {0};
    
    int in_rows = input->num_rows;
    int in_cols = input->num_cols;
    int in_size = in_rows * in_cols;
    int out_size = target_rows * target_cols;
    
    // Device memory
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_dense_in, *d_dense_out;
    cufftComplex *d_freq_in, *d_freq_out;
    int *d_nnz_flags;
    
    // Allocate
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (in_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, input->nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, input->nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense_in, in_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_freq_in, in_size * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_freq_out, out_size * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_nnz_flags, out_size * sizeof(int)));
    
    // Initialize dense to zero
    CUDA_CHECK(cudaMemset(d_dense_in, 0, in_size * sizeof(float)));
    
    // Copy input
    CUDA_CHECK(cudaMemcpy(d_row_ptr, input->row_ptr, (in_rows + 1) * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, input->col_idx, input->nnz * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, input->values, input->nnz * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Kernel config
    int block_size = 256;
    int grid_sparse = (in_rows + block_size - 1) / block_size;
    int grid_in = (in_size + block_size - 1) / block_size;
    int grid_out = (out_size + block_size - 1) / block_size;
    
    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;
    CUFFT_CHECK(cufftPlan2d(&plan_forward, in_rows, in_cols, CUFFT_C2C));
    CUFFT_CHECK(cufftPlan2d(&plan_inverse, target_rows, target_cols, CUFFT_C2C));
    
    // Timer
    GpuTimer timer;
    timer_init(&timer);
    
    float total_time = 0;
    
    for (int run = 0; run < num_runs; run++) {
        timer_start(&timer);
        
        // Step 1: Sparse to Dense
        CUDA_CHECK(cudaMemset(d_dense_in, 0, in_size * sizeof(float)));
        sparse_to_dense_kernel<<<grid_sparse, block_size>>>(
            d_row_ptr, d_col_idx, d_values, d_dense_in, in_rows, in_cols);
        
        // Step 2: Prepare for FFT
        prepare_dct_kernel<<<grid_in, block_size>>>(d_dense_in, d_freq_in, in_rows, in_cols);
        
        // Step 3: Forward FFT (approximates DCT)
        CUFFT_CHECK(cufftExecC2C(plan_forward, d_freq_in, d_freq_in, CUFFT_FORWARD));
        
        // Step 4: Apply DCT factors
        apply_dct_factors_kernel<<<grid_in, block_size>>>(d_freq_in, in_rows, in_cols);
        
        // Step 5: Resize coefficients
        resize_coefficients_kernel<<<grid_out, block_size>>>(
            d_freq_in, d_freq_out, in_rows, in_cols, target_rows, target_cols);
        
        // Step 6: Apply IDCT factors
        apply_idct_factors_kernel<<<grid_out, block_size>>>(d_freq_out, target_rows, target_cols);
        
        // Step 7: Inverse FFT
        CUFFT_CHECK(cufftExecC2C(plan_inverse, d_freq_out, d_freq_out, CUFFT_INVERSE));
        
        // Step 8: Threshold
        float scale = 1.0f / (target_rows * target_cols);
        threshold_kernel<<<grid_out, block_size>>>(
            d_freq_out, d_dense_out, d_nnz_flags, out_size, threshold, scale);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        total_time += timer_stop(&timer);
    }
    
    result.gpu_time_ms = total_time / num_runs;
    
    // Count nonzeros
    int* h_nnz_flags = (int*)malloc(out_size * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_nnz_flags, d_nnz_flags, out_size * sizeof(int), 
                         cudaMemcpyDeviceToHost));
    
    result.output_nnz = 0;
    for (int i = 0; i < out_size; i++) {
        result.output_nnz += h_nnz_flags[i];
    }
    
    // Cleanup
    free(h_nnz_flags);
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    timer_destroy(&timer);
    
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_dense_in));
    CUDA_CHECK(cudaFree(d_dense_out));
    CUDA_CHECK(cudaFree(d_freq_in));
    CUDA_CHECK(cudaFree(d_freq_out));
    CUDA_CHECK(cudaFree(d_nnz_flags));
    
    return result;
}

// ============================================================================
// Main
// ============================================================================
void print_header() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║      Parallel DCT Sparse Matrix Scaling (CUDA + cuFFT)           ║\n");
    printf("║                                                                  ║\n");
    printf("║      Authors: Ali Ahmet Taşkesen, Ömer Yıldırım                  ║\n");
    printf("║      Ankara Yıldırım Beyazıt University                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ GPU: %-58s │\n", prop.name);
    printf("│ Compute: %d.%d    SMs: %-3d    Memory: %.1f GB                    │\n", 
           prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, char** argv) {
    // Defaults
    int size = 1000;
    int bandwidth = 10;
    int num_runs = 5;
    float threshold = 1e-6f;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bandwidth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-s size] [-b bandwidth] [-r runs] [-t threshold]\n", argv[0]);
            return 0;
        }
    }
    
    print_header();
    print_device_info();
    
    // Generate matrix
    printf("Generating banded matrix: %d x %d (bandwidth=%d)\n", size, size, bandwidth);
    CSRMatrix* matrix = generate_banded_matrix(size, bandwidth);
    printf("NNZ: %d, Density: %.6f\n\n", matrix->nnz, 
           (float)matrix->nnz / (size * size));
    
    // Operations
    int n = matrix->num_rows;
    struct {
        const char* name;
        int target_rows;
        int target_cols;
    } operations[] = {
        {"Expand (+1)",      n + 1, n + 1},
        {"Upscale (2x)",     n * 2, n * 2},
        {"Reduce (-1)",      n - 1, n - 1},
        {"Downscale (1/2x)", n / 2, n / 2},
    };
    int num_ops = sizeof(operations) / sizeof(operations[0]);
    
    // Benchmark
    printf("┌───────────────────┬────────────┬───────────┬────────────┐\n");
    printf("│ Operation         │ Target     │ GPU (ms)  │ Output NNZ │\n");
    printf("├───────────────────┼────────────┼───────────┼────────────┤\n");
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = dct_scale_benchmark(matrix, 
                                                  operations[i].target_rows,
                                                  operations[i].target_cols,
                                                  threshold, num_runs);
        
        char target_str[20];
        sprintf(target_str, "%d x %d", operations[i].target_rows, operations[i].target_cols);
        
        printf("│ %-17s │ %-10s │ %9.3f │ %10d │\n",
               operations[i].name, target_str, res.gpu_time_ms, res.output_nnz);
    }
    
    printf("└───────────────────┴────────────┴───────────┴────────────┘\n\n");
    
    free_csr_matrix(matrix);
    printf("Done!\n");
    
    return 0;
}

/**
 * ============================================================================
 * Parallel Nearest Neighbor Sparse Matrix Scaling using CUDA
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * Based on MatGen paper by Pamuk et al. (2025)
 * "MatGen: A Realistic Sparse Matrix Generator Using Signal Processing 
 *  and Image Processing Methods"
 * 
 * ============================================================================
 * Compilation:
 *   nvcc -O3 -arch=sm_70 -o parallel_nn parallel_nn.cu
 * 
 * Usage:
 *   ./parallel_nn                    # Run with default 5000x5000 banded matrix
 *   ./parallel_nn -s 10000           # Run with 10000x10000 matrix
 *   ./parallel_nn -i matrix.mtx      # Load matrix from file
 *   ./parallel_nn -s 5000 -b 10      # 5000x5000 with bandwidth 10
 * 
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

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

// ============================================================================
// Timer Utilities
// ============================================================================
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

typedef struct {
    clock_t start;
    clock_t stop;
} CpuTimer;

void gpu_timer_init(GpuTimer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
}

void gpu_timer_start(GpuTimer* timer) {
    cudaEventRecord(timer->start, 0);
}

float gpu_timer_stop(GpuTimer* timer) {
    float elapsed;
    cudaEventRecord(timer->stop, 0);
    cudaEventSynchronize(timer->stop);
    cudaEventElapsedTime(&elapsed, timer->start, timer->stop);
    return elapsed;
}

void gpu_timer_destroy(GpuTimer* timer) {
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
}

void cpu_timer_start(CpuTimer* timer) {
    timer->start = clock();
}

float cpu_timer_stop(CpuTimer* timer) {
    timer->stop = clock();
    return ((float)(timer->stop - timer->start) / CLOCKS_PER_SEC) * 1000.0f;
}

// ============================================================================
// Data Structures
// ============================================================================

// CSR (Compressed Sparse Row) Matrix
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int* row_ptr;      // size: num_rows + 1
    int* col_idx;      // size: nnz
    float* values;     // size: nnz
} CSRMatrix;

// COO (Coordinate) Matrix - used for output
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int* row_idx;
    int* col_idx;
    float* values;
} COOMatrix;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel 1: Compute row index for each nonzero element
 * Uses binary search on row_ptr array
 * Each thread processes one nonzero
 */
__global__ void compute_row_indices_kernel(
    const int* __restrict__ d_row_ptr,
    int* __restrict__ d_row_indices,
    int num_rows,
    int nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        // Binary search to find which row this nonzero belongs to
        int low = 0, high = num_rows;
        while (low < high) {
            int mid = (low + high) / 2;
            if (d_row_ptr[mid + 1] <= tid) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        d_row_indices[tid] = low;
    }
}

/**
 * Kernel 2: Apply Nearest Neighbor scaling to coordinates
 * Each thread processes one nonzero element
 */
__global__ void nn_scale_kernel(
    const int* __restrict__ d_row_indices,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_values,
    int* __restrict__ d_out_rows,
    int* __restrict__ d_out_cols,
    float* __restrict__ d_out_values,
    int nnz,
    float scale_r,
    float scale_c,
    int target_rows,
    int target_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int row = d_row_indices[tid];
        int col = d_col_idx[tid];
        float val = d_values[tid];
        
        // Nearest Neighbor interpolation: round to nearest integer
        int new_row = (int)roundf((float)row / scale_r);
        int new_col = (int)roundf((float)col / scale_c);
        
        // Clip to valid range [0, target-1]
        new_row = min(max(new_row, 0), target_rows - 1);
        new_col = min(max(new_col, 0), target_cols - 1);
        
        // Store output
        d_out_rows[tid] = new_row;
        d_out_cols[tid] = new_col;
        d_out_values[tid] = val;
    }
}

/**
 * Combined kernel: Compute row indices and scale in one pass
 * More efficient for smaller matrices
 */
__global__ void nn_scale_combined_kernel(
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_col_idx,
    const float* __restrict__ d_values,
    int* __restrict__ d_out_rows,
    int* __restrict__ d_out_cols,
    float* __restrict__ d_out_values,
    int num_rows,
    int nnz,
    float scale_r,
    float scale_c,
    int target_rows,
    int target_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        // Binary search for row index
        int low = 0, high = num_rows;
        while (low < high) {
            int mid = (low + high) / 2;
            if (d_row_ptr[mid + 1] <= tid) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        int row = low;
        int col = d_col_idx[tid];
        float val = d_values[tid];
        
        // Scale coordinates
        int new_row = (int)roundf((float)row / scale_r);
        int new_col = (int)roundf((float)col / scale_c);
        
        // Clip
        new_row = min(max(new_row, 0), target_rows - 1);
        new_col = min(max(new_col, 0), target_cols - 1);
        
        d_out_rows[tid] = new_row;
        d_out_cols[tid] = new_col;
        d_out_values[tid] = val;
    }
}

// ============================================================================
// CPU Sequential Implementation (for comparison)
// ============================================================================

void cpu_nn_scale(CSRMatrix* input, int target_rows, int target_cols,
                  int* out_rows, int* out_cols, float* out_values) {
    float scale_r = (float)input->num_rows / target_rows;
    float scale_c = (float)input->num_cols / target_cols;
    
    int idx = 0;
    for (int row = 0; row < input->num_rows; row++) {
        for (int j = input->row_ptr[row]; j < input->row_ptr[row + 1]; j++) {
            int col = input->col_idx[j];
            float val = input->values[j];
            
            int new_row = (int)roundf((float)row / scale_r);
            int new_col = (int)roundf((float)col / scale_c);
            
            // Clip
            new_row = new_row < 0 ? 0 : (new_row >= target_rows ? target_rows - 1 : new_row);
            new_col = new_col < 0 ? 0 : (new_col >= target_cols ? target_cols - 1 : new_col);
            
            out_rows[idx] = new_row;
            out_cols[idx] = new_col;
            out_values[idx] = val;
            idx++;
        }
    }
}

// ============================================================================
// Matrix I/O Functions
// ============================================================================

CSRMatrix* read_matrix_market(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    char line[1024];
    int is_symmetric = 0;
    int is_pattern = 0;
    
    // Parse header
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') {
            if (strstr(line, "symmetric")) is_symmetric = 1;
            if (strstr(line, "pattern")) is_pattern = 1;
            continue;
        }
        break;
    }
    
    int num_rows, num_cols, nnz_file;
    sscanf(line, "%d %d %d", &num_rows, &num_cols, &nnz_file);
    
    // Allocate COO temporary storage
    int max_nnz = is_symmetric ? nnz_file * 2 : nnz_file;
    int* temp_rows = (int*)malloc(max_nnz * sizeof(int));
    int* temp_cols = (int*)malloc(max_nnz * sizeof(int));
    float* temp_vals = (float*)malloc(max_nnz * sizeof(float));
    
    int actual_nnz = 0;
    for (int i = 0; i < nnz_file; i++) {
        int r, c;
        float v = 1.0f;
        
        if (is_pattern) {
            if (fscanf(f, "%d %d", &r, &c) != 2) break;
        } else {
            if (fscanf(f, "%d %d %f", &r, &c, &v) != 3) break;
        }
        
        r--; c--;  // 0-indexed
        
        temp_rows[actual_nnz] = r;
        temp_cols[actual_nnz] = c;
        temp_vals[actual_nnz] = v;
        actual_nnz++;
        
        if (is_symmetric && r != c) {
            temp_rows[actual_nnz] = c;
            temp_cols[actual_nnz] = r;
            temp_vals[actual_nnz] = v;
            actual_nnz++;
        }
    }
    fclose(f);
    
    // Convert to CSR
    CSRMatrix* matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    matrix->num_rows = num_rows;
    matrix->num_cols = num_cols;
    matrix->nnz = actual_nnz;
    matrix->row_ptr = (int*)calloc(num_rows + 1, sizeof(int));
    matrix->col_idx = (int*)malloc(actual_nnz * sizeof(int));
    matrix->values = (float*)malloc(actual_nnz * sizeof(float));
    
    // Count per row
    for (int i = 0; i < actual_nnz; i++) {
        matrix->row_ptr[temp_rows[i] + 1]++;
    }
    
    // Prefix sum
    for (int i = 0; i < num_rows; i++) {
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];
    }
    
    // Fill data
    int* row_counts = (int*)calloc(num_rows, sizeof(int));
    for (int i = 0; i < actual_nnz; i++) {
        int row = temp_rows[i];
        int dest = matrix->row_ptr[row] + row_counts[row];
        matrix->col_idx[dest] = temp_cols[i];
        matrix->values[dest] = temp_vals[i];
        row_counts[row]++;
    }
    
    free(temp_rows);
    free(temp_cols);
    free(temp_vals);
    free(row_counts);
    
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

// Generate banded test matrix
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

// ============================================================================
// Main Scaling Functions
// ============================================================================

typedef struct {
    float gpu_time_ms;
    float cpu_time_ms;
    float speedup;
    int output_nnz;
    int errors;
} BenchmarkResult;

BenchmarkResult run_benchmark(CSRMatrix* input, int target_rows, int target_cols, 
                              int verify, int num_runs) {
    BenchmarkResult result = {0};
    
    int num_rows = input->num_rows;
    int nnz = input->nnz;
    float scale_r = (float)num_rows / target_rows;
    float scale_c = (float)input->num_cols / target_cols;
    
    // ========== GPU Execution ==========
    
    // Allocate device memory
    int *d_row_ptr, *d_col_idx;
    float *d_values;
    int *d_out_rows, *d_out_cols;
    float *d_out_values;
    
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_values, nnz * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, input->row_ptr, (num_rows + 1) * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, input->col_idx, nnz * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, input->values, nnz * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Kernel config
    int block_size = 256;
    int grid_size = (nnz + block_size - 1) / block_size;
    
    // Warmup
    nn_scale_combined_kernel<<<grid_size, block_size>>>(
        d_row_ptr, d_col_idx, d_values,
        d_out_rows, d_out_cols, d_out_values,
        num_rows, nnz, scale_r, scale_c, target_rows, target_cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    GpuTimer gpu_timer;
    gpu_timer_init(&gpu_timer);
    
    float total_gpu_time = 0;
    for (int r = 0; r < num_runs; r++) {
        gpu_timer_start(&gpu_timer);
        
        nn_scale_combined_kernel<<<grid_size, block_size>>>(
            d_row_ptr, d_col_idx, d_values,
            d_out_rows, d_out_cols, d_out_values,
            num_rows, nnz, scale_r, scale_c, target_rows, target_cols);
        
        total_gpu_time += gpu_timer_stop(&gpu_timer);
    }
    result.gpu_time_ms = total_gpu_time / num_runs;
    
    gpu_timer_destroy(&gpu_timer);
    
    // Copy results back
    int* h_gpu_rows = (int*)malloc(nnz * sizeof(int));
    int* h_gpu_cols = (int*)malloc(nnz * sizeof(int));
    float* h_gpu_vals = (float*)malloc(nnz * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_gpu_rows, d_out_rows, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_cols, d_out_cols, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_vals, d_out_values, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_out_rows));
    CUDA_CHECK(cudaFree(d_out_cols));
    CUDA_CHECK(cudaFree(d_out_values));
    
    // ========== CPU Execution (for comparison) ==========
    
    int* h_cpu_rows = (int*)malloc(nnz * sizeof(int));
    int* h_cpu_cols = (int*)malloc(nnz * sizeof(int));
    float* h_cpu_vals = (float*)malloc(nnz * sizeof(float));
    
    CpuTimer cpu_timer;
    
    float total_cpu_time = 0;
    for (int r = 0; r < num_runs; r++) {
        cpu_timer_start(&cpu_timer);
        cpu_nn_scale(input, target_rows, target_cols, h_cpu_rows, h_cpu_cols, h_cpu_vals);
        total_cpu_time += cpu_timer_stop(&cpu_timer);
    }
    result.cpu_time_ms = total_cpu_time / num_runs;
    
    // ========== Verification ==========
    
    if (verify) {
        result.errors = 0;
        for (int i = 0; i < nnz; i++) {
            if (h_gpu_rows[i] != h_cpu_rows[i] || 
                h_gpu_cols[i] != h_cpu_cols[i] ||
                fabsf(h_gpu_vals[i] - h_cpu_vals[i]) > 1e-5f) {
                result.errors++;
            }
        }
    }
    
    result.output_nnz = nnz;
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    // Cleanup
    free(h_gpu_rows);
    free(h_gpu_cols);
    free(h_gpu_vals);
    free(h_cpu_rows);
    free(h_cpu_cols);
    free(h_cpu_vals);
    
    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_header() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     Parallel Nearest Neighbor Sparse Matrix Scaling (CUDA)       ║\n");
    printf("║                                                                  ║\n");
    printf("║     Authors: Ali Ahmet Taşkesen, Ömer Yıldırım                   ║\n");
    printf("║     Ankara Yıldırım Beyazıt University                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ GPU Device Information                                          │\n");
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ Device:              %-42s │\n", prop.name);
    printf("│ Compute Capability:  %d.%d                                        │\n", prop.major, prop.minor);
    printf("│ Multiprocessors:     %-42d │\n", prop.multiProcessorCount);
    printf("│ Max Threads/Block:   %-42d │\n", prop.maxThreadsPerBlock);
    printf("│ Global Memory:       %-39.2f GB │\n", prop.totalGlobalMem / 1e9);
    printf("│ Shared Mem/Block:    %-39.0f KB │\n", prop.sharedMemPerBlock / 1024.0);
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -i <file>    Input matrix (Matrix Market format)\n");
    printf("  -s <size>    Generate test matrix of given size (default: 5000)\n");
    printf("  -b <bw>      Bandwidth for generated matrix (default: 10)\n");
    printf("  -r <runs>    Number of benchmark runs (default: 10)\n");
    printf("  -v           Enable verification (compare GPU vs CPU)\n");
    printf("  -h           Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -s 10000                    # 10000x10000 banded matrix\n", prog);
    printf("  %s -s 5000 -b 20 -v            # 5000x5000, bandwidth=20, verify\n", prog);
    printf("  %s -i cavity03.mtx             # Load from file\n", prog);
}

int main(int argc, char** argv) {
    // Defaults
    int size = 5000;
    int bandwidth = 10;
    int num_runs = 10;
    int verify = 0;
    char* input_file = NULL;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bandwidth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    print_header();
    print_device_info();
    
    // Load or generate matrix
    CSRMatrix* matrix;
    if (input_file) {
        printf("Loading matrix from: %s\n", input_file);
        matrix = read_matrix_market(input_file);
        if (!matrix) {
            fprintf(stderr, "Failed to load matrix!\n");
            return 1;
        }
    } else {
        printf("Generating banded test matrix...\n");
        printf("  Size: %d x %d\n", size, size);
        printf("  Bandwidth: %d\n", bandwidth);
        matrix = generate_banded_matrix(size, bandwidth);
    }
    
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ Matrix Properties                                               │\n");
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ Dimensions:          %d x %-32d │\n", matrix->num_rows, matrix->num_cols);
    printf("│ Nonzeros (NNZ):      %-42d │\n", matrix->nnz);
    printf("│ Density:             %-42.6f │\n", 
           (float)matrix->nnz / ((float)matrix->num_rows * matrix->num_cols));
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    // Define scaling operations
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
        {"Upscale (4x)",     n * 4, n * 4},
    };
    int num_ops = sizeof(operations) / sizeof(operations[0]);
    
    // Run benchmarks
    printf("Running benchmarks (%d runs each)...\n\n", num_runs);
    
    printf("┌───────────────────┬────────────┬───────────┬───────────┬─────────┬────────┐\n");
    printf("│ Operation         │ Target     │ GPU (ms)  │ CPU (ms)  │ Speedup │ Errors │\n");
    printf("├───────────────────┼────────────┼───────────┼───────────┼─────────┼────────┤\n");
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = run_benchmark(matrix, 
                                           operations[i].target_rows,
                                           operations[i].target_cols,
                                           verify, num_runs);
        
        char target_str[16];
        if (operations[i].target_rows >= 10000) {
            sprintf(target_str, "%dk x %dk", 
                    operations[i].target_rows / 1000,
                    operations[i].target_cols / 1000);
        } else {
            sprintf(target_str, "%d x %d", 
                    operations[i].target_rows,
                    operations[i].target_cols);
        }
        
        printf("│ %-17s │ %-10s │ %9.3f │ %9.3f │ %6.2fx │ %6d │\n",
               operations[i].name,
               target_str,
               res.gpu_time_ms,
               res.cpu_time_ms,
               res.speedup,
               verify ? res.errors : 0);
    }
    
    printf("└───────────────────┴────────────┴───────────┴───────────┴─────────┴────────┘\n");
    printf("\n");
    
    if (verify) {
        printf("✓ Verification: GPU results compared against CPU implementation\n");
    }
    
    // Cleanup
    free_csr_matrix(matrix);
    
    printf("\nDone!\n");
    return 0;
}

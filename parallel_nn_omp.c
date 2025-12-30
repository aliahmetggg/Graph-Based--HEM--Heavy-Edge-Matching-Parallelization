/**
 * ============================================================================
 * Parallel Nearest Neighbor Sparse Matrix Scaling using OpenMP
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * ============================================================================
 * Compilation:
 *   Linux:   gcc -O3 -fopenmp -o parallel_nn_omp parallel_nn_omp.c -lm
 *   Windows: cl /O2 /openmp parallel_nn_omp.c
 * 
 * Usage:
 *   ./parallel_nn_omp -s 5000 -v
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int* row_ptr;
    int* col_idx;
    float* values;
} CSRMatrix;

typedef struct {
    float omp_time_ms;
    float seq_time_ms;
    float speedup;
    int errors;
    int output_nnz;
} BenchmarkResult;

// ============================================================================
// Matrix Generation
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
    
    // Count per row
    for (int i = 0; i < nnz; i++) {
        matrix->row_ptr[temp_rows[i] + 1]++;
    }
    
    // Prefix sum
    for (int i = 0; i < size; i++) {
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];
    }
    
    // Fill
    int* row_counts = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < nnz; i++) {
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

// ============================================================================
// Sequential NN Scaling
// ============================================================================

void seq_nn_scale(CSRMatrix* input, int target_rows, int target_cols,
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
// OpenMP Parallel NN Scaling
// ============================================================================

void omp_nn_scale(CSRMatrix* input, int target_rows, int target_cols,
                  int* out_rows, int* out_cols, float* out_values) {
    float scale_r = (float)input->num_rows / target_rows;
    float scale_c = (float)input->num_cols / target_cols;
    int num_rows = input->num_rows;
    int nnz = input->nnz;
    
    // First, compute row indices for each nonzero (parallel)
    int* row_indices = (int*)malloc(nnz * sizeof(int));
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < nnz; i++) {
        // Binary search for row
        int low = 0, high = num_rows;
        while (low < high) {
            int mid = (low + high) / 2;
            if (input->row_ptr[mid + 1] <= i) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        row_indices[i] = low;
    }
    
    // Apply NN scaling (parallel)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nnz; i++) {
        int row = row_indices[i];
        int col = input->col_idx[i];
        float val = input->values[i];
        
        int new_row = (int)roundf((float)row / scale_r);
        int new_col = (int)roundf((float)col / scale_c);
        
        // Clip
        new_row = new_row < 0 ? 0 : (new_row >= target_rows ? target_rows - 1 : new_row);
        new_col = new_col < 0 ? 0 : (new_col >= target_cols ? target_cols - 1 : new_col);
        
        out_rows[i] = new_row;
        out_cols[i] = new_col;
        out_values[i] = val;
    }
    
    free(row_indices);
}

// ============================================================================
// Benchmark
// ============================================================================

BenchmarkResult run_benchmark(CSRMatrix* input, int target_rows, int target_cols,
                              int verify, int num_runs) {
    BenchmarkResult result = {0};
    int nnz = input->nnz;
    
    // Allocate output arrays
    int* omp_rows = (int*)malloc(nnz * sizeof(int));
    int* omp_cols = (int*)malloc(nnz * sizeof(int));
    float* omp_vals = (float*)malloc(nnz * sizeof(float));
    
    int* seq_rows = (int*)malloc(nnz * sizeof(int));
    int* seq_cols = (int*)malloc(nnz * sizeof(int));
    float* seq_vals = (float*)malloc(nnz * sizeof(float));
    
    // Warmup
    omp_nn_scale(input, target_rows, target_cols, omp_rows, omp_cols, omp_vals);
    
    // OpenMP timing
    double total_omp = 0;
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        omp_nn_scale(input, target_rows, target_cols, omp_rows, omp_cols, omp_vals);
        total_omp += (omp_get_wtime() - start) * 1000.0;
    }
    result.omp_time_ms = total_omp / num_runs;
    
    // Sequential timing
    double total_seq = 0;
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        seq_nn_scale(input, target_rows, target_cols, seq_rows, seq_cols, seq_vals);
        total_seq += (omp_get_wtime() - start) * 1000.0;
    }
    result.seq_time_ms = total_seq / num_runs;
    
    // Verify
    if (verify) {
        result.errors = 0;
        for (int i = 0; i < nnz; i++) {
            if (omp_rows[i] != seq_rows[i] || 
                omp_cols[i] != seq_cols[i] ||
                fabsf(omp_vals[i] - seq_vals[i]) > 1e-5f) {
                result.errors++;
            }
        }
    }
    
    result.output_nnz = nnz;
    result.speedup = result.seq_time_ms / result.omp_time_ms;
    
    free(omp_rows);
    free(omp_cols);
    free(omp_vals);
    free(seq_rows);
    free(seq_cols);
    free(seq_vals);
    
    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_header() {
    printf("\n");
    printf("==================================================================\n");
    printf("   Parallel Nearest Neighbor Sparse Matrix Scaling (OpenMP)\n");
    printf("\n");
    printf("   Authors: Ali Ahmet Taskesen, Omer Yildirim\n");
    printf("   Ankara Yildirim Beyazit University\n");
    printf("==================================================================\n");
    printf("\n");
}

void print_system_info() {
    printf("------------------------------------------------------------------\n");
    printf(" System Information\n");
    printf("------------------------------------------------------------------\n");
    printf(" OpenMP Threads:       %d\n", omp_get_max_threads());
    printf(" Max Threads:          %d\n", omp_get_num_procs());
    printf("------------------------------------------------------------------\n\n");
}

int main(int argc, char** argv) {
    int size = 5000;
    int bandwidth = 10;
    int num_runs = 10;
    int verify = 0;
    int num_threads = 0;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bandwidth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-s size] [-b bandwidth] [-r runs] [-t threads] [-v]\n", argv[0]);
            return 0;
        }
    }
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    print_header();
    print_system_info();
    
    // Generate matrix
    printf("Generating banded matrix: %d x %d (bandwidth=%d)\n", size, size, bandwidth);
    CSRMatrix* matrix = generate_banded_matrix(size, bandwidth);
    printf("NNZ: %d, Density: %.6f\n\n", matrix->nnz,
           (float)matrix->nnz / ((float)size * size));
    
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
        {"Upscale (4x)",     n * 4, n * 4},
    };
    int num_ops = sizeof(operations) / sizeof(operations[0]);
    
    printf("Running benchmarks (%d runs each)...\n\n", num_runs);
    
    printf("+-------------------+------------+-----------+-----------+---------+--------+\n");
    printf("| Operation         | Target     | OMP (ms)  | Seq (ms)  | Speedup | Errors |\n");
    printf("+-------------------+------------+-----------+-----------+---------+--------+\n");
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = run_benchmark(matrix,
                                           operations[i].target_rows,
                                           operations[i].target_cols,
                                           verify, num_runs);
        
        char target_str[20];
        if (operations[i].target_rows >= 10000) {
            sprintf(target_str, "%dk x %dk",
                    operations[i].target_rows / 1000,
                    operations[i].target_cols / 1000);
        } else {
            sprintf(target_str, "%d x %d",
                    operations[i].target_rows,
                    operations[i].target_cols);
        }
        
        printf("| %-17s | %-10s | %9.3f | %9.3f | %6.2fx | %6d |\n",
               operations[i].name,
               target_str,
               res.omp_time_ms,
               res.seq_time_ms,
               res.speedup,
               verify ? res.errors : 0);
    }
    
    printf("+-------------------+------------+-----------+-----------+---------+--------+\n");
    printf("\n");
    
    if (verify) {
        printf("Verification: OpenMP results compared against sequential implementation\n");
    }
    
    free_csr_matrix(matrix);
    printf("\nDone!\n");
    
    return 0;
}

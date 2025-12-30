/**
 * ============================================================================
 * Parallel DCT-Based Sparse Matrix Scaling using OpenMP
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * ============================================================================
 * Compilation:
 *   Linux:   gcc -O3 -fopenmp -o parallel_dct_omp parallel_dct_omp.c -lm -lfftw3 -lfftw3_omp
 *   Windows: cl /O2 /openmp parallel_dct_omp.c (requires FFTW library)
 * 
 * Note: This version uses a simple DCT implementation without FFTW dependency
 *       for portability. For production, use FFTW for better performance.
 * 
 * Usage:
 *   ./parallel_dct_omp -s 500 -v
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    
    for (int i = 0; i < nnz; i++) {
        matrix->row_ptr[temp_rows[i] + 1]++;
    }
    
    for (int i = 0; i < size; i++) {
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];
    }
    
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
// DCT Functions (Simple Implementation)
// ============================================================================

// 1D DCT-II
void dct_1d(float* input, float* output, int n) {
    for (int k = 0; k < n; k++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += input[i] * cosf((float)M_PI * k * (2.0f * i + 1.0f) / (2.0f * n));
        }
        float alpha = (k == 0) ? sqrtf(1.0f / n) : sqrtf(2.0f / n);
        output[k] = alpha * sum;
    }
}

// 1D IDCT-II
void idct_1d(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float sum = input[0] * sqrtf(1.0f / n);
        for (int k = 1; k < n; k++) {
            sum += input[k] * sqrtf(2.0f / n) * 
                   cosf((float)M_PI * k * (2.0f * i + 1.0f) / (2.0f * n));
        }
        output[i] = sum;
    }
}

// 2D DCT (row-column decomposition)
void dct_2d(float* input, float* output, int rows, int cols) {
    float* temp = (float*)malloc(rows * cols * sizeof(float));
    float* row_in = (float*)malloc(cols * sizeof(float));
    float* row_out = (float*)malloc(cols * sizeof(float));
    
    // DCT on rows
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            row_in[j] = input[i * cols + j];
        }
        dct_1d(row_in, row_out, cols);
        for (int j = 0; j < cols; j++) {
            temp[i * cols + j] = row_out[j];
        }
    }
    
    // DCT on columns
    float* col_in = (float*)malloc(rows * sizeof(float));
    float* col_out = (float*)malloc(rows * sizeof(float));
    
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            col_in[i] = temp[i * cols + j];
        }
        dct_1d(col_in, col_out, rows);
        for (int i = 0; i < rows; i++) {
            output[i * cols + j] = col_out[i];
        }
    }
    
    free(temp);
    free(row_in);
    free(row_out);
    free(col_in);
    free(col_out);
}

// 2D IDCT
void idct_2d(float* input, float* output, int rows, int cols) {
    float* temp = (float*)malloc(rows * cols * sizeof(float));
    
    // IDCT on columns
    float* col_in = (float*)malloc(rows * sizeof(float));
    float* col_out = (float*)malloc(rows * sizeof(float));
    
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            col_in[i] = input[i * cols + j];
        }
        idct_1d(col_in, col_out, rows);
        for (int i = 0; i < rows; i++) {
            temp[i * cols + j] = col_out[i];
        }
    }
    
    // IDCT on rows
    float* row_in = (float*)malloc(cols * sizeof(float));
    float* row_out = (float*)malloc(cols * sizeof(float));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            row_in[j] = temp[i * cols + j];
        }
        idct_1d(row_in, row_out, cols);
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = row_out[j];
        }
    }
    
    free(temp);
    free(col_in);
    free(col_out);
    free(row_in);
    free(row_out);
}

// ============================================================================
// OpenMP Parallel DCT Functions
// ============================================================================

void omp_dct_2d(float* input, float* output, int rows, int cols) {
    float* temp = (float*)malloc(rows * cols * sizeof(float));
    
    // DCT on rows (parallel)
    #pragma omp parallel
    {
        float* row_in = (float*)malloc(cols * sizeof(float));
        float* row_out = (float*)malloc(cols * sizeof(float));
        
        #pragma omp for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                row_in[j] = input[i * cols + j];
            }
            dct_1d(row_in, row_out, cols);
            for (int j = 0; j < cols; j++) {
                temp[i * cols + j] = row_out[j];
            }
        }
        
        free(row_in);
        free(row_out);
    }
    
    // DCT on columns (parallel)
    #pragma omp parallel
    {
        float* col_in = (float*)malloc(rows * sizeof(float));
        float* col_out = (float*)malloc(rows * sizeof(float));
        
        #pragma omp for
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                col_in[i] = temp[i * cols + j];
            }
            dct_1d(col_in, col_out, rows);
            for (int i = 0; i < rows; i++) {
                output[i * cols + j] = col_out[i];
            }
        }
        
        free(col_in);
        free(col_out);
    }
    
    free(temp);
}

void omp_idct_2d(float* input, float* output, int rows, int cols) {
    float* temp = (float*)malloc(rows * cols * sizeof(float));
    
    // IDCT on columns (parallel)
    #pragma omp parallel
    {
        float* col_in = (float*)malloc(rows * sizeof(float));
        float* col_out = (float*)malloc(rows * sizeof(float));
        
        #pragma omp for
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                col_in[i] = input[i * cols + j];
            }
            idct_1d(col_in, col_out, rows);
            for (int i = 0; i < rows; i++) {
                temp[i * cols + j] = col_out[i];
            }
        }
        
        free(col_in);
        free(col_out);
    }
    
    // IDCT on rows (parallel)
    #pragma omp parallel
    {
        float* row_in = (float*)malloc(cols * sizeof(float));
        float* row_out = (float*)malloc(cols * sizeof(float));
        
        #pragma omp for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                row_in[j] = temp[i * cols + j];
            }
            idct_1d(row_in, row_out, cols);
            for (int j = 0; j < cols; j++) {
                output[i * cols + j] = row_out[j];
            }
        }
        
        free(row_in);
        free(row_out);
    }
    
    free(temp);
}

// ============================================================================
// Sequential DCT Scaling
// ============================================================================

int seq_dct_scale(CSRMatrix* input, int target_rows, int target_cols,
                  float threshold, int** out_rows, int** out_cols, float** out_values) {
    int in_rows = input->num_rows;
    int in_cols = input->num_cols;
    int in_size = in_rows * in_cols;
    int out_size = target_rows * target_cols;
    
    // Sparse to dense
    float* dense = (float*)calloc(in_size, sizeof(float));
    for (int row = 0; row < in_rows; row++) {
        for (int j = input->row_ptr[row]; j < input->row_ptr[row + 1]; j++) {
            int col = input->col_idx[j];
            dense[row * in_cols + col] = input->values[j];
        }
    }
    
    // Forward DCT
    float* freq = (float*)malloc(in_size * sizeof(float));
    dct_2d(dense, freq, in_rows, in_cols);
    
    // Resize coefficients
    float* freq_resized = (float*)calloc(out_size, sizeof(float));
    int copy_rows = (target_rows < in_rows) ? target_rows : in_rows;
    int copy_cols = (target_cols < in_cols) ? target_cols : in_cols;
    
    for (int i = 0; i < copy_rows; i++) {
        for (int j = 0; j < copy_cols; j++) {
            freq_resized[i * target_cols + j] = freq[i * in_cols + j];
        }
    }
    
    // Inverse DCT
    float* result = (float*)malloc(out_size * sizeof(float));
    idct_2d(freq_resized, result, target_rows, target_cols);
    
    // Count nonzeros and threshold
    int nnz = 0;
    for (int i = 0; i < out_size; i++) {
        if (fabsf(result[i]) > threshold) nnz++;
    }
    
    // Extract sparse output
    *out_rows = (int*)malloc(nnz * sizeof(int));
    *out_cols = (int*)malloc(nnz * sizeof(int));
    *out_values = (float*)malloc(nnz * sizeof(float));
    
    int idx = 0;
    for (int i = 0; i < target_rows; i++) {
        for (int j = 0; j < target_cols; j++) {
            float val = result[i * target_cols + j];
            if (fabsf(val) > threshold) {
                (*out_rows)[idx] = i;
                (*out_cols)[idx] = j;
                (*out_values)[idx] = val;
                idx++;
            }
        }
    }
    
    free(dense);
    free(freq);
    free(freq_resized);
    free(result);
    
    return nnz;
}

// ============================================================================
// OpenMP Parallel DCT Scaling
// ============================================================================

int omp_dct_scale(CSRMatrix* input, int target_rows, int target_cols,
                  float threshold, int** out_rows, int** out_cols, float** out_values) {
    int in_rows = input->num_rows;
    int in_cols = input->num_cols;
    int in_size = in_rows * in_cols;
    int out_size = target_rows * target_cols;
    
    // Sparse to dense (parallel)
    float* dense = (float*)calloc(in_size, sizeof(float));
    
    #pragma omp parallel for
    for (int row = 0; row < in_rows; row++) {
        for (int j = input->row_ptr[row]; j < input->row_ptr[row + 1]; j++) {
            int col = input->col_idx[j];
            dense[row * in_cols + col] = input->values[j];
        }
    }
    
    // Forward DCT (parallel)
    float* freq = (float*)malloc(in_size * sizeof(float));
    omp_dct_2d(dense, freq, in_rows, in_cols);
    
    // Resize coefficients (parallel)
    float* freq_resized = (float*)calloc(out_size, sizeof(float));
    int copy_rows = (target_rows < in_rows) ? target_rows : in_rows;
    int copy_cols = (target_cols < in_cols) ? target_cols : in_cols;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < copy_rows; i++) {
        for (int j = 0; j < copy_cols; j++) {
            freq_resized[i * target_cols + j] = freq[i * in_cols + j];
        }
    }
    
    // Inverse DCT (parallel)
    float* result = (float*)malloc(out_size * sizeof(float));
    omp_idct_2d(freq_resized, result, target_rows, target_cols);
    
    // Count nonzeros (parallel reduction)
    int nnz = 0;
    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < out_size; i++) {
        if (fabsf(result[i]) > threshold) nnz++;
    }
    
    // Mark nonzeros
    int* nnz_flags = (int*)malloc(out_size * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < out_size; i++) {
        nnz_flags[i] = (fabsf(result[i]) > threshold) ? 1 : 0;
    }
    
    // Prefix sum for output indices
    int* nnz_idx = (int*)malloc(out_size * sizeof(int));
    nnz_idx[0] = 0;
    for (int i = 1; i < out_size; i++) {
        nnz_idx[i] = nnz_idx[i-1] + nnz_flags[i-1];
    }
    
    // Extract sparse output (parallel)
    *out_rows = (int*)malloc(nnz * sizeof(int));
    *out_cols = (int*)malloc(nnz * sizeof(int));
    *out_values = (float*)malloc(nnz * sizeof(float));
    
    #pragma omp parallel for
    for (int i = 0; i < out_size; i++) {
        if (nnz_flags[i]) {
            int idx = nnz_idx[i];
            int row = i / target_cols;
            int col = i % target_cols;
            (*out_rows)[idx] = row;
            (*out_cols)[idx] = col;
            (*out_values)[idx] = result[i];
        }
    }
    
    free(dense);
    free(freq);
    free(freq_resized);
    free(result);
    free(nnz_flags);
    free(nnz_idx);
    
    return nnz;
}

// ============================================================================
// Benchmark
// ============================================================================

BenchmarkResult run_benchmark(CSRMatrix* input, int target_rows, int target_cols,
                              float threshold, int verify, int num_runs) {
    BenchmarkResult result = {0};
    
    int *omp_rows, *omp_cols;
    float *omp_vals;
    int omp_nnz;
    
    // Warmup
    omp_nnz = omp_dct_scale(input, target_rows, target_cols, threshold,
                            &omp_rows, &omp_cols, &omp_vals);
    free(omp_rows); free(omp_cols); free(omp_vals);
    
    // OpenMP timing
    double total_omp = 0;
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        omp_nnz = omp_dct_scale(input, target_rows, target_cols, threshold,
                                 &omp_rows, &omp_cols, &omp_vals);
        total_omp += (omp_get_wtime() - start) * 1000.0;
        
        if (r < num_runs - 1) {
            free(omp_rows); free(omp_cols); free(omp_vals);
        }
    }
    result.omp_time_ms = total_omp / num_runs;
    result.output_nnz = omp_nnz;
    
    // Sequential timing (if verify)
    if (verify) {
        int *seq_rows, *seq_cols;
        float *seq_vals;
        
        double start = omp_get_wtime();
        int seq_nnz = seq_dct_scale(input, target_rows, target_cols, threshold,
                                     &seq_rows, &seq_cols, &seq_vals);
        result.seq_time_ms = (omp_get_wtime() - start) * 1000.0;
        result.speedup = result.seq_time_ms / result.omp_time_ms;
        
        free(seq_rows); free(seq_cols); free(seq_vals);
    }
    
    free(omp_rows); free(omp_cols); free(omp_vals);
    
    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_header() {
    printf("\n");
    printf("==================================================================\n");
    printf("   Parallel DCT Sparse Matrix Scaling (OpenMP)\n");
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
    int size = 500;  // DCT is O(n^2) so smaller default
    int bandwidth = 10;
    int num_runs = 3;
    int verify = 0;
    int num_threads = 0;
    float threshold = 1e-6f;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            bandwidth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-th") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-s size] [-b bandwidth] [-r runs] [-t threads] [-th threshold] [-v]\n", argv[0]);
            printf("\nNote: DCT is O(n^2) so recommended size <= 1000\n");
            return 0;
        }
    }
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    print_header();
    print_system_info();
    
    printf("Generating banded matrix: %d x %d (bandwidth=%d)\n", size, size, bandwidth);
    CSRMatrix* matrix = generate_banded_matrix(size, bandwidth);
    printf("NNZ: %d, Density: %.6f\n", matrix->nnz,
           (float)matrix->nnz / ((float)size * size));
    printf("Threshold: %.2e\n\n", threshold);
    
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
    
    printf("Running benchmarks (%d runs each)...\n\n", num_runs);
    
    if (verify) {
        printf("+-------------------+------------+-----------+-----------+------------+---------+\n");
        printf("| Operation         | Target     | OMP (ms)  | Seq (ms)  | Output NNZ | Speedup |\n");
        printf("+-------------------+------------+-----------+-----------+------------+---------+\n");
    } else {
        printf("+-------------------+------------+-----------+------------+\n");
        printf("| Operation         | Target     | OMP (ms)  | Output NNZ |\n");
        printf("+-------------------+------------+-----------+------------+\n");
    }
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = run_benchmark(matrix,
                                           operations[i].target_rows,
                                           operations[i].target_cols,
                                           threshold, verify, num_runs);
        
        char target_str[20];
        sprintf(target_str, "%d x %d", operations[i].target_rows, operations[i].target_cols);
        
        if (verify) {
            printf("| %-17s | %-10s | %9.1f | %9.1f | %10d | %6.2fx |\n",
                   operations[i].name, target_str,
                   res.omp_time_ms, res.seq_time_ms,
                   res.output_nnz, res.speedup);
        } else {
            printf("| %-17s | %-10s | %9.1f | %10d |\n",
                   operations[i].name, target_str,
                   res.omp_time_ms, res.output_nnz);
        }
    }
    
    if (verify) {
        printf("+-------------------+------------+-----------+-----------+------------+---------+\n");
    } else {
        printf("+-------------------+------------+-----------+------------+\n");
    }
    
    printf("\nNote: This uses a simple O(n^2) DCT implementation.\n");
    printf("For production, use FFTW library for O(n log n) performance.\n");
    
    free_csr_matrix(matrix);
    printf("\nDone!\n");
    
    return 0;
}

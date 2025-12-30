/**
 * ============================================================================
 * Parallel Graph-Based Sparse Matrix Scaling using OpenMP
 * Heavy-Edge Matching (HEM) Approach
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * ============================================================================
 * Compilation:
 *   Linux:   gcc -O3 -fopenmp -o parallel_graph_hem_omp parallel_graph_hem_omp.c -lm
 *   Windows: cl /O2 /openmp parallel_graph_hem_omp.c
 * 
 * Usage:
 *   ./parallel_graph_hem_omp -s 1000 -v
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
    int num_clusters;
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
// Sequential Graph-HEM
// ============================================================================

int seq_graph_hem_scale(CSRMatrix* input, int target_rows, int target_cols,
                        int** out_rows, int** out_cols, float** out_values) {
    int num_rows = input->num_rows;
    int num_cols = input->num_cols;
    int nnz = input->nnz;
    
    // Step 1: Build node list
    int* node_rows = (int*)malloc(nnz * sizeof(int));
    int* node_cols = (int*)malloc(nnz * sizeof(int));
    float* node_values = (float*)malloc(nnz * sizeof(float));
    
    int idx = 0;
    for (int row = 0; row < num_rows; row++) {
        for (int j = input->row_ptr[row]; j < input->row_ptr[row + 1]; j++) {
            node_rows[idx] = row;
            node_cols[idx] = input->col_idx[j];
            node_values[idx] = input->values[j];
            idx++;
        }
    }
    
    // Step 2: Create position map
    int* pos_to_node = (int*)malloc(num_rows * num_cols * sizeof(int));
    for (int i = 0; i < num_rows * num_cols; i++) pos_to_node[i] = -1;
    
    for (int i = 0; i < nnz; i++) {
        int pos = node_rows[i] * num_cols + node_cols[i];
        pos_to_node[pos] = i;
    }
    
    // Step 3: Build adjacency
    int* adj_ptr = (int*)calloc(nnz + 1, sizeof(int));
    
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        int count = 0;
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                if (pos_to_node[npos] >= 0) count++;
            }
        }
        adj_ptr[i + 1] = count;
    }
    
    for (int i = 0; i < nnz; i++) {
        adj_ptr[i + 1] += adj_ptr[i];
    }
    
    int total_edges = adj_ptr[nnz];
    int* adj_list = (int*)malloc(total_edges * sizeof(int));
    float* adj_weights = (float*)malloc(total_edges * sizeof(float));
    
    int* edge_idx = (int*)calloc(nnz, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        edge_idx[i] = adj_ptr[i];
    }
    
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        float val = node_values[i];
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                int neighbor = pos_to_node[npos];
                if (neighbor >= 0) {
                    adj_list[edge_idx[i]] = neighbor;
                    adj_weights[edge_idx[i]] = fabsf(val) + fabsf(node_values[neighbor]);
                    edge_idx[i]++;
                }
            }
        }
    }
    
    // Step 4: HEM matching
    int* match = (int*)malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; i++) match[i] = -1;
    
    for (int i = 0; i < nnz; i++) {
        if (match[i] >= 0) continue;
        
        int best = -1;
        float best_weight = -1.0f;
        
        for (int e = adj_ptr[i]; e < adj_ptr[i + 1]; e++) {
            int neighbor = adj_list[e];
            if (match[neighbor] < 0) {
                float w = adj_weights[e];
                if (w > best_weight) {
                    best_weight = w;
                    best = neighbor;
                }
            }
        }
        
        if (best >= 0) {
            match[i] = best;
            match[best] = i;
        }
    }
    
    // Step 5: Assign clusters
    int* cluster_id = (int*)malloc(nnz * sizeof(int));
    int num_clusters = 0;
    
    for (int i = 0; i < nnz; i++) {
        int partner = match[i];
        if (partner < 0) {
            cluster_id[i] = num_clusters++;
        } else if (i < partner) {
            cluster_id[i] = num_clusters;
            cluster_id[partner] = num_clusters;
            num_clusters++;
        }
    }
    
    // Step 6: Compute cluster info and scale
    float* cluster_row_sum = (float*)calloc(num_clusters, sizeof(float));
    float* cluster_col_sum = (float*)calloc(num_clusters, sizeof(float));
    float* cluster_val_sum = (float*)calloc(num_clusters, sizeof(float));
    int* cluster_count = (int*)calloc(num_clusters, sizeof(int));
    
    for (int i = 0; i < nnz; i++) {
        int cid = cluster_id[i];
        cluster_row_sum[cid] += node_rows[i];
        cluster_col_sum[cid] += node_cols[i];
        cluster_val_sum[cid] += node_values[i];
        cluster_count[cid]++;
    }
    
    // Allocate output
    *out_rows = (int*)malloc(num_clusters * sizeof(int));
    *out_cols = (int*)malloc(num_clusters * sizeof(int));
    *out_values = (float*)malloc(num_clusters * sizeof(float));
    
    float scale_r = (float)(target_rows - 1) / (num_rows - 1);
    float scale_c = (float)(target_cols - 1) / (num_cols - 1);
    
    for (int c = 0; c < num_clusters; c++) {
        if (cluster_count[c] > 0) {
            float centroid_row = cluster_row_sum[c] / cluster_count[c];
            float centroid_col = cluster_col_sum[c] / cluster_count[c];
            
            int new_row = (int)roundf(centroid_row * scale_r);
            int new_col = (int)roundf(centroid_col * scale_c);
            
            new_row = new_row < 0 ? 0 : (new_row >= target_rows ? target_rows - 1 : new_row);
            new_col = new_col < 0 ? 0 : (new_col >= target_cols ? target_cols - 1 : new_col);
            
            (*out_rows)[c] = new_row;
            (*out_cols)[c] = new_col;
            (*out_values)[c] = cluster_val_sum[c] / cluster_count[c];
        }
    }
    
    // Cleanup
    free(node_rows);
    free(node_cols);
    free(node_values);
    free(pos_to_node);
    free(adj_ptr);
    free(adj_list);
    free(adj_weights);
    free(edge_idx);
    free(match);
    free(cluster_id);
    free(cluster_row_sum);
    free(cluster_col_sum);
    free(cluster_val_sum);
    free(cluster_count);
    
    return num_clusters;
}

// ============================================================================
// OpenMP Parallel Graph-HEM
// ============================================================================

int omp_graph_hem_scale(CSRMatrix* input, int target_rows, int target_cols,
                        int** out_rows, int** out_cols, float** out_values) {
    int num_rows = input->num_rows;
    int num_cols = input->num_cols;
    int nnz = input->nnz;
    
    // Step 1: Build node list (parallel)
    int* node_rows = (int*)malloc(nnz * sizeof(int));
    int* node_cols = (int*)malloc(nnz * sizeof(int));
    float* node_values = (float*)malloc(nnz * sizeof(float));
    
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
        node_rows[i] = low;
        node_cols[i] = input->col_idx[i];
        node_values[i] = input->values[i];
    }
    
    // Step 2: Create position map (parallel)
    int map_size = num_rows * num_cols;
    int* pos_to_node = (int*)malloc(map_size * sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < map_size; i++) {
        pos_to_node[i] = -1;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int pos = node_rows[i] * num_cols + node_cols[i];
        pos_to_node[pos] = i;
    }
    
    // Step 3: Count edges (parallel)
    int* edge_counts = (int*)malloc(nnz * sizeof(int));
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        int count = 0;
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                if (pos_to_node[npos] >= 0) count++;
            }
        }
        edge_counts[i] = count;
    }
    
    // Prefix sum (sequential - small overhead)
    int* adj_ptr = (int*)malloc((nnz + 1) * sizeof(int));
    adj_ptr[0] = 0;
    for (int i = 0; i < nnz; i++) {
        adj_ptr[i + 1] = adj_ptr[i] + edge_counts[i];
    }
    
    int total_edges = adj_ptr[nnz];
    int* adj_list = (int*)malloc(total_edges * sizeof(int));
    float* adj_weights = (float*)malloc(total_edges * sizeof(float));
    
    // Build adjacency (parallel)
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        float val = node_values[i];
        int idx = adj_ptr[i];
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                int neighbor = pos_to_node[npos];
                if (neighbor >= 0) {
                    adj_list[idx] = neighbor;
                    adj_weights[idx] = fabsf(val) + fabsf(node_values[neighbor]);
                    idx++;
                }
            }
        }
    }
    
    // Step 4: Find best match (parallel)
    int* best_match = (int*)malloc(nnz * sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int best = -1;
        float best_weight = -1.0f;
        
        for (int e = adj_ptr[i]; e < adj_ptr[i + 1]; e++) {
            int neighbor = adj_list[e];
            float w = adj_weights[e];
            if (w > best_weight) {
                best_weight = w;
                best = neighbor;
            }
        }
        best_match[i] = best;
    }
    
    // Step 5: Parallel matching (multiple iterations)
    int* match = (int*)malloc(nnz * sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        match[i] = -1;
    }
    
    for (int iter = 0; iter < 3; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < nnz; i++) {
            if (match[i] >= 0) continue;
            
            int partner = best_match[i];
            if (partner < 0) continue;
            if (match[partner] >= 0) continue;
            
            // Only lower ID initiates
            if (i < partner && best_match[partner] == i) {
                // Try to match (race condition possible but okay for HEM)
                #pragma omp critical
                {
                    if (match[i] < 0 && match[partner] < 0) {
                        match[i] = partner;
                        match[partner] = i;
                    }
                }
            }
        }
    }
    
    // Step 6: Assign clusters (parallel prefix)
    int* is_cluster_leader = (int*)calloc(nnz, sizeof(int));
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int partner = match[i];
        if (partner < 0 || i < partner) {
            is_cluster_leader[i] = 1;
        }
    }
    
    // Prefix sum for cluster IDs
    int* cluster_id = (int*)malloc(nnz * sizeof(int));
    int num_clusters = 0;
    for (int i = 0; i < nnz; i++) {
        if (is_cluster_leader[i]) {
            cluster_id[i] = num_clusters++;
            int partner = match[i];
            if (partner >= 0) {
                cluster_id[partner] = cluster_id[i];
            }
        }
    }
    
    // Step 7: Compute cluster info (parallel with reduction)
    float* cluster_row_sum = (float*)calloc(num_clusters, sizeof(float));
    float* cluster_col_sum = (float*)calloc(num_clusters, sizeof(float));
    float* cluster_val_sum = (float*)calloc(num_clusters, sizeof(float));
    int* cluster_count = (int*)calloc(num_clusters, sizeof(int));
    
    #pragma omp parallel
    {
        float* local_row = (float*)calloc(num_clusters, sizeof(float));
        float* local_col = (float*)calloc(num_clusters, sizeof(float));
        float* local_val = (float*)calloc(num_clusters, sizeof(float));
        int* local_cnt = (int*)calloc(num_clusters, sizeof(int));
        
        #pragma omp for nowait
        for (int i = 0; i < nnz; i++) {
            int cid = cluster_id[i];
            local_row[cid] += node_rows[i];
            local_col[cid] += node_cols[i];
            local_val[cid] += node_values[i];
            local_cnt[cid]++;
        }
        
        #pragma omp critical
        {
            for (int c = 0; c < num_clusters; c++) {
                cluster_row_sum[c] += local_row[c];
                cluster_col_sum[c] += local_col[c];
                cluster_val_sum[c] += local_val[c];
                cluster_count[c] += local_cnt[c];
            }
        }
        
        free(local_row);
        free(local_col);
        free(local_val);
        free(local_cnt);
    }
    
    // Step 8: Scale clusters (parallel)
    *out_rows = (int*)malloc(num_clusters * sizeof(int));
    *out_cols = (int*)malloc(num_clusters * sizeof(int));
    *out_values = (float*)malloc(num_clusters * sizeof(float));
    
    float scale_r = (float)(target_rows - 1) / (num_rows - 1);
    float scale_c = (float)(target_cols - 1) / (num_cols - 1);
    
    #pragma omp parallel for
    for (int c = 0; c < num_clusters; c++) {
        if (cluster_count[c] > 0) {
            float centroid_row = cluster_row_sum[c] / cluster_count[c];
            float centroid_col = cluster_col_sum[c] / cluster_count[c];
            
            int new_row = (int)roundf(centroid_row * scale_r);
            int new_col = (int)roundf(centroid_col * scale_c);
            
            new_row = new_row < 0 ? 0 : (new_row >= target_rows ? target_rows - 1 : new_row);
            new_col = new_col < 0 ? 0 : (new_col >= target_cols ? target_cols - 1 : new_col);
            
            (*out_rows)[c] = new_row;
            (*out_cols)[c] = new_col;
            (*out_values)[c] = cluster_val_sum[c] / cluster_count[c];
        }
    }
    
    // Cleanup
    free(node_rows);
    free(node_cols);
    free(node_values);
    free(pos_to_node);
    free(edge_counts);
    free(adj_ptr);
    free(adj_list);
    free(adj_weights);
    free(best_match);
    free(match);
    free(is_cluster_leader);
    free(cluster_id);
    free(cluster_row_sum);
    free(cluster_col_sum);
    free(cluster_val_sum);
    free(cluster_count);
    
    return num_clusters;
}

// ============================================================================
// Benchmark
// ============================================================================

BenchmarkResult run_benchmark(CSRMatrix* input, int target_rows, int target_cols,
                              int verify, int num_runs) {
    BenchmarkResult result = {0};
    
    int *omp_rows, *omp_cols;
    float *omp_vals;
    int omp_nnz;
    
    // Warmup
    omp_nnz = omp_graph_hem_scale(input, target_rows, target_cols,
                                   &omp_rows, &omp_cols, &omp_vals);
    free(omp_rows); free(omp_cols); free(omp_vals);
    
    // OpenMP timing
    double total_omp = 0;
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        omp_nnz = omp_graph_hem_scale(input, target_rows, target_cols,
                                       &omp_rows, &omp_cols, &omp_vals);
        total_omp += (omp_get_wtime() - start) * 1000.0;
        
        if (r < num_runs - 1) {
            free(omp_rows); free(omp_cols); free(omp_vals);
        }
    }
    result.omp_time_ms = total_omp / num_runs;
    result.num_clusters = omp_nnz;
    result.output_nnz = omp_nnz;
    
    // Sequential timing (if verify)
    if (verify) {
        int *seq_rows, *seq_cols;
        float *seq_vals;
        
        double start = omp_get_wtime();
        int seq_nnz = seq_graph_hem_scale(input, target_rows, target_cols,
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
    printf("   Parallel Graph-Based Sparse Matrix Scaling (OpenMP + HEM)\n");
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
    int size = 1000;
    int bandwidth = 5;
    int num_runs = 5;
    int verify = 0;
    int num_threads = 0;
    
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
    
    printf("Generating banded matrix: %d x %d (bandwidth=%d)\n", size, size, bandwidth);
    CSRMatrix* matrix = generate_banded_matrix(size, bandwidth);
    printf("Input NNZ: %d, Density: %.6f\n\n", matrix->nnz,
           (float)matrix->nnz / ((float)size * size));
    
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
        printf("+-------------------+------------+-----------+-----------+----------+---------+\n");
        printf("| Operation         | Target     | OMP (ms)  | Seq (ms)  | Clusters | Speedup |\n");
        printf("+-------------------+------------+-----------+-----------+----------+---------+\n");
    } else {
        printf("+-------------------+------------+-----------+----------+-----------+\n");
        printf("| Operation         | Target     | OMP (ms)  | Clusters | Out NNZ   |\n");
        printf("+-------------------+------------+-----------+----------+-----------+\n");
    }
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = run_benchmark(matrix,
                                           operations[i].target_rows,
                                           operations[i].target_cols,
                                           verify, num_runs);
        
        char target_str[20];
        sprintf(target_str, "%d x %d", operations[i].target_rows, operations[i].target_cols);
        
        if (verify) {
            printf("| %-17s | %-10s | %9.3f | %9.3f | %8d | %6.2fx |\n",
                   operations[i].name, target_str,
                   res.omp_time_ms, res.seq_time_ms,
                   res.num_clusters, res.speedup);
        } else {
            printf("| %-17s | %-10s | %9.3f | %8d | %9d |\n",
                   operations[i].name, target_str,
                   res.omp_time_ms, res.num_clusters, res.output_nnz);
        }
    }
    
    if (verify) {
        printf("+-------------------+------------+-----------+-----------+----------+---------+\n");
    } else {
        printf("+-------------------+------------+-----------+----------+-----------+\n");
    }
    
    printf("\nAlgorithm: Graph-based Heavy-Edge Matching (HEM)\n");
    printf("Connectivity: 4-connectivity (Von Neumann neighborhood)\n");
    
    free_csr_matrix(matrix);
    printf("\nDone!\n");
    
    return 0;
}

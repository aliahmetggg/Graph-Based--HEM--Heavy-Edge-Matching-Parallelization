/**
 * ============================================================================
 * Parallel Graph-Based Sparse Matrix Scaling using CUDA
 * Heavy-Edge Matching (HEM) Approach
 * ============================================================================
 * 
 * Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
 * Course: Parallel Programming
 * University: Ankara Yıldırım Beyazıt University
 * 
 * Novel Approach:
 *   Instead of signal/image processing techniques (NN, DCT), we treat the
 *   sparse matrix as a graph and use Heavy-Edge Matching for coarsening
 *   (downscaling) and refinement (upscaling).
 * 
 * Algorithm:
 *   1. Convert sparse matrix to graph (4-connectivity)
 *   2. Apply HEM for clustering
 *   3. Map clusters to new matrix dimensions
 *   4. Generate scaled sparse matrix
 * 
 * ============================================================================
 * Compilation:
 *   nvcc -O3 -arch=sm_70 -o parallel_graph_hem parallel_graph_hem.cu
 * 
 * Usage:
 *   ./parallel_graph_hem                    # Default 1000x1000 banded matrix
 *   ./parallel_graph_hem -s 2000            # 2000x2000 matrix
 *   ./parallel_graph_hem -s 1000 -b 5       # bandwidth=5
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
    cudaEvent_t start, stop;
} GpuTimer;

typedef struct {
    clock_t start, stop;
} CpuTimer;

void gpu_timer_init(GpuTimer* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

void gpu_timer_start(GpuTimer* t) {
    cudaEventRecord(t->start, 0);
}

float gpu_timer_stop(GpuTimer* t) {
    float ms;
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

void gpu_timer_destroy(GpuTimer* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

void cpu_timer_start(CpuTimer* t) { t->start = clock(); }
float cpu_timer_stop(CpuTimer* t) {
    t->stop = clock();
    return ((float)(t->stop - t->start) / CLOCKS_PER_SEC) * 1000.0f;
}

// ============================================================================
// Data Structures
// ============================================================================

// CSR Matrix
typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int* row_ptr;
    int* col_idx;
    float* values;
} CSRMatrix;

// Graph Node (represents a nonzero entry)
typedef struct {
    int row;
    int col;
    float value;
    int node_id;
} GraphNode;

// Graph Edge (4-connectivity between adjacent nonzeros)
typedef struct {
    int src;        // source node id
    int dst;        // destination node id
    float weight;   // edge weight (based on value similarity)
} GraphEdge;

// Graph structure for sparse matrix
typedef struct {
    int num_nodes;
    int num_edges;
    GraphNode* nodes;
    GraphEdge* edges;
    int* adj_ptr;       // CSR-style adjacency: pointer to start of each node's edges
    int* adj_list;      // adjacency list (destination nodes)
    float* adj_weights; // edge weights
} SparseGraph;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel: Build node list from sparse matrix
 * Each thread processes one nonzero element
 */
__global__ void build_nodes_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    int* __restrict__ node_rows,
    int* __restrict__ node_cols,
    float* __restrict__ node_values,
    int num_rows,
    int nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        // Binary search to find row
        int low = 0, high = num_rows;
        while (low < high) {
            int mid = (low + high) / 2;
            if (row_ptr[mid + 1] <= tid) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        node_rows[tid] = low;
        node_cols[tid] = col_idx[tid];
        node_values[tid] = values[tid];
    }
}

/**
 * Kernel: Create position-to-node mapping
 * Maps (row, col) -> node_id for quick neighbor lookup
 */
__global__ void create_position_map_kernel(
    const int* __restrict__ node_rows,
    const int* __restrict__ node_cols,
    int* __restrict__ pos_to_node,
    int nnz,
    int num_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int row = node_rows[tid];
        int col = node_cols[tid];
        int pos = row * num_cols + col;
        pos_to_node[pos] = tid;  // node_id = tid
    }
}

/**
 * Kernel: Count edges for each node (4-connectivity)
 * Each thread processes one node
 */
__global__ void count_edges_kernel(
    const int* __restrict__ node_rows,
    const int* __restrict__ node_cols,
    const int* __restrict__ pos_to_node,
    int* __restrict__ edge_counts,
    int nnz,
    int num_rows,
    int num_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int row = node_rows[tid];
        int col = node_cols[tid];
        int count = 0;
        
        // 4-connectivity: up, down, left, right
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                if (pos_to_node[npos] >= 0) {
                    count++;
                }
            }
        }
        
        edge_counts[tid] = count;
    }
}

/**
 * Kernel: Build adjacency list
 */
__global__ void build_adjacency_kernel(
    const int* __restrict__ node_rows,
    const int* __restrict__ node_cols,
    const float* __restrict__ node_values,
    const int* __restrict__ pos_to_node,
    const int* __restrict__ adj_ptr,
    int* __restrict__ adj_list,
    float* __restrict__ adj_weights,
    int nnz,
    int num_rows,
    int num_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int row = node_rows[tid];
        int col = node_cols[tid];
        float val = node_values[tid];
        
        int edge_idx = adj_ptr[tid];
        
        // 4-connectivity
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                int neighbor_id = pos_to_node[npos];
                
                if (neighbor_id >= 0) {
                    adj_list[edge_idx] = neighbor_id;
                    // Edge weight: similarity between values
                    float neighbor_val = node_values[neighbor_id];
                    adj_weights[edge_idx] = fabsf(val) + fabsf(neighbor_val);
                    edge_idx++;
                }
            }
        }
    }
}

/**
 * Kernel: Heavy-Edge Matching (HEM)
 * Each node tries to match with its heaviest unmatched neighbor
 * Uses parallel randomized matching for efficiency
 */
__global__ void hem_matching_kernel(
    const int* __restrict__ adj_ptr,
    const int* __restrict__ adj_list,
    const float* __restrict__ adj_weights,
    int* __restrict__ match,          // match[i] = matched partner of node i (-1 if unmatched)
    const int* __restrict__ node_rows,
    const int* __restrict__ node_cols,
    int nnz,
    unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        // Skip if already matched
        if (match[tid] >= 0) return;
        
        int start = adj_ptr[tid];
        int end = adj_ptr[tid + 1];
        
        // Find heaviest unmatched neighbor
        int best_neighbor = -1;
        float best_weight = -1.0f;
        
        for (int e = start; e < end; e++) {
            int neighbor = adj_list[e];
            
            // Check if neighbor is unmatched
            if (match[neighbor] < 0) {
                float weight = adj_weights[e];
                if (weight > best_weight) {
                    best_weight = weight;
                    best_neighbor = neighbor;
                }
            }
        }
        
        // Try to match with best neighbor (use atomicCAS for thread safety)
        if (best_neighbor >= 0 && best_neighbor > tid) {
            // Only lower-id node initiates matching to avoid conflicts
            int expected = -1;
            int old = atomicCAS(&match[best_neighbor], expected, tid);
            
            if (old == -1) {
                // Successfully matched
                match[tid] = best_neighbor;
            }
        }
    }
}

/**
 * Kernel: Assign cluster IDs based on matching
 * Matched pairs get same cluster, unmatched nodes get unique cluster
 */
__global__ void assign_clusters_kernel(
    const int* __restrict__ match,
    int* __restrict__ cluster_id,
    int* __restrict__ cluster_counter,
    int nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int partner = match[tid];
        
        if (partner < 0) {
            // Unmatched: get new cluster ID
            cluster_id[tid] = atomicAdd(cluster_counter, 1);
        } else if (tid < partner) {
            // Matched pair: lower ID node assigns cluster
            int cid = atomicAdd(cluster_counter, 1);
            cluster_id[tid] = cid;
            cluster_id[partner] = cid;
        }
        // Note: if tid > partner, cluster was already assigned by partner
    }
}

/**
 * Kernel: Compute cluster centroids and values
 */
__global__ void compute_cluster_info_kernel(
    const int* __restrict__ node_rows,
    const int* __restrict__ node_cols,
    const float* __restrict__ node_values,
    const int* __restrict__ cluster_id,
    float* __restrict__ cluster_row_sum,
    float* __restrict__ cluster_col_sum,
    float* __restrict__ cluster_val_sum,
    int* __restrict__ cluster_count,
    int nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nnz) {
        int cid = cluster_id[tid];
        
        atomicAdd(&cluster_row_sum[cid], (float)node_rows[tid]);
        atomicAdd(&cluster_col_sum[cid], (float)node_cols[tid]);
        atomicAdd(&cluster_val_sum[cid], node_values[tid]);
        atomicAdd(&cluster_count[cid], 1);
    }
}

/**
 * Kernel: Scale cluster centroids to target dimensions
 */
__global__ void scale_clusters_kernel(
    const float* __restrict__ cluster_row_sum,
    const float* __restrict__ cluster_col_sum,
    const float* __restrict__ cluster_val_sum,
    const int* __restrict__ cluster_count,
    int* __restrict__ out_rows,
    int* __restrict__ out_cols,
    float* __restrict__ out_values,
    int num_clusters,
    int src_rows,
    int src_cols,
    int target_rows,
    int target_cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_clusters) {
        int count = cluster_count[tid];
        
        if (count > 0) {
            // Compute centroid
            float centroid_row = cluster_row_sum[tid] / count;
            float centroid_col = cluster_col_sum[tid] / count;
            float total_val = cluster_val_sum[tid];
            
            // Scale to target dimensions
            float scale_r = (float)(target_rows - 1) / (src_rows - 1);
            float scale_c = (float)(target_cols - 1) / (src_cols - 1);
            
            int new_row = (int)roundf(centroid_row * scale_r);
            int new_col = (int)roundf(centroid_col * scale_c);
            
            // Clip to valid range
            new_row = min(max(new_row, 0), target_rows - 1);
            new_col = min(max(new_col, 0), target_cols - 1);
            
            out_rows[tid] = new_row;
            out_cols[tid] = new_col;
            out_values[tid] = total_val / count;  // Average value
        } else {
            out_rows[tid] = -1;  // Invalid marker
            out_cols[tid] = -1;
            out_values[tid] = 0.0f;
        }
    }
}

// ============================================================================
// CPU Sequential Implementation (for comparison)
// ============================================================================

void cpu_graph_hem_scale(CSRMatrix* input, int target_rows, int target_cols,
                         int** out_rows, int** out_cols, float** out_values, 
                         int* out_nnz) {
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
    
    // Step 3: Build adjacency (4-connectivity)
    int* adj_ptr = (int*)calloc(nnz + 1, sizeof(int));
    
    // Count edges
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        int count = 0;
        
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        
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
    
    // Prefix sum
    for (int i = 0; i < nnz; i++) {
        adj_ptr[i + 1] += adj_ptr[i];
    }
    
    int num_edges = adj_ptr[nnz];
    int* adj_list = (int*)malloc(num_edges * sizeof(int));
    float* adj_weights = (float*)malloc(num_edges * sizeof(float));
    
    // Fill adjacency
    int* temp_ptr = (int*)malloc(nnz * sizeof(int));
    memcpy(temp_ptr, adj_ptr, nnz * sizeof(int));
    
    for (int i = 0; i < nnz; i++) {
        int row = node_rows[i];
        int col = node_cols[i];
        float val = node_values[i];
        
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        
        for (int d = 0; d < 4; d++) {
            int nr = row + dr[d];
            int nc = col + dc[d];
            if (nr >= 0 && nr < num_rows && nc >= 0 && nc < num_cols) {
                int npos = nr * num_cols + nc;
                int neighbor = pos_to_node[npos];
                if (neighbor >= 0) {
                    int e = temp_ptr[i]++;
                    adj_list[e] = neighbor;
                    adj_weights[e] = fabsf(val) + fabsf(node_values[neighbor]);
                }
            }
        }
    }
    free(temp_ptr);
    
    // Step 4: Heavy-Edge Matching
    int* match = (int*)malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; i++) match[i] = -1;
    
    // Simple greedy HEM
    for (int i = 0; i < nnz; i++) {
        if (match[i] >= 0) continue;
        
        int best_neighbor = -1;
        float best_weight = -1.0f;
        
        for (int e = adj_ptr[i]; e < adj_ptr[i + 1]; e++) {
            int neighbor = adj_list[e];
            if (match[neighbor] < 0 && adj_weights[e] > best_weight) {
                best_weight = adj_weights[e];
                best_neighbor = neighbor;
            }
        }
        
        if (best_neighbor >= 0) {
            match[i] = best_neighbor;
            match[best_neighbor] = i;
        }
    }
    
    // Step 5: Assign cluster IDs
    int* cluster_id = (int*)malloc(nnz * sizeof(int));
    int num_clusters = 0;
    
    for (int i = 0; i < nnz; i++) cluster_id[i] = -1;
    
    for (int i = 0; i < nnz; i++) {
        if (cluster_id[i] >= 0) continue;
        
        cluster_id[i] = num_clusters;
        if (match[i] >= 0) {
            cluster_id[match[i]] = num_clusters;
        }
        num_clusters++;
    }
    
    // Step 6: Compute cluster centroids
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
    
    // Step 7: Scale clusters to target dimensions
    *out_rows = (int*)malloc(num_clusters * sizeof(int));
    *out_cols = (int*)malloc(num_clusters * sizeof(int));
    *out_values = (float*)malloc(num_clusters * sizeof(float));
    
    float scale_r = (float)(target_rows - 1) / (num_rows - 1);
    float scale_c = (float)(target_cols - 1) / (num_cols - 1);
    
    int valid_count = 0;
    for (int i = 0; i < num_clusters; i++) {
        if (cluster_count[i] > 0) {
            float centroid_row = cluster_row_sum[i] / cluster_count[i];
            float centroid_col = cluster_col_sum[i] / cluster_count[i];
            
            int new_row = (int)roundf(centroid_row * scale_r);
            int new_col = (int)roundf(centroid_col * scale_c);
            
            new_row = new_row < 0 ? 0 : (new_row >= target_rows ? target_rows - 1 : new_row);
            new_col = new_col < 0 ? 0 : (new_col >= target_cols ? target_cols - 1 : new_col);
            
            (*out_rows)[valid_count] = new_row;
            (*out_cols)[valid_count] = new_col;
            (*out_values)[valid_count] = cluster_val_sum[i] / cluster_count[i];
            valid_count++;
        }
    }
    
    *out_nnz = valid_count;
    
    // Cleanup
    free(node_rows); free(node_cols); free(node_values);
    free(pos_to_node);
    free(adj_ptr); free(adj_list); free(adj_weights);
    free(match); free(cluster_id);
    free(cluster_row_sum); free(cluster_col_sum);
    free(cluster_val_sum); free(cluster_count);
}

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
// GPU Benchmark Function
// ============================================================================

typedef struct {
    float gpu_time_ms;
    float cpu_time_ms;
    float speedup;
    int input_nnz;
    int output_nnz;
    int num_clusters;
    int errors;
} BenchmarkResult;

BenchmarkResult run_benchmark(CSRMatrix* input, int target_rows, int target_cols,
                              int verify, int num_runs) {
    BenchmarkResult result = {0};
    
    int num_rows = input->num_rows;
    int num_cols = input->num_cols;
    int nnz = input->nnz;
    int total_size = num_rows * num_cols;
    
    result.input_nnz = nnz;
    
    // ========== Device Memory Allocation ==========
    int *d_row_ptr, *d_col_idx;
    float *d_values;
    int *d_node_rows, *d_node_cols;
    float *d_node_values;
    int *d_pos_to_node;
    int *d_edge_counts, *d_adj_ptr, *d_adj_list;
    float *d_adj_weights;
    int *d_match, *d_cluster_id, *d_cluster_counter;
    float *d_cluster_row_sum, *d_cluster_col_sum, *d_cluster_val_sum;
    int *d_cluster_count;
    int *d_out_rows, *d_out_cols;
    float *d_out_values;
    
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_node_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_to_node, total_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_counts, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_ptr, (nnz + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_match, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_id, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counter, sizeof(int)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, input->row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, input->col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, input->values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize pos_to_node to -1
    CUDA_CHECK(cudaMemset(d_pos_to_node, -1, total_size * sizeof(int)));
    
    int block_size = 256;
    int grid_nnz = (nnz + block_size - 1) / block_size;
    
    GpuTimer timer;
    gpu_timer_init(&timer);
    
    float total_gpu_time = 0;
    int final_num_clusters = 0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset for each run
        CUDA_CHECK(cudaMemset(d_pos_to_node, -1, total_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_match, -1, nnz * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cluster_counter, 0, sizeof(int)));
        
        gpu_timer_start(&timer);
        
        // Step 1: Build nodes
        build_nodes_kernel<<<grid_nnz, block_size>>>(
            d_row_ptr, d_col_idx, d_values,
            d_node_rows, d_node_cols, d_node_values,
            num_rows, nnz);
        
        // Step 2: Create position map
        create_position_map_kernel<<<grid_nnz, block_size>>>(
            d_node_rows, d_node_cols, d_pos_to_node, nnz, num_cols);
        
        // Step 3: Count edges
        count_edges_kernel<<<grid_nnz, block_size>>>(
            d_node_rows, d_node_cols, d_pos_to_node,
            d_edge_counts, nnz, num_rows, num_cols);
        
        // Step 4: Prefix sum for adjacency pointers (simple CPU version for now)
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int* h_edge_counts = (int*)malloc(nnz * sizeof(int));
        int* h_adj_ptr = (int*)malloc((nnz + 1) * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_edge_counts, d_edge_counts, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        
        h_adj_ptr[0] = 0;
        for (int i = 0; i < nnz; i++) {
            h_adj_ptr[i + 1] = h_adj_ptr[i] + h_edge_counts[i];
        }
        int num_edges = h_adj_ptr[nnz];
        
        CUDA_CHECK(cudaMemcpy(d_adj_ptr, h_adj_ptr, (nnz + 1) * sizeof(int), cudaMemcpyHostToDevice));
        
        // Allocate adjacency arrays
        CUDA_CHECK(cudaMalloc(&d_adj_list, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_adj_weights, num_edges * sizeof(float)));
        
        // Step 5: Build adjacency
        build_adjacency_kernel<<<grid_nnz, block_size>>>(
            d_node_rows, d_node_cols, d_node_values,
            d_pos_to_node, d_adj_ptr,
            d_adj_list, d_adj_weights,
            nnz, num_rows, num_cols);
        
        // Step 6: HEM Matching (multiple iterations for better matching)
        for (int iter = 0; iter < 3; iter++) {
            hem_matching_kernel<<<grid_nnz, block_size>>>(
                d_adj_ptr, d_adj_list, d_adj_weights,
                d_match, d_node_rows, d_node_cols, nnz, iter * 12345);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Step 7: Assign clusters
        assign_clusters_kernel<<<grid_nnz, block_size>>>(
            d_match, d_cluster_id, d_cluster_counter, nnz);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get number of clusters
        int h_num_clusters;
        CUDA_CHECK(cudaMemcpy(&h_num_clusters, d_cluster_counter, sizeof(int), cudaMemcpyDeviceToHost));
        final_num_clusters = h_num_clusters;
        
        // Allocate cluster arrays
        CUDA_CHECK(cudaMalloc(&d_cluster_row_sum, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cluster_col_sum, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cluster_val_sum, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cluster_count, h_num_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_rows, h_num_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_cols, h_num_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_values, h_num_clusters * sizeof(float)));
        
        CUDA_CHECK(cudaMemset(d_cluster_row_sum, 0, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cluster_col_sum, 0, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cluster_val_sum, 0, h_num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_cluster_count, 0, h_num_clusters * sizeof(int)));
        
        // Step 8: Compute cluster info
        compute_cluster_info_kernel<<<grid_nnz, block_size>>>(
            d_node_rows, d_node_cols, d_node_values,
            d_cluster_id,
            d_cluster_row_sum, d_cluster_col_sum, d_cluster_val_sum,
            d_cluster_count, nnz);
        
        // Step 9: Scale clusters
        int grid_clusters = (h_num_clusters + block_size - 1) / block_size;
        scale_clusters_kernel<<<grid_clusters, block_size>>>(
            d_cluster_row_sum, d_cluster_col_sum, d_cluster_val_sum,
            d_cluster_count,
            d_out_rows, d_out_cols, d_out_values,
            h_num_clusters, num_rows, num_cols, target_rows, target_cols);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        total_gpu_time += gpu_timer_stop(&timer);
        
        // Free per-run allocations
        free(h_edge_counts);
        free(h_adj_ptr);
        CUDA_CHECK(cudaFree(d_adj_list));
        CUDA_CHECK(cudaFree(d_adj_weights));
        CUDA_CHECK(cudaFree(d_cluster_row_sum));
        CUDA_CHECK(cudaFree(d_cluster_col_sum));
        CUDA_CHECK(cudaFree(d_cluster_val_sum));
        CUDA_CHECK(cudaFree(d_cluster_count));
        CUDA_CHECK(cudaFree(d_out_rows));
        CUDA_CHECK(cudaFree(d_out_cols));
        CUDA_CHECK(cudaFree(d_out_values));
    }
    
    result.gpu_time_ms = total_gpu_time / num_runs;
    result.num_clusters = final_num_clusters;
    result.output_nnz = final_num_clusters;
    
    gpu_timer_destroy(&timer);
    
    // ========== CPU Execution (for comparison) ==========
    if (verify) {
        int *cpu_rows, *cpu_cols;
        float *cpu_vals;
        int cpu_nnz;
        
        CpuTimer cpu_timer;
        cpu_timer_start(&cpu_timer);
        cpu_graph_hem_scale(input, target_rows, target_cols,
                           &cpu_rows, &cpu_cols, &cpu_vals, &cpu_nnz);
        result.cpu_time_ms = cpu_timer_stop(&cpu_timer);
        
        result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        
        free(cpu_rows);
        free(cpu_cols);
        free(cpu_vals);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_node_rows));
    CUDA_CHECK(cudaFree(d_node_cols));
    CUDA_CHECK(cudaFree(d_node_values));
    CUDA_CHECK(cudaFree(d_pos_to_node));
    CUDA_CHECK(cudaFree(d_edge_counts));
    CUDA_CHECK(cudaFree(d_adj_ptr));
    CUDA_CHECK(cudaFree(d_match));
    CUDA_CHECK(cudaFree(d_cluster_id));
    CUDA_CHECK(cudaFree(d_cluster_counter));
    
    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_header() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║   Parallel Graph-Based Sparse Matrix Scaling (CUDA + HEM)        ║\n");
    printf("║                                                                  ║\n");
    printf("║   Authors: Ali Ahmet Taşkesen, Ömer Yıldırım                     ║\n");
    printf("║   Ankara Yıldırım Beyazıt University                             ║\n");
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

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -s <size>    Matrix size (default: 1000)\n");
    printf("  -b <bw>      Bandwidth (default: 5)\n");
    printf("  -r <runs>    Number of runs (default: 5)\n");
    printf("  -v           Verify against CPU\n");
    printf("  -h           Show help\n");
}

int main(int argc, char** argv) {
    int size = 1000;
    int bandwidth = 5;
    int num_runs = 5;
    int verify = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
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
    
    printf("Generating banded matrix: %d x %d (bandwidth=%d)\n", size, size, bandwidth);
    CSRMatrix* matrix = generate_banded_matrix(size, bandwidth);
    printf("Input NNZ: %d, Density: %.6f\n\n", matrix->nnz,
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
    };
    int num_ops = sizeof(operations) / sizeof(operations[0]);
    
    printf("Running benchmarks (%d runs each)...\n\n", num_runs);
    
    if (verify) {
        printf("┌───────────────────┬────────────┬───────────┬───────────┬──────────┬─────────┐\n");
        printf("│ Operation         │ Target     │ GPU (ms)  │ CPU (ms)  │ Clusters │ Speedup │\n");
        printf("├───────────────────┼────────────┼───────────┼───────────┼──────────┼─────────┤\n");
    } else {
        printf("┌───────────────────┬────────────┬───────────┬──────────┬───────────┐\n");
        printf("│ Operation         │ Target     │ GPU (ms)  │ Clusters │ Out NNZ   │\n");
        printf("├───────────────────┼────────────┼───────────┼──────────┼───────────┤\n");
    }
    
    for (int i = 0; i < num_ops; i++) {
        BenchmarkResult res = run_benchmark(matrix,
                                           operations[i].target_rows,
                                           operations[i].target_cols,
                                           verify, num_runs);
        
        char target_str[20];
        sprintf(target_str, "%d x %d", operations[i].target_rows, operations[i].target_cols);
        
        if (verify) {
            printf("│ %-17s │ %-10s │ %9.3f │ %9.3f │ %8d │ %6.2fx │\n",
                   operations[i].name, target_str,
                   res.gpu_time_ms, res.cpu_time_ms,
                   res.num_clusters, res.speedup);
        } else {
            printf("│ %-17s │ %-10s │ %9.3f │ %8d │ %9d │\n",
                   operations[i].name, target_str,
                   res.gpu_time_ms, res.num_clusters, res.output_nnz);
        }
    }
    
    if (verify) {
        printf("└───────────────────┴────────────┴───────────┴───────────┴──────────┴─────────┘\n");
    } else {
        printf("└───────────────────┴────────────┴───────────┴──────────┴───────────┘\n");
    }
    
    printf("\n");
    printf("Algorithm: Graph-based Heavy-Edge Matching (HEM)\n");
    printf("Connectivity: 4-connectivity (Von Neumann neighborhood)\n");
    printf("\nDone!\n");
    
    free_csr_matrix(matrix);
    return 0;
}

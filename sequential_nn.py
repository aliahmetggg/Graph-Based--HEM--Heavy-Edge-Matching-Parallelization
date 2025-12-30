"""
Sequential Nearest Neighbor Sparse Matrix Scaling
Baseline implementation for comparison with CUDA version

Based on MatGen paper by Pamuk et al. (2025)
"""

import numpy as np
from scipy import sparse
from scipy.io import mmread, mmwrite
import time
import argparse


def nn_scale_sparse(matrix, target_rows, target_cols):
    """
    Scale a sparse matrix using Nearest Neighbor interpolation.
    
    Args:
        matrix: scipy sparse matrix (any format, will be converted to COO)
        target_rows: target number of rows
        target_cols: target number of columns
    
    Returns:
        Scaled sparse matrix in CSR format
    """
    # Convert to COO for easy iteration
    coo = matrix.tocoo()
    
    original_rows, original_cols = matrix.shape
    
    # Compute scaling factors
    scale_r = original_rows / target_rows
    scale_c = original_cols / target_cols
    
    # Map each nonzero to new position
    new_rows = np.round(coo.row / scale_r).astype(np.int32)
    new_cols = np.round(coo.col / scale_c).astype(np.int32)
    
    # Clip to valid range (edge case handling)
    new_rows = np.clip(new_rows, 0, target_rows - 1)
    new_cols = np.clip(new_cols, 0, target_cols - 1)
    
    # Create new sparse matrix (duplicates are summed automatically)
    scaled_matrix = sparse.coo_matrix(
        (coo.data, (new_rows, new_cols)),
        shape=(target_rows, target_cols)
    )
    
    return scaled_matrix.tocsr()


def compute_structural_features(matrix):
    """
    Compute structural features for similarity comparison.
    Based on MatGen paper metrics.
    """
    csr = matrix.tocsr()
    coo = matrix.tocoo()
    
    n_rows, n_cols = matrix.shape
    nnz = matrix.nnz
    
    features = {}
    
    # Basic features
    features['size'] = n_rows
    features['nnz'] = nnz
    features['density'] = nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0
    
    # Diagonal features
    diag_mask = coo.row == coo.col
    features['diag_nnz'] = np.sum(diag_mask)
    
    # Distance from diagonal
    distances = np.abs(coo.row - coo.col)
    features['avg_diag_dist'] = np.mean(distances) if len(distances) > 0 else 0
    features['bandwidth'] = np.max(distances) if len(distances) > 0 else 0
    
    # Row-wise statistics
    row_nnz = np.diff(csr.indptr)
    features['row_max'] = np.max(row_nnz) if len(row_nnz) > 0 else 0
    features['row_min'] = np.min(row_nnz) if len(row_nnz) > 0 else 0
    features['row_std'] = np.std(row_nnz) if len(row_nnz) > 0 else 0
    
    # Number of diagonals with nonzeros
    unique_diags = len(np.unique(coo.col - coo.row))
    features['num_diags'] = unique_diags
    
    # Pattern symmetry
    coo_t = matrix.T.tocoo()
    sym_set = set(zip(coo.row, coo.col))
    sym_t_set = set(zip(coo_t.row, coo_t.col))
    sym_count = len(sym_set & sym_t_set)
    features['pattern_symmetry'] = sym_count / nnz if nnz > 0 else 1.0
    
    return features


def cosine_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors.
    Features are normalized by matrix size for fair comparison.
    """
    # Normalize features by size
    size1 = features1['size']
    size2 = features2['size']
    
    # Create normalized feature vectors
    keys = ['density', 'avg_diag_dist', 'bandwidth', 'row_max', 'row_min', 
            'row_std', 'num_diags', 'pattern_symmetry']
    
    # Normalize size-dependent features
    v1 = np.array([
        features1['density'],
        features1['avg_diag_dist'] / size1,
        features1['bandwidth'] / size1,
        features1['row_max'] / size1,
        features1['row_min'] / size1,
        features1['row_std'] / size1,
        features1['num_diags'] / size1,
        features1['pattern_symmetry']
    ])
    
    v2 = np.array([
        features2['density'],
        features2['avg_diag_dist'] / size2,
        features2['bandwidth'] / size2,
        features2['row_max'] / size2,
        features2['row_min'] / size2,
        features2['row_std'] / size2,
        features2['num_diags'] / size2,
        features2['pattern_symmetry']
    ])
    
    # Cosine similarity
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def benchmark_nn_scaling(matrix, operations):
    """
    Benchmark NN scaling with different operations.
    
    Args:
        matrix: input sparse matrix
        operations: dict of operation_name -> (target_rows, target_cols)
    
    Returns:
        Dictionary of results
    """
    results = {}
    original_features = compute_structural_features(matrix)
    
    for op_name, (target_rows, target_cols) in operations.items():
        print(f"\n--- {op_name}: {matrix.shape} -> ({target_rows}, {target_cols}) ---")
        
        # Time the scaling
        start = time.perf_counter()
        scaled = nn_scale_sparse(matrix, target_rows, target_cols)
        elapsed = time.perf_counter() - start
        
        # Compute features and similarity
        scaled_features = compute_structural_features(scaled)
        similarity = cosine_similarity(original_features, scaled_features)
        
        results[op_name] = {
            'time_ms': elapsed * 1000,
            'input_shape': matrix.shape,
            'output_shape': scaled.shape,
            'input_nnz': matrix.nnz,
            'output_nnz': scaled.nnz,
            'similarity': similarity
        }
        
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Input NNZ: {matrix.nnz}, Output NNZ: {scaled.nnz}")
        print(f"  Cosine Similarity: {similarity:.4f}")
    
    return results


def generate_test_matrix(size, density=0.05, pattern='banded'):
    """
    Generate a test sparse matrix with specific pattern.
    """
    if pattern == 'banded':
        # Create banded matrix (like finite difference)
        diagonals = [
            np.ones(size),      # main diagonal
            np.ones(size-1),    # super diagonal
            np.ones(size-1),    # sub diagonal
            np.ones(size-5) if size > 5 else np.array([]),  # far diagonal
        ]
        offsets = [0, 1, -1, 5]
        matrix = sparse.diags(diagonals[:len([d for d in diagonals if len(d) > 0])], 
                             offsets[:len([d for d in diagonals if len(d) > 0])],
                             shape=(size, size), format='csr')
    elif pattern == 'random':
        matrix = sparse.random(size, size, density=density, format='csr')
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return matrix


def main():
    parser = argparse.ArgumentParser(description='Sequential NN Sparse Matrix Scaling')
    parser.add_argument('--input', '-i', type=str, help='Input matrix file (Matrix Market format)')
    parser.add_argument('--size', '-s', type=int, default=1000, help='Size for generated test matrix')
    parser.add_argument('--pattern', '-p', type=str, default='banded', 
                       choices=['banded', 'random'], help='Pattern for generated matrix')
    parser.add_argument('--density', '-d', type=float, default=0.01, help='Density for random matrix')
    args = parser.parse_args()
    
    # Load or generate matrix
    if args.input:
        print(f"Loading matrix from {args.input}...")
        matrix = mmread(args.input).tocsr()
    else:
        print(f"Generating {args.pattern} test matrix of size {args.size}...")
        matrix = generate_test_matrix(args.size, args.density, args.pattern)
    
    print(f"Matrix shape: {matrix.shape}, NNZ: {matrix.nnz}")
    print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
    
    # Define scaling operations
    n = matrix.shape[0]
    operations = {
        'expand_plus1': (n + 1, n + 1),
        'upscale_2x': (n * 2, n * 2),
        'reduce_minus1': (n - 1, n - 1),
        'downscale_half': (n // 2, n // 2),
    }
    
    # Run benchmark
    results = benchmark_nn_scaling(matrix, operations)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for op_name, res in results.items():
        print(f"{op_name:20s}: {res['time_ms']:8.2f} ms, similarity: {res['similarity']:.4f}")


if __name__ == '__main__':
    main()

"""
DCT-Based Sparse Matrix Scaling
Sequential and Parallel Implementation

Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
Based on MatGen paper by Pamuk et al. (2025)

DCT (Discrete Cosine Transform) method operates in frequency domain:
1. Apply 2D DCT to sparse matrix
2. Resize coefficient matrix (zero-pad or truncate)
3. Apply inverse DCT
4. Threshold small values to maintain sparsity
"""

import numpy as np
from scipy import sparse
from scipy.fftpack import dct, idct
from scipy.io import mmread
import time
import argparse

# Try CuPy for GPU
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available - using CPU only")


def dct2(matrix):
    """Apply 2D DCT (Type-II)"""
    return dct(dct(matrix, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(matrix):
    """Apply 2D Inverse DCT (Type-III)"""
    return idct(idct(matrix, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct_scale_dense(matrix_dense, target_rows, target_cols, threshold=1e-10):
    """
    Scale a dense matrix using DCT method.
    
    Args:
        matrix_dense: 2D numpy array
        target_rows: target number of rows
        target_cols: target number of columns
        threshold: values below this are set to zero
    
    Returns:
        Scaled dense matrix
    """
    orig_rows, orig_cols = matrix_dense.shape
    
    # Step 1: Apply 2D DCT
    dct_coeffs = dct2(matrix_dense)
    
    # Step 2: Resize coefficient matrix
    if target_rows >= orig_rows and target_cols >= orig_cols:
        # Upscaling: zero-pad
        resized = np.zeros((target_rows, target_cols), dtype=np.float64)
        resized[:orig_rows, :orig_cols] = dct_coeffs
    else:
        # Downscaling: truncate (keep low frequencies)
        min_rows = min(target_rows, orig_rows)
        min_cols = min(target_cols, orig_cols)
        resized = np.zeros((target_rows, target_cols), dtype=np.float64)
        resized[:min_rows, :min_cols] = dct_coeffs[:min_rows, :min_cols]
    
    # Step 3: Apply inverse DCT
    reconstructed = idct2(resized)
    
    # Step 4: Apply threshold to maintain sparsity
    reconstructed[np.abs(reconstructed) < threshold] = 0
    
    return reconstructed


def dct_scale_sparse(matrix, target_rows, target_cols, threshold=None, block_size=None):
    """
    Scale a sparse matrix using DCT method.
    
    Args:
        matrix: scipy sparse matrix
        target_rows: target number of rows  
        target_cols: target number of columns
        threshold: sparsity threshold (auto-computed if None)
        block_size: process in blocks for large matrices (None = full matrix)
    
    Returns:
        Scaled sparse matrix in CSR format
    """
    orig_rows, orig_cols = matrix.shape
    
    # Convert to dense (required for DCT)
    # For very large matrices, use block-wise processing
    if block_size is not None and (orig_rows > block_size or orig_cols > block_size):
        return dct_scale_blockwise(matrix, target_rows, target_cols, threshold, block_size)
    
    # Full matrix processing
    dense = matrix.toarray().astype(np.float64)
    
    # Auto-compute threshold based on original matrix statistics
    if threshold is None:
        nonzero_vals = np.abs(dense[dense != 0])
        if len(nonzero_vals) > 0:
            threshold = np.percentile(nonzero_vals, 5) * 0.1  # 10% of 5th percentile
        else:
            threshold = 1e-10
    
    # Apply DCT scaling
    scaled_dense = dct_scale_dense(dense, target_rows, target_cols, threshold)
    
    # Convert back to sparse
    scaled_sparse = sparse.csr_matrix(scaled_dense)
    
    return scaled_sparse


def dct_scale_blockwise(matrix, target_rows, target_cols, threshold, block_size):
    """
    Block-wise DCT scaling for large matrices.
    Processes the matrix in blocks to reduce memory usage.
    """
    orig_rows, orig_cols = matrix.shape
    scale_r = target_rows / orig_rows
    scale_c = target_cols / orig_cols
    
    # Initialize output
    output_data = []
    output_rows = []
    output_cols = []
    
    # Process blocks
    for i_start in range(0, orig_rows, block_size):
        i_end = min(i_start + block_size, orig_rows)
        
        for j_start in range(0, orig_cols, block_size):
            j_end = min(j_start + block_size, orig_cols)
            
            # Extract block
            block = matrix[i_start:i_end, j_start:j_end].toarray().astype(np.float64)
            
            # Skip empty blocks
            if np.count_nonzero(block) == 0:
                continue
            
            # Compute target block size
            target_i_start = int(i_start * scale_r)
            target_i_end = int(i_end * scale_r)
            target_j_start = int(j_start * scale_c)
            target_j_end = int(j_end * scale_c)
            
            target_block_rows = target_i_end - target_i_start
            target_block_cols = target_j_end - target_j_start
            
            if target_block_rows <= 0 or target_block_cols <= 0:
                continue
            
            # Scale block
            scaled_block = dct_scale_dense(block, target_block_rows, target_block_cols, threshold)
            
            # Collect nonzeros
            nz_rows, nz_cols = np.nonzero(scaled_block)
            for idx in range(len(nz_rows)):
                output_rows.append(target_i_start + nz_rows[idx])
                output_cols.append(target_j_start + nz_cols[idx])
                output_data.append(scaled_block[nz_rows[idx], nz_cols[idx]])
    
    # Create sparse matrix
    if len(output_data) > 0:
        output = sparse.coo_matrix(
            (output_data, (output_rows, output_cols)),
            shape=(target_rows, target_cols)
        ).tocsr()
    else:
        output = sparse.csr_matrix((target_rows, target_cols))
    
    return output


class DCTScaler:
    """DCT-based sparse matrix scaler with optional GPU support"""
    
    def __init__(self, use_gpu=True, threshold=None, block_size=None):
        self.use_gpu = use_gpu and HAS_CUPY
        self.threshold = threshold
        self.block_size = block_size
        
        if self.use_gpu:
            print("DCT Scaler: Using GPU (CuPy)")
        else:
            print("DCT Scaler: Using CPU")
    
    def scale(self, matrix, target_rows, target_cols):
        """
        Scale sparse matrix using DCT.
        
        Returns:
            tuple: (scaled_matrix, time_ms)
        """
        start = time.perf_counter()
        
        if self.use_gpu:
            scaled = self._scale_gpu(matrix, target_rows, target_cols)
        else:
            scaled = dct_scale_sparse(matrix, target_rows, target_cols, 
                                      self.threshold, self.block_size)
        
        elapsed = (time.perf_counter() - start) * 1000
        return scaled, elapsed
    
    def _scale_gpu(self, matrix, target_rows, target_cols):
        """GPU implementation using CuPy"""
        # Convert to dense on GPU
        dense_cpu = matrix.toarray().astype(np.float64)
        dense_gpu = cp.asarray(dense_cpu)
        
        orig_rows, orig_cols = matrix.shape
        
        # CuPy DCT (via FFT)
        # DCT-II can be computed using FFT with some preprocessing
        dct_coeffs = self._dct2_gpu(dense_gpu)
        
        # Resize
        if target_rows >= orig_rows and target_cols >= orig_cols:
            resized = cp.zeros((target_rows, target_cols), dtype=cp.float64)
            resized[:orig_rows, :orig_cols] = dct_coeffs
        else:
            min_rows = min(target_rows, orig_rows)
            min_cols = min(target_cols, orig_cols)
            resized = cp.zeros((target_rows, target_cols), dtype=cp.float64)
            resized[:min_rows, :min_cols] = dct_coeffs[:min_rows, :min_cols]
        
        # Inverse DCT
        reconstructed = self._idct2_gpu(resized)
        
        # Threshold
        if self.threshold is None:
            nonzero_vals = cp.abs(dense_gpu[dense_gpu != 0])
            if len(nonzero_vals) > 0:
                threshold = float(cp.percentile(nonzero_vals, 5)) * 0.1
            else:
                threshold = 1e-10
        else:
            threshold = self.threshold
        
        reconstructed[cp.abs(reconstructed) < threshold] = 0
        
        # Back to CPU sparse
        result_cpu = cp.asnumpy(reconstructed)
        return sparse.csr_matrix(result_cpu)
    
    def _dct2_gpu(self, x):
        """2D DCT using CuPy FFT"""
        # DCT-II via FFT: mirror and apply FFT
        n1, n2 = x.shape
        
        # DCT along axis 0
        v = cp.concatenate([x, x[::-1, :]], axis=0)
        V = cp.fft.fft(v, axis=0)[:n1, :]
        k = cp.arange(n1).reshape(-1, 1)
        dct1 = cp.real(V * cp.exp(-1j * cp.pi * k / (2 * n1))) * cp.sqrt(2.0 / n1)
        dct1[0, :] /= cp.sqrt(2)
        
        # DCT along axis 1
        v = cp.concatenate([dct1, dct1[:, ::-1]], axis=1)
        V = cp.fft.fft(v, axis=1)[:, :n2]
        k = cp.arange(n2).reshape(1, -1)
        dct2 = cp.real(V * cp.exp(-1j * cp.pi * k / (2 * n2))) * cp.sqrt(2.0 / n2)
        dct2[:, 0] /= cp.sqrt(2)
        
        return dct2
    
    def _idct2_gpu(self, x):
        """2D Inverse DCT using CuPy FFT"""
        n1, n2 = x.shape
        
        # IDCT along axis 0
        x_scaled = x.copy()
        x_scaled[0, :] *= cp.sqrt(2)
        k = cp.arange(n1).reshape(-1, 1)
        V = x_scaled * cp.exp(1j * cp.pi * k / (2 * n1)) * cp.sqrt(n1 / 2.0)
        v = cp.zeros((2 * n1, n2), dtype=cp.complex128)
        v[:n1, :] = V
        v[n1+1:, :] = cp.conj(V[1:, :][::-1, :])
        idct1 = cp.real(cp.fft.ifft(v, axis=0))[:n1, :]
        
        # IDCT along axis 1
        idct1[:, 0] *= cp.sqrt(2)
        k = cp.arange(n2).reshape(1, -1)
        V = idct1 * cp.exp(1j * cp.pi * k / (2 * n2)) * cp.sqrt(n2 / 2.0)
        v = cp.zeros((n1, 2 * n2), dtype=cp.complex128)
        v[:, :n2] = V
        v[:, n2+1:] = cp.conj(V[:, 1:][:, ::-1])
        idct2 = cp.real(cp.fft.ifft(v, axis=1))[:, :n2]
        
        return idct2


def compute_structural_features(matrix):
    """Compute structural features for similarity comparison."""
    csr = matrix.tocsr()
    coo = matrix.tocoo()
    
    n_rows, n_cols = matrix.shape
    nnz = matrix.nnz
    
    if nnz == 0:
        return {
            'size': n_rows, 'nnz': 0, 'density': 0,
            'diag_nnz': 0, 'avg_diag_dist': 0, 'bandwidth': 0,
            'row_max': 0, 'row_min': 0, 'row_std': 0,
            'num_diags': 0, 'pattern_symmetry': 1.0
        }
    
    features = {
        'size': n_rows,
        'nnz': nnz,
        'density': nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0
    }
    
    # Diagonal features
    diag_mask = coo.row == coo.col
    features['diag_nnz'] = int(np.sum(diag_mask))
    
    distances = np.abs(coo.row - coo.col)
    features['avg_diag_dist'] = float(np.mean(distances))
    features['bandwidth'] = int(np.max(distances))
    
    # Row-wise statistics
    row_nnz = np.diff(csr.indptr)
    features['row_max'] = int(np.max(row_nnz))
    features['row_min'] = int(np.min(row_nnz))
    features['row_std'] = float(np.std(row_nnz))
    
    features['num_diags'] = len(np.unique(coo.col - coo.row))
    
    # Pattern symmetry
    sym_set = set(zip(coo.row.tolist(), coo.col.tolist()))
    sym_t_set = set(zip(coo.col.tolist(), coo.row.tolist()))
    features['pattern_symmetry'] = len(sym_set & sym_t_set) / nnz
    
    return features


def cosine_similarity(f1, f2):
    """Compute cosine similarity between feature vectors."""
    size1, size2 = f1['size'], f2['size']
    
    v1 = np.array([
        f1['density'],
        f1['avg_diag_dist'] / size1 if size1 > 0 else 0,
        f1['bandwidth'] / size1 if size1 > 0 else 0,
        f1['row_max'] / size1 if size1 > 0 else 0,
        f1['row_min'] / size1 if size1 > 0 else 0,
        f1['row_std'] / size1 if size1 > 0 else 0,
        f1['num_diags'] / size1 if size1 > 0 else 0,
        f1['pattern_symmetry']
    ])
    
    v2 = np.array([
        f2['density'],
        f2['avg_diag_dist'] / size2 if size2 > 0 else 0,
        f2['bandwidth'] / size2 if size2 > 0 else 0,
        f2['row_max'] / size2 if size2 > 0 else 0,
        f2['row_min'] / size2 if size2 > 0 else 0,
        f2['row_std'] / size2 if size2 > 0 else 0,
        f2['num_diags'] / size2 if size2 > 0 else 0,
        f2['pattern_symmetry']
    ])
    
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def generate_test_matrix(size, density=0.01, pattern='banded', bandwidth=5):
    """Generate test sparse matrix."""
    if pattern == 'banded':
        diagonals = [np.ones(size)]
        offsets = [0]
        for k in range(1, bandwidth + 1):
            if size - k > 0:
                diagonals.extend([np.ones(size - k) * (1.0 - k * 0.1), 
                                  np.ones(size - k) * (1.0 - k * 0.1)])
                offsets.extend([k, -k])
        return sparse.diags(diagonals, offsets, shape=(size, size), format='csr')
    elif pattern == 'random':
        return sparse.random(size, size, density=density, format='csr')
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def benchmark_dct(matrix, num_runs=3):
    """Run DCT scaling benchmark."""
    scaler = DCTScaler(use_gpu=False)
    n = matrix.shape[0]
    
    operations = {
        'Expand (+1)': (n + 1, n + 1),
        'Upscale (2x)': (n * 2, n * 2),
        'Reduce (-1)': (n - 1, n - 1),
        'Downscale (1/2x)': (n // 2, n // 2),
    }
    
    original_features = compute_structural_features(matrix)
    results = {}
    
    print("\n" + "=" * 75)
    print("DCT SCALING BENCHMARK")
    print("=" * 75)
    print(f"{'Operation':<20} {'Target':>12} {'Time (ms)':>12} {'NNZ':>10} {'Similarity':>12}")
    print("-" * 75)
    
    for op_name, (target_r, target_c) in operations.items():
        times = []
        for _ in range(num_runs):
            scaled, elapsed = scaler.scale(matrix, target_r, target_c)
            times.append(elapsed)
        
        avg_time = np.mean(times)
        scaled, _ = scaler.scale(matrix, target_r, target_c)
        scaled_features = compute_structural_features(scaled)
        similarity = cosine_similarity(original_features, scaled_features)
        
        results[op_name] = {
            'target': (target_r, target_c),
            'time_ms': avg_time,
            'output_nnz': scaled.nnz,
            'similarity': similarity
        }
        
        print(f"{op_name:<20} {target_r}x{target_c:>6} {avg_time:>12.2f} {scaled.nnz:>10} {similarity:>12.4f}")
    
    print("=" * 75)
    return results


def compare_nn_vs_dct(matrix, num_runs=3):
    """Compare NN and DCT methods."""
    from parallel_nn_python import NNScaler
    
    nn_scaler = NNScaler(use_gpu=False)
    dct_scaler = DCTScaler(use_gpu=False)
    
    n = matrix.shape[0]
    operations = [
        ('Expand (+1)', n + 1, n + 1),
        ('Upscale (2x)', n * 2, n * 2),
        ('Reduce (-1)', n - 1, n - 1),
        ('Downscale (1/2x)', n // 2, n // 2),
    ]
    
    original_features = compute_structural_features(matrix)
    
    print("\n" + "=" * 85)
    print("NN vs DCT COMPARISON")
    print("=" * 85)
    print(f"{'Operation':<18} {'NN Time':>10} {'NN Sim':>10} {'DCT Time':>10} {'DCT Sim':>10} {'Winner':>12}")
    print("-" * 85)
    
    for op_name, target_r, target_c in operations:
        # NN
        nn_times = []
        for _ in range(num_runs):
            _, elapsed = nn_scaler.scale(matrix, target_r, target_c)
            nn_times.append(elapsed)
        nn_scaled, _ = nn_scaler.scale(matrix, target_r, target_c)
        nn_sim = cosine_similarity(original_features, compute_structural_features(nn_scaled))
        nn_avg = np.mean(nn_times)
        
        # DCT
        dct_times = []
        for _ in range(num_runs):
            _, elapsed = dct_scaler.scale(matrix, target_r, target_c)
            dct_times.append(elapsed)
        dct_scaled, _ = dct_scaler.scale(matrix, target_r, target_c)
        dct_sim = cosine_similarity(original_features, compute_structural_features(dct_scaled))
        dct_avg = np.mean(dct_times)
        
        # Winner (based on similarity, since that's the goal)
        if nn_sim > dct_sim:
            winner = "NN"
        elif dct_sim > nn_sim:
            winner = "DCT"
        else:
            winner = "Tie"
        
        print(f"{op_name:<18} {nn_avg:>10.2f} {nn_sim:>10.4f} {dct_avg:>10.2f} {dct_sim:>10.4f} {winner:>12}")
    
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description='DCT Sparse Matrix Scaling')
    parser.add_argument('--input', '-i', type=str, help='Input matrix file (MTX format)')
    parser.add_argument('--size', '-s', type=int, default=500, help='Test matrix size')
    parser.add_argument('--pattern', '-p', type=str, default='banded', 
                       choices=['banded', 'random'])
    parser.add_argument('--bandwidth', '-b', type=int, default=5, help='Bandwidth for banded')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare NN vs DCT')
    parser.add_argument('--runs', '-r', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    # Load or generate matrix
    if args.input:
        print(f"Loading matrix from {args.input}...")
        matrix = mmread(args.input).tocsr()
    else:
        print(f"Generating {args.pattern} test matrix (size={args.size})...")
        matrix = generate_test_matrix(args.size, pattern=args.pattern, bandwidth=args.bandwidth)
    
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"NNZ: {matrix.nnz}")
    print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
    
    # Run benchmark
    results = benchmark_dct(matrix, num_runs=args.runs)
    
    # Compare if requested
    if args.compare:
        compare_nn_vs_dct(matrix, num_runs=args.runs)
    
    return results


if __name__ == '__main__':
    main()

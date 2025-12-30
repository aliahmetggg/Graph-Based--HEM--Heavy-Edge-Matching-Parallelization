"""
Parallel Nearest Neighbor Sparse Matrix Scaling
Python implementation with optional GPU acceleration via CuPy

Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
"""

import numpy as np
from scipy import sparse
from scipy.io import mmread, mmwrite
import time
import argparse

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    HAS_CUPY = True
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    HAS_CUPY = False
    print("CuPy not available - using CPU only")


class NNScaler:
    """Nearest Neighbor Sparse Matrix Scaler with optional GPU support"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            # Get GPU info
            device = cp.cuda.Device()
            print(f"Using GPU: {device.id}")
            print(f"  Memory: {device.mem_info[1] / 1e9:.2f} GB total")
    
    def scale(self, matrix, target_rows, target_cols):
        """
        Scale sparse matrix using Nearest Neighbor interpolation.
        
        Args:
            matrix: scipy sparse matrix
            target_rows: target number of rows
            target_cols: target number of columns
        
        Returns:
            tuple: (scaled_matrix, time_ms)
        """
        if self.use_gpu:
            return self._scale_gpu(matrix, target_rows, target_cols)
        else:
            return self._scale_cpu(matrix, target_rows, target_cols)
    
    def _scale_cpu(self, matrix, target_rows, target_cols):
        """CPU implementation"""
        start = time.perf_counter()
        
        coo = matrix.tocoo()
        original_rows, original_cols = matrix.shape
        
        scale_r = original_rows / target_rows
        scale_c = original_cols / target_cols
        
        # Vectorized coordinate transformation
        new_rows = np.round(coo.row / scale_r).astype(np.int32)
        new_cols = np.round(coo.col / scale_c).astype(np.int32)
        
        # Clip to valid range
        new_rows = np.clip(new_rows, 0, target_rows - 1)
        new_cols = np.clip(new_cols, 0, target_cols - 1)
        
        # Create new sparse matrix
        scaled = sparse.coo_matrix(
            (coo.data, (new_rows, new_cols)),
            shape=(target_rows, target_cols)
        ).tocsr()
        
        elapsed = (time.perf_counter() - start) * 1000
        return scaled, elapsed
    
    def _scale_gpu(self, matrix, target_rows, target_cols):
        """GPU implementation using CuPy"""
        # Convert to COO and transfer to GPU
        coo = matrix.tocoo()
        original_rows, original_cols = matrix.shape
        
        scale_r = original_rows / target_rows
        scale_c = original_cols / target_cols
        
        # Transfer to GPU
        start = time.perf_counter()
        
        d_rows = cp.asarray(coo.row)
        d_cols = cp.asarray(coo.col)
        d_data = cp.asarray(coo.data)
        
        # GPU computation - vectorized
        new_rows = cp.round(d_rows / scale_r).astype(cp.int32)
        new_cols = cp.round(d_cols / scale_c).astype(cp.int32)
        
        # Clip to valid range
        new_rows = cp.clip(new_rows, 0, target_rows - 1)
        new_cols = cp.clip(new_cols, 0, target_cols - 1)
        
        # Create sparse matrix on GPU
        scaled_gpu = cp_sparse.coo_matrix(
            (d_data, (new_rows, new_cols)),
            shape=(target_rows, target_cols)
        ).tocsr()
        
        # Synchronize and measure time
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        # Transfer back to CPU
        scaled_cpu = sparse.csr_matrix(
            (cp.asnumpy(scaled_gpu.data), 
             cp.asnumpy(scaled_gpu.indices), 
             cp.asnumpy(scaled_gpu.indptr)),
            shape=(target_rows, target_cols)
        )
        
        return scaled_cpu, elapsed


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
    
    # Distance from diagonal
    distances = np.abs(coo.row - coo.col)
    features['avg_diag_dist'] = float(np.mean(distances))
    features['bandwidth'] = int(np.max(distances))
    
    # Row-wise statistics
    row_nnz = np.diff(csr.indptr)
    features['row_max'] = int(np.max(row_nnz))
    features['row_min'] = int(np.min(row_nnz))
    features['row_std'] = float(np.std(row_nnz))
    
    # Number of diagonals with nonzeros
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


def download_suitesparse_matrix(name):
    """Download matrix from SuiteSparse collection."""
    import urllib.request
    import tarfile
    import os
    
    # Common matrices for testing
    matrices = {
        'cavity03': 'https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity03.tar.gz',
        'bcsstk01': 'https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk01.tar.gz',
    }
    
    if name not in matrices:
        print(f"Matrix {name} not in predefined list. Available: {list(matrices.keys())}")
        return None
    
    url = matrices[name]
    tar_file = f"/tmp/{name}.tar.gz"
    
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, tar_file)
    
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall('/tmp')
    
    mtx_file = f"/tmp/{name}/{name}.mtx"
    if os.path.exists(mtx_file):
        return mmread(mtx_file).tocsr()
    return None


def benchmark(matrix, scaler, num_runs=5):
    """Run benchmark on all scaling operations."""
    n = matrix.shape[0]
    
    operations = {
        'Expand (+1)': (n + 1, n + 1),
        'Upscale (2x)': (n * 2, n * 2),
        'Reduce (-1)': (n - 1, n - 1),
        'Downscale (1/2x)': (n // 2, n // 2),
        'Upscale (4x)': (n * 4, n * 4),
    }
    
    original_features = compute_structural_features(matrix)
    results = {}
    
    print("\n" + "=" * 70)
    print(f"{'Operation':<20} {'Target':>12} {'Time (ms)':>12} {'NNZ':>10} {'Similarity':>12}")
    print("-" * 70)
    
    for op_name, (target_r, target_c) in operations.items():
        times = []
        for _ in range(num_runs):
            scaled, elapsed = scaler.scale(matrix, target_r, target_c)
            times.append(elapsed)
        
        avg_time = np.mean(times)
        scaled_features = compute_structural_features(scaled)
        similarity = cosine_similarity(original_features, scaled_features)
        
        results[op_name] = {
            'target': (target_r, target_c),
            'time_ms': avg_time,
            'output_nnz': scaled.nnz,
            'similarity': similarity
        }
        
        print(f"{op_name:<20} {target_r}x{target_c:>6} {avg_time:>12.3f} {scaled.nnz:>10} {similarity:>12.4f}")
    
    print("=" * 70)
    return results


def compare_cpu_gpu(matrix, num_runs=5):
    """Compare CPU and GPU performance."""
    if not HAS_CUPY:
        print("CuPy not available, skipping GPU comparison")
        return
    
    cpu_scaler = NNScaler(use_gpu=False)
    gpu_scaler = NNScaler(use_gpu=True)
    
    n = matrix.shape[0]
    operations = [
        ('2x', n * 2, n * 2),
        ('4x', n * 4, n * 4),
    ]
    
    print("\n" + "=" * 70)
    print("CPU vs GPU COMPARISON")
    print("=" * 70)
    print(f"{'Operation':<15} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for name, tr, tc in operations:
        # CPU timing
        cpu_times = []
        for _ in range(num_runs):
            _, elapsed = cpu_scaler.scale(matrix, tr, tc)
            cpu_times.append(elapsed)
        cpu_avg = np.mean(cpu_times)
        
        # GPU timing
        gpu_times = []
        for _ in range(num_runs):
            _, elapsed = gpu_scaler.scale(matrix, tr, tc)
            gpu_times.append(elapsed)
        gpu_avg = np.mean(gpu_times)
        
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        print(f"{name:<15} {cpu_avg:>12.3f} {gpu_avg:>12.3f} {speedup:>10.2f}x")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Parallel NN Sparse Matrix Scaling')
    parser.add_argument('--input', '-i', type=str, help='Input matrix file (MTX format)')
    parser.add_argument('--size', '-s', type=int, default=2000, help='Test matrix size')
    parser.add_argument('--pattern', '-p', type=str, default='banded', 
                       choices=['banded', 'random'])
    parser.add_argument('--bandwidth', '-b', type=int, default=10, help='Bandwidth for banded')
    parser.add_argument('--density', '-d', type=float, default=0.01, help='Density for random')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--compare', action='store_true', help='Compare CPU vs GPU')
    parser.add_argument('--runs', '-r', type=int, default=5, help='Number of benchmark runs')
    args = parser.parse_args()
    
    # Load or generate matrix
    if args.input:
        print(f"Loading matrix from {args.input}...")
        matrix = mmread(args.input).tocsr()
    else:
        print(f"Generating {args.pattern} test matrix (size={args.size})...")
        matrix = generate_test_matrix(args.size, args.density, args.pattern, args.bandwidth)
    
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"NNZ: {matrix.nnz}")
    print(f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
    
    # Run benchmarks
    scaler = NNScaler(use_gpu=args.gpu)
    results = benchmark(matrix, scaler, num_runs=args.runs)
    
    # Compare CPU vs GPU if requested
    if args.compare:
        compare_cpu_gpu(matrix, num_runs=args.runs)
    
    return results


if __name__ == '__main__':
    main()

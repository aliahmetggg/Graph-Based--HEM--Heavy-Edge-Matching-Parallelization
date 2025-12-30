"""
Comprehensive Benchmark Script for Parallel NN Sparse Matrix Scaling
Generates results for the paper

Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
"""

import numpy as np
from scipy import sparse
from scipy.io import mmread
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our implementation
from parallel_nn_python import NNScaler, compute_structural_features, cosine_similarity, generate_test_matrix


def run_scalability_benchmark(sizes, num_runs=3):
    """Test scalability with increasing matrix sizes."""
    print("\n" + "=" * 70)
    print("SCALABILITY BENCHMARK")
    print("=" * 70)
    
    scaler = NNScaler(use_gpu=False)
    results = []
    
    for size in sizes:
        print(f"\nTesting size {size}x{size}...")
        matrix = generate_test_matrix(size, pattern='banded', bandwidth=10)
        
        # Test 2x upscaling
        times = []
        for _ in range(num_runs):
            _, elapsed = scaler.scale(matrix, size * 2, size * 2)
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = matrix.nnz / (avg_time / 1000)  # nnz per second
        
        results.append({
            'size': size,
            'nnz': matrix.nnz,
            'time_ms': avg_time,
            'throughput': throughput
        })
        
        print(f"  NNZ: {matrix.nnz}, Time: {avg_time:.3f} ms, Throughput: {throughput:.0f} nnz/s")
    
    return results


def run_operation_benchmark(matrix, num_runs=5):
    """Benchmark all scaling operations on a given matrix."""
    scaler = NNScaler(use_gpu=False)
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
    
    for op_name, (target_r, target_c) in operations.items():
        times = []
        for _ in range(num_runs):
            scaled, elapsed = scaler.scale(matrix, target_r, target_c)
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        scaled_features = compute_structural_features(scaled)
        similarity = cosine_similarity(original_features, scaled_features)
        
        results[op_name] = {
            'input_shape': matrix.shape,
            'target_shape': (target_r, target_c),
            'input_nnz': matrix.nnz,
            'output_nnz': scaled.nnz,
            'time_ms': avg_time,
            'time_std': std_time,
            'similarity': similarity
        }
    
    return results


def plot_scalability(results, output_path):
    """Generate scalability plot."""
    sizes = [r['size'] for r in results]
    times = [r['time_ms'] for r in results]
    nnzs = [r['nnz'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time vs Size
    ax1.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Matrix Size (N)', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('NN Scaling Time vs Matrix Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Throughput vs Size
    throughputs = [r['throughput'] for r in results]
    ax2.plot(sizes, throughputs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Matrix Size (N)', fontsize=12)
    ax2.set_ylabel('Throughput (nnz/s)', fontsize=12)
    ax2.set_title('Processing Throughput vs Matrix Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved scalability plot to {output_path}")
    plt.close()


def plot_similarity_comparison(results_dict, output_path):
    """Generate similarity comparison bar chart."""
    operations = ['Expand (+1)', 'Upscale (2x)', 'Reduce (-1)', 'Downscale (1/2x)']
    
    # MatGen reference values from the paper
    matgen_values = {
        'Expand (+1)': 0.999,
        'Upscale (2x)': 0.918,
        'Reduce (-1)': 0.999,
        'Downscale (1/2x)': 0.945
    }
    
    our_values = [results_dict.get(op, {}).get('similarity', 0) for op in operations]
    matgen = [matgen_values[op] for op in operations]
    
    x = np.arange(len(operations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, matgen, width, label='MatGen (Sequential)', color='steelblue')
    bars2 = ax.bar(x + width/2, our_values, width, label='Ours (Parallel)', color='coral')
    
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Structural Similarity Comparison: MatGen vs Our Implementation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(operations, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved similarity plot to {output_path}")
    plt.close()


def generate_latex_table(results_dict, matrix_name):
    """Generate LaTeX table for paper."""
    print(f"\n% LaTeX table for {matrix_name}")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Benchmark Results for {matrix_name}}}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Operation & Target Size & Time (ms) & Output NNZ & Similarity \\\\")
    print("\\midrule")
    
    for op_name, data in results_dict.items():
        target = f"{data['target_shape'][0]}×{data['target_shape'][1]}"
        print(f"{op_name} & {target} & {data['time_ms']:.3f} & {data['output_nnz']} & {data['similarity']:.4f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    output_dir = '/home/claude/results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PARALLEL NN SPARSE MATRIX SCALING - BENCHMARK SUITE")
    print("=" * 70)
    
    # 1. Scalability benchmark
    print("\n1. Running scalability benchmark...")
    sizes = [500, 1000, 2000, 5000, 10000, 20000]
    scalability_results = run_scalability_benchmark(sizes)
    
    # Save scalability plot
    plot_scalability(scalability_results, f'{output_dir}/scalability.png')
    
    # 2. Operation benchmark on different matrix sizes
    print("\n2. Running operation benchmarks...")
    
    test_matrices = {
        'small_banded': generate_test_matrix(500, pattern='banded', bandwidth=5),
        'medium_banded': generate_test_matrix(2000, pattern='banded', bandwidth=10),
        'large_banded': generate_test_matrix(5000, pattern='banded', bandwidth=15),
        'medium_random': generate_test_matrix(2000, pattern='random', density=0.005),
    }
    
    all_results = {}
    for name, matrix in test_matrices.items():
        print(f"\nBenchmarking {name} ({matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz})...")
        results = run_operation_benchmark(matrix, num_runs=5)
        all_results[name] = results
        
        # Print results
        print(f"\n{'Operation':<20} {'Time (ms)':>10} {'Output NNZ':>12} {'Similarity':>12}")
        print("-" * 60)
        for op, data in results.items():
            print(f"{op:<20} {data['time_ms']:>10.3f} {data['output_nnz']:>12} {data['similarity']:>12.4f}")
    
    # 3. Generate plots
    print("\n3. Generating plots...")
    
    # Similarity comparison using medium_banded results
    if 'medium_banded' in all_results:
        plot_similarity_comparison(all_results['medium_banded'], f'{output_dir}/similarity_comparison.png')
    
    # 4. Generate LaTeX tables
    print("\n4. Generating LaTeX tables...")
    for name, results in all_results.items():
        generate_latex_table(results, name)
    
    # 5. Save all results as JSON
    results_file = f'{output_dir}/benchmark_results.json'
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj
    
    json_results = {}
    for matrix_name, results in all_results.items():
        json_results[matrix_name] = {}
        for op_name, data in results.items():
            json_results[matrix_name][op_name] = {k: convert_for_json(v) for k, v in data.items()}
    
    with open(results_file, 'w') as f:
        json.dump({
            'scalability': [
                {k: convert_for_json(v) for k, v in r.items()} 
                for r in scalability_results
            ],
            'operations': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nScalability: Tested sizes from {min(sizes)} to {max(sizes)}")
    print(f"Max throughput: {max(r['throughput'] for r in scalability_results):.0f} nnz/s")
    
    if 'medium_banded' in all_results:
        print(f"\nSimilarity scores (medium_banded matrix):")
        for op, data in all_results['medium_banded'].items():
            print(f"  {op}: {data['similarity']:.4f}")
    
    print(f"\nOutput files:")
    print(f"  - {output_dir}/scalability.png")
    print(f"  - {output_dir}/similarity_comparison.png")
    print(f"  - {output_dir}/benchmark_results.json")


if __name__ == '__main__':
    main()

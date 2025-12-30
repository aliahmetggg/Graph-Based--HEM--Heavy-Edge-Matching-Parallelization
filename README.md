# CUDA Parallel Sparse Matrix Scaling

[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

GPU-accelerated sparse matrix scaling using CUDA. Implements three methods: **Nearest Neighbor (NN)**, **Discrete Cosine Transform (DCT)**, and a novel **Graph-based Heavy-Edge Matching (Graph-HEM)**.

<p align="center">
  <img src="https://img.shields.io/badge/NN-25x%20speedup-blue" />
  <img src="https://img.shields.io/badge/DCT-19x%20speedup-orange" />
  <img src="https://img.shields.io/badge/Graph--HEM-171x%20speedup-brightgreen" />
</p>

## 🚀 Highlights

- **171x speedup** with Graph-HEM on 20K×20K matrices
- Three scaling methods for different use cases
- Maintains structural similarity (>0.999 cosine similarity)
- Supports matrices up to 20,000×20,000+

## 📊 Performance Comparison

| Method | Speedup | Memory | Best For |
|--------|---------|--------|----------|
| NN | ~25x | Low O(nnz) | General purpose |
| DCT | ~19x | High O(n²) | High-quality upscaling |
| **Graph-HEM** | **~171x** | Medium O(nnz) | Large matrices |

### Speedup on 20K×20K Matrix (RTX 4060)

| Operation | NN | DCT | Graph-HEM |
|-----------|-----|-----|-----------|
| Expand (+1) | 26.6x | 18.4x | **171x** |
| Upscale (2x) | 25.4x | 20.1x | **166x** |
| Reduce (-1) | 23.5x | 17.2x | **177x** |
| Downscale (1/2x) | 24.8x | 19.6x | **170x** |

## 🛠️ Requirements

- Windows 10/11
- NVIDIA GPU (RTX series recommended)
- [Visual Studio 2022](https://visualstudio.microsoft.com/) with "Desktop development with C++"
- [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cuda-sparse-matrix-scaling.git
cd cuda-sparse-matrix-scaling
```

2. Open **x64 Native Tools Command Prompt for VS 2022**

3. Set library path (adjust version number if needed):
```cmd
set LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64;%LIB%
```

4. Build all methods:
```cmd
nvcc -O3 -arch=sm_89 -o parallel_nn.exe parallel_nn_v2.cu
nvcc -O3 -arch=sm_89 -lcufft -o parallel_dct.exe parallel_dct.cu
nvcc -O3 -arch=sm_89 -o parallel_graph_hem.exe parallel_graph_hem.cu
```

> **Note:** Change `-arch=sm_89` based on your GPU:
> - RTX 4060/4070/4080/4090: `sm_89`
> - RTX 3060/3070/3080: `sm_86`
> - RTX 2060/2070/2080: `sm_75`

## 🎮 Usage

### Basic Usage

```cmd
parallel_nn.exe -s 5000 -v
parallel_dct.exe -s 5000 -v
parallel_graph_hem.exe -s 5000 -v
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-s <size>` | Matrix size | `-s 5000` → 5000×5000 |
| `-b <band>` | Bandwidth | `-b 10` |
| `-r <reps>` | Benchmark repetitions | `-r 10` |
| `-v` | Verification mode (GPU vs CPU) | `-v` |

### Example Output

```
┌───────────────────┬────────────┬───────────┬───────────┬─────────┬────────┐
│ Operation         │ Target     │ GPU (ms)  │ CPU (ms)  │ Speedup │ Errors │
├───────────────────┼────────────┼───────────┼───────────┼─────────┼────────┤
│ Expand (+1)       │ 20001x20001│     1.87  │   319.00  │ 170.59x │      0 │
│ Upscale (2x)      │ 40k x 40k  │     1.89  │   314.00  │ 166.14x │      0 │
│ Reduce (-1)       │ 19999x19999│     1.83  │   325.00  │ 177.60x │      0 │
│ Downscale (1/2x)  │ 10k x 10k  │     1.90  │   322.00  │ 169.47x │      0 │
└───────────────────┴────────────┴───────────┴───────────┴─────────┴────────┘
```

## 🧠 Algorithms

### Nearest Neighbor (NN)
Simple coordinate scaling with rounding. Each nonzero maps to scaled position.

```
(i', j') = (round(i/sr), round(j/sc))
```

### DCT (Discrete Cosine Transform)
Frequency-domain scaling using cuFFT. Best quality for upscaling.

```
1. Forward DCT → 2. Resize coefficients → 3. Inverse DCT → 4. Threshold
```

### Graph-HEM (Novel)
Treats sparse matrix as a graph with 4-connectivity and applies Heavy-Edge Matching.

```
1. Build graph (nonzeros → nodes, 4-connectivity → edges)
2. Find heaviest neighbor for each node
3. Parallel matching with atomicCAS
4. Cluster and scale centroids
```

**Why Graph-HEM is faster:**
- Better parallelism (independent node processing)
- ~37% output reduction through clustering
- Cache-friendly 4-connectivity access patterns
- Efficient atomics on modern GPUs

## 📁 Project Structure

```
cuda-sparse-matrix-scaling/
├── parallel_nn_v2.cu          # Nearest Neighbor (CUDA)
├── parallel_dct.cu            # DCT method (cuFFT)
├── parallel_graph_hem.cu      # Graph-HEM method (Novel)
├── run_tests.bat              # Windows test script
├── README.md                  # This file
├── KULLANIM_KILAVUZU.md       # Turkish documentation
└── paper_updated.pdf          # Academic paper
```

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{taskesen2025parallel,
  title={Parallel Sparse Matrix Scaling Using CUDA: Accelerating Nearest Neighbor, DCT, and Graph-Based Methods},
  author={Ta{\c{s}}kesen, Ali Ahmet and Y{\i}ld{\i}r{\i}m, {\"O}mer},
  institution={Ankara Y{\i}ld{\i}r{\i}m Beyaz{\i}t University},
  year={2025}
}
```

## 📚 References

- [MatGen: A realistic sparse matrix generator](https://doi.org/xxx) - Pamuk et al., 2025
- [Multilevel graph partitioning](https://doi.org/10.1137/S1064827595287997) - Karypis & Kumar, 1998
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) - Davis & Hu, 2011

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ali Ahmet Taşkesen** - [aliahmetaskesen@gmail.com](mailto:aliahmetaskesen@gmail.com)
- **Ömer Yıldırım** - [flashomer@gmail.com](mailto:flashomer@gmail.com)

*Ankara Yıldırım Beyazıt University - Parallel Programming Course - 2025*

# ============================================================================
# Makefile for Parallel NN Sparse Matrix Scaling
# ============================================================================
# Authors: Ali Ahmet Taşkesen, Ömer Yıldırım
# ============================================================================

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 -lineinfo

# For older GPUs (e.g., GTX 1060, GTX 1080)
# NVCC_FLAGS = -O3 -arch=sm_61 -lineinfo

# For newer GPUs (e.g., RTX 3000 series)
# NVCC_FLAGS = -O3 -arch=sm_86 -lineinfo

# For Tesla V100
# NVCC_FLAGS = -O3 -arch=sm_70 -lineinfo

# Targets
TARGETS = parallel_nn

# Default target
all: $(TARGETS)

# Main program
parallel_nn: parallel_nn_v2.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean
clean:
	rm -f $(TARGETS) *.o

# Run with default settings
run: parallel_nn
	./parallel_nn -s 5000 -b 10 -v

# Run with large matrix
run-large: parallel_nn
	./parallel_nn -s 20000 -b 15

# Run scalability test
scalability: parallel_nn
	@echo "=== Scalability Test ==="
	@for size in 1000 2000 5000 10000 20000; do \
		echo "\n--- Size: $$size ---"; \
		./parallel_nn -s $$size -b 10 -r 5; \
	done

# Help
help:
	@echo "Parallel NN Sparse Matrix Scaling"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build the program"
	@echo "  make run       - Run with default settings (5000x5000, verify)"
	@echo "  make run-large - Run with large matrix (20000x20000)"
	@echo "  make scalability - Run scalability tests"
	@echo "  make clean     - Remove compiled files"
	@echo ""
	@echo "Compiler flags can be changed for different GPU architectures:"
	@echo "  sm_61 - GTX 1060, 1070, 1080"
	@echo "  sm_70 - Tesla V100, Titan V"
	@echo "  sm_75 - RTX 2000 series"
	@echo "  sm_80 - A100"
	@echo "  sm_86 - RTX 3000 series"
	@echo "  sm_89 - RTX 4000 series"

.PHONY: all clean run run-large scalability help

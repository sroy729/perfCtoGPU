# GPU Compiler and flags
NVCC = nvcc
NVCCFLAGS = -arch=sm_86 -lcublas  # Adjust 'sm_86' for your GPU, e.g., sm_86 for RTX 3050


# GPU Source files (if any)
GPU_SRCS = opt_matmul \
		   lib_matmul


# GPU Executables
GPU_TARGETS = opt_matmul \
			  lib_matmul

# Matrix size (default value)
SIZE ?= 10

# Default rule
all: $(GPU_TARGETS)

# GPU targets
$(GPU_TARGETS): %: %.cu
	$(NVCC) $(NVCCFLAGS) -DSIZE=$(SIZE) $< -o $@

# Clean rule
clean:
	rm -f $(GPU_TARGETS)

# Phony targets
.PHONY: all clean


# Compiler and flags
CXX = g++
CXXFLAGS=-O0
SIMDFLAGS=-mavx2 -march=native

# Source files
SRCS = vanilla_matrix_mul.cpp \
	   loop_reordering.cpp \
	   blocking.cpp \
	   simd.cpp

# Executables (one for each source file)
TARGETS = vanilla_matrix_mul \
		  loop_reordering \
		  blocking \
		  simd

SIZE ?= 10

# Default rule
all: $(TARGETS)

# Compile each source file into its executable
$(TARGETS): %: %.cpp
	$(CXX) $(CXXCFLAGS) -DSIZE=$(SIZE) $< -o $@

simd: simd.cpp
	$(CXX) $(CXXCFLAGS) $(SIMDFLAGS) -DSIZE=$(SIZE) $< -o $@

# Clean up build artifacts
clean:
	rm -f $(TARGETS)

# Phony targets
.PHONY: all clean

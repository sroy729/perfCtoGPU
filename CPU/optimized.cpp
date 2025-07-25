#include <iostream>
#include <stdlib.h>
#include <chrono> 
#include <omp.h>
#include <immintrin.h> // AVX2
#include <cstring>     // for memset

#ifndef SIZE
#define SIZE 2048
#endif

#define dataType double
#define SEED 69
#define BLOCK_SIZE 64 // Adjust based on L1/L2 cache size
#define ALIGNMENT 32  // AVX2 alignment for double (32 bytes = 4 doubles)

using namespace std;

void matrix_mult(const double *A, const double *B, double *C, int size) {
    // Zero initialize result matrix
    memset(C, 0, size * size * sizeof(double));
    
    #pragma omp parallel for
    for (int i0 = 0; i0 < size; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < size; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < size; k0 += BLOCK_SIZE) {
                for (int i = i0; i < std::min(i0 + BLOCK_SIZE, size); ++i) {
                    for (int j = j0; j < std::min(j0 + BLOCK_SIZE, size); j += 4) {
                        // Check bounds for vectorization
                        if (j + 3 < size) {
                            __m256d c_vec = _mm256_loadu_pd(&C[i * size + j]);
                            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, size); ++k) {
                                __m256d a_vec = _mm256_set1_pd(A[i * size + k]);
                                __m256d b_vec = _mm256_loadu_pd(&B[k * size + j]);
                                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                            }
                            _mm256_storeu_pd(&C[i * size + j], c_vec);
                        } else {
                            // Handle remaining elements without vectorization
                            for (int jj = j; jj < std::min(j0 + BLOCK_SIZE, size); ++jj) {
                                for (int k = k0; k < std::min(k0 + BLOCK_SIZE, size); ++k) {
                                    C[i * size + jj] += A[i * size + k] * B[k * size + jj];
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

/* Function to randomly initialize a matrix */
void init_matrix(dataType *arr, int size) {
    srand(SEED);
    for(int i = 0; i < size * size; i++){  // ✅ Initialize all SIZE*SIZE elements
        arr[i] = rand() % (size + 1); 
    }
}

/* Simple verification function */
void verify_result(const double *A, const double *B, const double *C, int size) {
    // Check a few elements with naive multiplication
    bool correct = true;
    for (int i = 0; i < std::min(3, size); i++) {
        for (int j = 0; j < std::min(3, size); j++) {
            double expected = 0.0;
            for (int k = 0; k < size; k++) {
                expected += A[i * size + k] * B[k * size + j];
            }
            if (abs(C[i * size + j] - expected) > 1e-10) {
                correct = false;
                cout << "Mismatch at (" << i << "," << j << "): got " 
                     << C[i * size + j] << ", expected " << expected << endl;
            }
        }
    }
    cout << "Verification: " << (correct ? "PASSED" : "FAILED") << endl;
}

int main(){
    cout << "Matrix size: " << SIZE << "x" << SIZE << endl;
    cout << "Total operations: ~" << (2.0 * SIZE * SIZE * SIZE / 1e9) << " billion" << endl;
    
    dataType* A = new dataType[SIZE*SIZE];
    dataType* B = new dataType[SIZE*SIZE];
    dataType* C = new dataType[SIZE*SIZE];
    
    cout << "Initializing matrices..." << endl;
    init_matrix(A, SIZE);
    init_matrix(B, SIZE);
    
    cout << "Starting matrix multiplication..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    matrix_mult(A, B, C, SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    cout << "Execution Time: " << duration.count() << " ms" << endl;
    
    // Calculate GFLOPS
    double gflops = (2.0 * SIZE * SIZE * SIZE) / (duration.count() * 1e6);
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    
    // Verify correctness for small matrices
    if (SIZE <= 100) {
        verify_result(A, B, C, SIZE);
    }
    
    delete[] A;
    delete[] B; 
    delete[] C;
    return 0; 
}

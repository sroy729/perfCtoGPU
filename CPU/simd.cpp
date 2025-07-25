#include <iostream>
#include <stdlib.h>
#include <chrono> 
#include <immintrin.h> // for AVX2 intrinsics
#include <algorithm>

#ifndef SIZE
#define SIZE 10
#endif
#define dataType uint32_t
#define SEED 69
#define TILE_SIZE 64

using namespace std;

void matrix_mult(dataType *A, dataType *B, dataType *C){
	int size = SIZE;

    // Initialize C to zero
    for (int i = 0; i < size * size; i++) {
        C[i] = 0;
    }

    for (int ii = 0; ii < size; ii += TILE_SIZE) {
        for (int jj = 0; jj < size; jj += TILE_SIZE) {
            for (int kk = 0; kk < size; kk += TILE_SIZE) {
                for (int i = ii; i < min(ii + TILE_SIZE, size); i++) {
                    for (int k = kk; k < min(kk + TILE_SIZE, size); k++) {
                        __m256i a_vec = _mm256_set1_epi32(A[i * size + k]);  // broadcast A[i][k]

                        int j = jj;
                        for (; j <= min(jj + TILE_SIZE, size) - 8; j += 8) {
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)&B[k * size + j]);
                            __m256i c_vec = _mm256_loadu_si256((__m256i*)&C[i * size + j]);

                            __m256i prod = _mm256_mullo_epi32(a_vec, b_vec);
                            __m256i sum = _mm256_add_epi32(c_vec, prod);

                            _mm256_storeu_si256((__m256i*)&C[i * size + j], sum);
                        }

                        // Handle remaining elements (scalar)
                        for (; j < min(jj + TILE_SIZE, size); j++) {
                            C[i * size + j] += A[i * size + k] * B[k * size + j];
                        }
                    }
                }
            }
        }
    }	

}

/* Function to randomly initize an array */
void init_arr(dataType *arr) {
	srand(SEED);
	for(int i =0; i<SIZE; i++){
		arr[i] = rand() % (SIZE + 1); 
	}
}

int main(){
	dataType* A = new dataType[SIZE*SIZE];
	dataType* B = new dataType[SIZE*SIZE];
	dataType* C = new dataType[SIZE*SIZE];
	init_arr(A);
	init_arr(B);

	auto start = std::chrono::high_resolution_clock::now();
	matrix_mult(A, B, C);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;

	cout<<"Execution Time: "<<duration.count()<<" ms\n";

	delete A;
	delete B;
	delete C;
	return 0; 
}

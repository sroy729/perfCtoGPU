#include <iostream>
#include <stdlib.h>
#include <chrono> 
#include <emmintrin.h> // for AVX2 intrinsics
#include <algorithm>

#ifndef SIZE
#define SIZE 10
#endif
#define dataType double
#define SEED 69

using namespace std;

void matrix_mult(dataType *A, dataType *B, dataType *C){
	int size = SIZE;


    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; j += 2) {
            __m128d sum = _mm_setzero_pd(); // Holds 2 double-precision values

            for (int k = 0; k < size; ++k) {
                // Load A[i][k] and broadcast it
                __m128d a_val = _mm_set1_pd(A[i * size + k]);

                // Load two values from B[k][j] and B[k][j+1]
                __m128d b_val = _mm_loadu_pd(&B[k * size + j]);

                // Multiply and accumulate
                sum = _mm_add_pd(sum, _mm_mul_pd(a_val, b_val));
            }

            // Store the result to C[i][j] and C[i][j+1]
            _mm_storeu_pd(&C[i * size + j], sum);
        }
    }

}

/* Function to randomly initize an array */
void init_arr(dataType *arr) {
	srand(SEED);
	for(int i =0; i<SIZE*SIZE; i++){
		arr[i] = rand() % (SIZE + 1); 
	}
}

int main(){
	dataType* A = new dataType[SIZE*SIZE];
	dataType* B = new dataType[SIZE*SIZE];
	dataType* C = new dataType[SIZE*SIZE];
	init_arr(A);
	init_arr(B);
	// Initialize C to 0
    for (int i = 0; i < SIZE * SIZE; ++i) {
        C[i] = 0.0;
    }

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

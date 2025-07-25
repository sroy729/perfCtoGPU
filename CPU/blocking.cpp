#include <iostream>
#include <stdlib.h>
#include <chrono> 

#ifndef SIZE
#define SIZE 16
#endif
#define dataType uint32_t
#define SEED 69
#define TILE_SIZE 64

using namespace std;

void matrix_mult(dataType *A, dataType *B, dataType *C){
	//dataType size = SIZE;

	//for(int ii = 0; ii < size; ii += TILE_SIZE) {
	//	for(int jj = 0; jj < size; jj += TILE_SIZE) {
	//		for(int kk = 0; kk < size; kk += TILE_SIZE) {
	//			for(int i = ii; i < ii + TILE_SIZE && i<size; i++) {
	//				for(int k = kk; k < kk + TILE_SIZE && k<size; k++) {
	//					for(int j = jj; j < jj + TILE_SIZE &&  j<size; j++) {
	//						C[i*size+j] += A[i*size+k]*B[k*size+j];
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	dataType size = SIZE;

    for (int ii = 0; ii < size; ii += TILE_SIZE) {
        for (int jj = 0; jj < size; jj += TILE_SIZE) {
            for (int kk = 0; kk < size; kk += TILE_SIZE) {
                for (int i = ii; i < ii + TILE_SIZE && i < size; i++) {
                    for (int j = jj; j < jj + TILE_SIZE && j < size; j++) {
                        dataType res = C[i * size + j];
                        for (int k = kk; k < kk + TILE_SIZE && k < size; k++) {
                            res += A[i * size + k] * B[k * size + j];
                        }
                        C[i * size + j] = res;
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

	for (int i = 0; i < SIZE * SIZE; ++i) C[i] = 0;

	auto start = std::chrono::high_resolution_clock::now();
	matrix_mult(A, B, C);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> duration = end - start;

	cout<<"Execution Time: "<<duration.count()<<" ms\n";

	delete[] A;
	delete[] B;
	delete[] C;
	return 0; 
}

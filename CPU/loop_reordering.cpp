#include <iostream>
#include <stdlib.h>
#include <chrono> 

#ifndef SIZE
#define SIZE 10
#endif
#define dataType uint32_t
#define SEED 69

using namespace std;

void matrix_mult(dataType *A, dataType *B, dataType *C){
	dataType size = SIZE;

	for(int i = 0; i<size; i++){						// select a row in A
		for(int k = 0; k<size; k++){					// select a col in B
			for(int j = 0; j<size; j++){				// no. of operation for ele in C
				C[i*size+j] += A[i*size+k]*B[k*size+j];
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

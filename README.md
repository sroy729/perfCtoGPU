# perfCtoGPU

The aim to realise the performance gain of a matrix multiplication code when we go from a CPU to GPU.
running the code 
```
python3 plot.py <config>
```


The GPU under test is RTX 3050 mobile.

The CPU under test is **AMD Ryzen 7.**
```
===== CPU Info =====
CPU Model: AMD Ryzen 7 5800H with Radeon Graphics
Vendor: AuthenticAMD
Physical Cores: 8
Logical Cores: 16

===== Cache Sizes =====
L1d cache:                            256 KiB
L1i cache:                            256 KiB
L2 cache:                             4 MiB
L3 cache:                             16 MiB

===== Prefetchers =====
Prefetcher check not supported for vendor: AuthenticAMD (likely AMD)
Hint: Check BIOS/UEFI settings for prefetcher controls on AMD CPUs.
```

Metrics to record: 
- Execution time
- Number of instruction 
- CPU cache accesses
	- **L1D** : `Cache hit`, `Cache miss`, `MPKI(cal)` 
	- **L2** :  `Cache hit`, `Cache miss`, `MPKI(cal)`
	- **LLC** : `Cache hit`, `Cache miss`, `MPKI(cal)` 

Code version: 
**CPU: Optimisation strategies**
	`vanilla` 
	**Data Layout and locality :** 
		`row majow and column major[rcm]`
		`blocking`/`tiling`/
	`loop_optimization` 
	`tiling`
	`stencile`
	`blocking`
	`loop_unrolling`
	`software_prefetching`
	`SIMD`

**CPU: Optimisation strategies:** 
	`vanilla`
	**Data layout and locality:**
		`loop_reordering`
		`loop_unrolling`
	**Cache optimisation:**
		`blocking`/`tiling`
		`prefetching`
	**SIMD**: 
		`simd`
	**Compiler and instruction-level optimisation:**
	**Multi-threading and parallelism**
	**Algorithmic Improvements**
	 
	
### Vanilla Matrix multiplication code: `vanilla`

The code in very simple. There are three loops 
- first loop iterates over the number of row in A
- second loop iterates over the number of column
- third loop is for operation done generating each element in C, using A and B
![[figures/mat_mul.svg]] 



```c++
	for(int i = 0; i<size; i++){						// select a row in A
		for(int j = 0; j<size; j++){					// select a col in B
			for(int k = 0; k<size; k++){				// no. of operation for ele in C
				C[i*size+j] += A[i*size+k]*B[k*size+j];
			}
		}
	}
```

### Loop reordering: `loop_reordering`
The matrix is stored in a row major format, i.e. storing order will be A0, A1, A2, A3, A4 ... , all of these indexes will have locality to it.
So exploiting that we will access array B and C in that way only. The access to array A row will remain same, but instead of accesing each column and computing a element of C; we will go over the matrix B in rowise and calculate the partial sum of C in each pass of row.

To achieve this just switch the for loops of j and k. 
```c++
for(int i = 0; i<size; i++){						// select a row in A
		for(int j = 0; j<size; j++){					// select a col in B
			for(int k = 0; k<size; k++){				// no. of operation for ele in C
				C[i*size+j] += A[i*size+k]*B[k*size+j];
			}
		}
	}
```

![[figures/execution_time_plot.png]]


### Blocking/Tiling: `blocking`

The idea is rather than working on the whole matrix, work on a smaller sub-matrix.
![[figures/blocking.png]]

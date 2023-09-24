%%cu

#include <stdio.h>

#define N 4096
#define SEGMENT_SIZE 32

///////////////////////////////////////////////////////////////////////////////
//
// Computes the scalar product of two vectors of N elements on GPU.
//
///////////////////////////////////////////////////////////////////////////////
__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < nElem) {
		C[tid] = A[tid]*B[tid];
	}
}
/////////////////////////////////////////////////////////////////
//
// Computes a standard parallel reduction on GPU.
//
/////////////////////////////////////////////////////////////////


__global__ void vectorReduce(float *R, const float *C, int nElem)
{
	// Reserva de espacio en la zona de memoria compartida
	extern __shared__ float sdata[];
	
	// Indice local de cada hilo -> kernel con un solo bloque
	int tid = threadIdx.x;
	
	// Copiamos en 'temporal' el vector y sincronizamos	
	sdata[tid] = C[tid];
	
	__syncthreads();
	
	//Reduccion paralela
	int salto = nElem/2;
	
	// Realizamos log2(N) iteraciones
	while(salto){
		// Solo trabajan la mitad de los hilos
		if(tid < salto){
			sdata[tid] = sdata[tid] + sdata[tid+salto];
		}
		__syncthreads();
		salto = salto/2;
	}
	
	// El hilo no.'0' escribe el resultado final en la memoria global
	if(tid==0){
		*R = sdata[tid];
    //printf("La posicion %d de R va a ser %f\n",iter,R[0]);
	}
}

// -----------------------------------------------
// Host Utility Routines
// -----------------------------------------------
float scalarProd_CPU(float *A, float *B, int nElem) {
    float suma = 0.0f;
    for (int i = 0; i < nElem; i++) {
        suma += A[i] * B[i];
    }
    return suma;
}

float randFloat(float low, float high) {
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main(void) {
    // Array Elements
    int n_elem = N;

    // Block Dimension
    int block_dim = SEGMENT_SIZE;

    // Number of Blocks
    int n_block = (N % block_dim == 0) ? (N / block_dim) : (N / block_dim) + 1;

    // Execution Configuration Parameters
    dim3 blocks(n_block);
    dim3 threads(block_dim);

    // Size (in bytes) Required to Store the Matrix
    size_t n_bytes = (n_elem * sizeof(float));

    // Allocate Host Memory
    float *h_A, *h_B, *h_R;
    cudaMallocHost((void **)&h_A, n_bytes);
    cudaMallocHost((void **)&h_B, n_bytes);
    cudaMallocHost((void **)&h_R, n_block * sizeof(float));

    // Initialize Host Data
    srand(123);

    // Generating input data on CPU
    for (int i = 0; i < n_elem; i++) {
        h_A[i] = randFloat(0.0f, 1.0f);
        h_B[i] = randFloat(0.0f, 1.0f);
    }

    // Compute Reference CPU Solution
    float result_cpu = scalarProd_CPU(h_A, h_B, n_elem);

    // CUDA Events
    cudaEvent_t start, stop;

    // Allocate Device Memory
    float *d_A, *d_B, *d_C, *d_R;
    cudaMalloc((void **)&d_A, n_bytes);
    cudaMalloc((void **)&d_B, n_bytes);
    cudaMalloc((void **)&d_C, n_bytes);
    cudaMalloc((void **)&d_R, n_block * sizeof(float));

    // Init Events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start Time Measurement
    cudaEventRecord(start, 0);

    // Copy Host Data to Device
    cudaMemcpyAsync(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

    // Launch kernels asynchronously
    scalarProd<<<blocks, threads, 0>>>(d_C, d_A, d_B, n_elem);
    vectorReduce<<<blocks, threads, block_dim * sizeof(float)>>>(d_R, d_C, n_elem);

    // Copy Device Data to Host
    cudaMemcpyAsync(h_R, d_R, n_block * sizeof(float), cudaMemcpyDeviceToHost);

    // End Time Measurement
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    printf("Execution Time by the GPU: %.2f\n", kernel_time);

    float result_gpu = 0.0f;
    for (int i = 0; i < n_block; i++) {
        result_gpu += h_R[i];
    }

    // Free Host Memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_R);

    // Free Device Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_R);

    // Destroy Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float tolerance = 1e-5;
    if (fabs(result_cpu - result_gpu) < tolerance) {
        printf("Test Failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test Passed\n");
    exit(EXIT_SUCCESS);
}


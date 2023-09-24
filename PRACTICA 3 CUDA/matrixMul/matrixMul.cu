%%cu
/**
 * Matrix Multiplication: C = A * B.
 *
 * This file contains both device and host code to compute a matrix multiplication.
 *
 */

#include <math.h>
#include <stdio.h>

#define MATRIX_DIM	 32
#define SEGMENT_SIZE 64
// --------------------
// Device Kernels
// --------------------
__global__ void transposeMatrix(float *d_data, int mat_dim) {

	
	// Array in Shared Memory
	extern __shared__ float sdata[];
	
	int tid_b = threadIdx.x;
	int tid_g = blockIdx.x*gridDim.x*blockDim.x+blockIdx.y*blockDim.x+tid_b;
  

	for (int i=0;i<blockDim.x;i++)
    //printf("El hilo %d al transponer en shared es %d y en data es %d\n",tid_g,tid_b+i*blockDim.x,blockIdx.x*MATRIX_DIM+tid_g+i*MATRIX_DIM);  
		sdata[tid_b+i*blockDim.x] = d_data[blockIdx.x*MATRIX_DIM+tid_g+i*MATRIX_DIM];  
	
  	__syncthreads();
  
	tid_b = threadIdx.x;
  	tid_g = blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+tid_b;

	for (int i=0;i<blockDim.x;i++){
    //printf("El hilo %d en shared es %d y en data es %d\n",tid_g,tid_b*blockDim.x+i,blockIdx.y*MATRIX_DIM+tid_g+i*MATRIX_DIM);  
    d_data[blockIdx.y*MATRIX_DIM+tid_g+i*MATRIX_DIM] = sdata[tid_b*blockDim.x+i];  
  } 
 
}

__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < nElem) {
		C[tid] = A[tid]*B[tid];
	}
}

__global__ void vectorReduce(float *R, const float *C, int nElem,int iter)
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

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const float *A, const float *B, float *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float acum = 0.0;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(float *h_C, float *d_C, int n)
{
	double eps = 1e-4;
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
			return false;
		}
	}
	return true;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0 - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {
	
	// Matrix Dimensions
	int dim_x = MATRIX_DIM;
	int dim_y = dim_x;
	
	// Matrix Size
	int mat_size = dim_x * dim_y;
	
	// Block Dimension
	int block_dim = SEGMENT_SIZE;
	
	// Number of Blocks
	int n_block = (dim_x % block_dim ==0)?(dim_x/block_dim): (dim_x/block_dim)+1;
	
	// Execution Configuration Parameters
	dim3 blocksPerGrid(n_block, n_block);
	dim3 threadsPerBlock(block_dim);
	
	// Size Required to Store the Matrix
	size_t n_bytes = (mat_size * sizeof(float));
	
	// Allocate Pinned Host Memory
	float *h_A, *h_B, *h_C, *h_R;
	cudaMallocHost((void**)&h_A, n_bytes);
	cudaMallocHost((void**)&h_B, n_bytes);
	cudaMallocHost((void**)&h_C, n_bytes);
	cudaMallocHost((void**)&h_R, n_bytes);
	
	// Initialize Host Data
	srand(123);
	
	// Generating input data on CPU
	for (int i=0; i < mat_size; i++) {
		h_A[i] = randFloat(0.0, 1.0);
		h_B[i] = randFloat(0.0, 1.0);
	}
	
	// Compute Reference Matrix Multiplication
	matrixMul(h_A, h_B, h_C, dim_x);
	
	//Debugging para comprobar que la matriz host resultado es igual a la del device
	/*
	printf("La matriz h_C es:\n");
	for(int i=0;i<dim_x;i++){
			for(int j=0;j<dim_y;j++)
					printf("%f ",h_C[i]);
			printf("\n");
	}*/

  //Porque collab emplea Tesla T4
  int nStreams=30;


	cudaStream_t *stream = (cudaStream_t *) malloc(nStreams * sizeof(cudaStream_t));

	for (int i = 0; i < nStreams; i++) {
		cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking);   // create streams
	}
	
	// Performance Data
	float kernel_time, kernel_bandwidth;
	
	// Allocate Device Memory
	float *d_A, *d_B, *d_C, *d_R;
	cudaMalloc((void**)&d_A, n_bytes);
	cudaMalloc((void**)&d_B, n_bytes);
	cudaMalloc((void**)&d_C, n_bytes);
	cudaMalloc((void**)&d_R, n_bytes);
	
	// CUDA Events
	cudaEvent_t start, stop;
	
	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Start Time Measurement
	cudaEventRecord(start, stream[0]);
	
	// Copy Host Data to Device
  cudaMemcpyAsync(d_A, h_A, n_bytes, cudaMemcpyHostToDevice , stream[0]);
  cudaMemcpyAsync(d_B, h_B, n_bytes, cudaMemcpyHostToDevice , stream[0]);

	
	cudaStreamSynchronize(stream[0]);

	transposeMatrix<<<blocksPerGrid, threadsPerBlock, block_dim * block_dim * sizeof(float) , stream[0]>>>(d_B, dim_x);
  cudaStreamSynchronize(stream[0]);


	for(int i = 0; i < dim_y; i++) {
		for(int j = 0; j < dim_x; j++) {
			scalarProd<<<n_block, block_dim, block_dim * sizeof(float),stream[i%nStreams]>>>(d_C, &d_A[i * dim_x], &d_B[j * dim_x], dim_x);
			vectorReduce<<<n_block, block_dim, block_dim * sizeof(float),stream[(i+1)%nStreams]>>>(&d_R[i*dim_y+j],d_C, dim_x,i*dim_y+j);
			cudaStreamSynchronize(stream[i % nStreams]);
			cudaStreamSynchronize(stream[(i+1) % nStreams]);
		}
	}
	cudaDeviceSynchronize();
	
	// Copy Device Data to Host
	cudaMemcpyAsync(h_R, d_R, n_bytes, cudaMemcpyDeviceToHost, stream[0]);

  cudaStreamSynchronize(stream[0]);
	
	// End Time Measurement
	cudaEventRecord(stop, stream[0]);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&kernel_time, start, stop);
	
	//Debugging para comprobar que la matriz device resultado es igual a la del host
	/*printf("La matriz h_R es:\n");
	for(int i=0;i<dim_x;i++){
			for(int j=0;j<dim_y;j++)
					printf("%f ",h_R[i]);
			printf("\n");
	}*/

	bool res = compareData(h_C, h_R, dim_x);

	if (res == true) {
		// Report Effective Bandwidth
		kernel_bandwidth = (2.0 * 1000.0 * n_bytes)/(1024 * 1024 * 1024);
		kernel_bandwidth /= kernel_time;
		
		printf( "Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
				 kernel_bandwidth, kernel_time, (dim_x * dim_y) );
	}
	
	// Free Host Memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFreeHost(h_R);
	
	// Free Device Memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_R);
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Destroy Streams
  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(stream[i]);
  }
  
	if (res == false) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);

}

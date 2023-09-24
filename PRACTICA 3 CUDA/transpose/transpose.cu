%%cu

/*
 * Matrix Transpose
 *
 * This file contains both device and host code for transposing a matrix.
 *
 */

#include <stdio.h>
 
#define MATRIX_DIM	 6
#define SEGMENT_SIZE 6

///////////////////////////////////////////////////////////
//
// Computes the Transpose of a Matrix
//
///////////////////////////////////////////////////////////
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

// ---------------------
// Host Utility Routines
// ---------------------
void transpose(float *At, float *A, const int dim_x, const int dim_y)
{
	for (int y = 0; y < dim_y; y++) {
		for (int x = 0; x < dim_x; x++) {
			At[(x * dim_y) + y] = A[(y * dim_x) + x];
		}
	}
}

bool compareData(float *d_data, float *h_data, int n) {
	for (int i = 0; i < n * n; i++) {
		if (d_data[i] != h_data[i]) {
			return false;
		}
	}
	return true;
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Matrix Dimensions
	int dim_x = MATRIX_DIM;
	int dim_y = dim_x;
	
	// Block Dimension
	int block_dim = SEGMENT_SIZE;
	
	// Number of Blocks
	int n_block = (dim_x % block_dim == 0) ? (dim_x / block_dim) : (dim_x / block_dim) + 1;

	// Execution Configuration Parameters
	dim3 blocksPerGrid(n_block,n_block);
	dim3 threadsPerBlock(block_dim);

	// Size (in bytes) Required to Store the Matrix
	size_t n_bytes = (dim_x * dim_y * sizeof(float));
	
	// Allocate Host Memory
	float *A = (float *) malloc(n_bytes);
	float *At = (float *) malloc(n_bytes);
	float *Aux  = (float *) malloc(n_bytes);
	
	// Initialize Host Data
	for (int i = 0; i < (dim_x * dim_y); i++) {
		A[i] = (float) i;
	}
	
	// Compute Reference Transpose Solution
	transpose(At, A, dim_x, dim_y);
	/*
  for(int i=0;i<dim_x*dim_y;i++){
          printf("La posicion %d es de la matriz At es %f \n",i,At[i]);
      }*/

	// CUDA Events
	cudaEvent_t start, stop;
	
	// Performance Data
	float kernel_time, kernel_bandwidth;
	
	// Allocate Device Memory
	float *d_data;
	cudaMalloc((void**)&d_data,n_bytes);

	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// Start Time Measurement
    cudaEventRecord(start, 0);
	
	// Copy Host Data to Device
	cudaMemcpy(d_data,A,n_bytes,cudaMemcpyHostToDevice);
	
    transposeMatrix<<<blocksPerGrid,threadsPerBlock,dim_x*dim_x*sizeof(float)>>>(d_data, dim_x*dim_y);
	
	// Copy Device Data to Host
	
	cudaMemcpy(Aux,d_data,n_bytes,cudaMemcpyDeviceToHost);
    
	// End Time Measurement
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);

  /*for(int i=0;i<dim_x*dim_y;i++){
          printf("La posicion %d es de la matriz transpuesta es %f\n",i,At[i]);
      }*/

	bool res = compareData(Aux, At, dim_x);
	
	if (res == true) {
		// Report Effective Bandwidth
		kernel_bandwidth = (2.0f * 1000.0f * n_bytes)/(1024 * 1024 * 1024);
		kernel_bandwidth /= kernel_time;
		
		printf( "Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
				kernel_bandwidth, kernel_time, (dim_x * dim_y) );
	}
	
	// Free Host Memory
	free(A); free(At); free(Aux);
	
	// Free Device Memory
	cudaFree(d_data);
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	if (res == false) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}



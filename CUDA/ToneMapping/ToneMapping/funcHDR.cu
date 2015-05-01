#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define BLOCK_SIZE 1024

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

/// Reduction kernel to find the min and max value of an Array
__global__
void findMinMax(const float *d_in_min, const float *d_in_max, float *d_out_min, float *d_out_max)
{
	/// Share memory
	__shared__ float ds_min[BLOCK_SIZE];
	__shared__ float ds_max[BLOCK_SIZE];

	/// Which thread is?
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x;
	int t2;
	float aux;

	ds_min[threadIdx.x] = d_in_min[t];
	ds_max[threadIdx.x] = d_in_max[t];

	__syncthreads();

	while (totalThreads > 1)
	{
		int halfPoint = (totalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (threadIdx.x < halfPoint)
		{
			t2 = threadIdx.x + halfPoint;

			// Get the shared value stored by another thread
			aux = ds_min[t2];
			if (aux < ds_min[threadIdx.x])
				ds_min[threadIdx.x] = aux;

			aux = ds_max[t2];
			if (aux > ds_max[threadIdx.x])
				ds_max[threadIdx.x] = aux;
		}
		__syncthreads();

		// Reducing the binary tree size by two:
		totalThreads = halfPoint;
	}
	//Save the min and max at the block
	if (threadIdx.x == 0){
		d_out_min[blockIdx.x] = ds_min[0];
		d_out_max[blockIdx.x] = ds_max[0];
	}

}

void calculate_cdf(const float* const d_logLuminance, unsigned int* const d_cdf, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols, const size_t numBins)
{
	/// Calcule the size of grid and block
	int blockSize = BLOCK_SIZE;
	int gridSize = ceil((float)(numRows*numCols) / (float) BLOCK_SIZE); // Upper round

	///	1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance

	/// Declare and allocate variables to calculate the max and min values and allocate the memory for it.
	float *d_min, *d_max, *d_min_aux, *d_max_aux;
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float) * BLOCK_SIZE));
	checkCudaErrors(cudaMalloc(&d_max, sizeof(float) * BLOCK_SIZE));

	/// Execute kernels
	findMinMax << <gridSize, blockSize >> >(d_logLuminance, d_logLuminance, d_min, d_max);
	printf("Raquel => min = %d  -- max = %d\n", d_min, d_max);

	while (gridSize>1)
	{
		//Se reduce el tamaño del grid BLOCKSIZE veces
		gridSize = ceil((float)gridSize / (float)BLOCK_SIZE);
		
		//Iteración Minimo
		checkCudaErrors(cudaMalloc(&d_min_aux, sizeof(float)*gridSize));
		checkCudaErrors(cudaMalloc(&d_max_aux, sizeof(float)*gridSize));
		
		findMinMax << <gridSize, blockSize >> >(d_min, d_max, d_min_aux, d_max_aux);

		d_min = d_min_aux;
		d_max = d_max_aux;
		
	}

	/// Copy to Host the results of max and min and clean memory
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min_aux, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max_aux, sizeof(float), cudaMemcpyDeviceToHost));

	/// Free memory
	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_min_aux));
	checkCudaErrors(cudaFree(d_max_aux));

	///	2) Obtener el rango a representar
	float range = max_logLum - min_logLum;

	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//		bin = (Lum [i] - lumMin) / lumRange * numBins

	/// Recalcule the size of grid and block
	gridSize = ceil((float)(numRows*numCols) / (float)BLOCK_SIZE);
	blockSize = BLOCK_SIZE;

	/// Declare, allocate and zero histogram variable
	int* d_histogram;
	cudaMalloc(&d_histogram, sizeof(int)*numBins);
	cudaMemset(d_histogram, 0, sizeof(int)*numBins);


	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//	de los valores de luminancia. Se debe almacenar en el puntero c_cdf

}

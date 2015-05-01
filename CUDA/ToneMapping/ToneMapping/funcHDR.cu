#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define BLOCK_SIZE 32

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
	__shared__ double min[BLOCK_SIZE];
	__shared__ double max[BLOCK_SIZE];

	/// Which thread is?
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = blockDim.x;
	int t2;
	float aux;

	min[threadIdx.x] = d_in_min[t];
	max[threadIdx.x] = d_in_max[t];

	__syncthreads();

	while (totalThreads > 1)
	{
		int halfPoint = (totalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (threadIdx.x < halfPoint)
		{
			t2 = threadIdx.x + halfPoint;

			// Get the shared value stored by another thread
			aux = min[t2];
			if (aux < min[threadIdx.x])
				min[threadIdx.x] = aux;

			aux = max[t2];
			if (aux > max[threadIdx.x])
				max[threadIdx.x] = aux;
		}
		__syncthreads();

		// Reducing the binary tree size by two:
		totalThreads = halfPoint;
	}
	//Save the min and max at the block
	if (threadIdx.x == 0){
		d_out_min[blockIdx.x] = min[0];
		d_out_max[blockIdx.x] = max[0];
	}

}


void calculate_cdf(const float* const d_logLuminance, unsigned int* const d_cdf, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols, const size_t numBins)
{
	int blockSize = BLOCK_SIZE;
	int gridSize = (numRows*numCols) / BLOCK_SIZE;

	///	1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance

	/// Variables to calculate the max and min values and allocate the memory for it.
	float *d_min, *d_max, *d_min_aux, *d_max_aux;
	checkCudaErrors(cudaMalloc(&d_min0, sizeof(float) * BLOCK_SIZE));
	checkCudaErrors(cudaMalloc(&d_max0, sizeof(float) * BLOCK_SIZE));

	/// Execute kernels
	findMinMax << <gridSize, blockSize >> >(d_logLuminance, d_logLuminance, d_min, d_max);

	while (gridSize>1)
	{
		//Se reduce el tamaño del grid BLOCKSIZE veces
		gridSize = gridSize / BLOCK_SIZE;

		//Iteración Minimo
		checkCudaErrors(cudaMalloc(&d_min_aux, sizeof(float)*gridSize));
		checkCudaErrors(cudaMalloc(&d_max_aux, sizeof(float)*gridSize));
		
		findMinMax << <gridSize, blockSize >> >(d_min, d_max, d_min_aux, d_max_aux);

		d_min = d_min_aux;
		d_max = d_max_aux;
		
	}

	/// Copy to Host the results of max and min and clean memory
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_max));

	//2) Obtener el rango a representar

	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//		bin = (Lum [i] - lumMin) / lumRange * numBins

	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//	de los valores de luminancia. Se debe almacenar en el puntero c_cdf

}

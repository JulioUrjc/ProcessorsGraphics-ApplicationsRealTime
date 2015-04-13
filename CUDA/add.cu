#include <stdio.h>
#include "timer.h"

#define NUM_THREADS 500000
#define ARRAY_SIZE  100

//Numero de threads en un bloque
#define BLOCK_WIDTH 500

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment(int *g)
{
	// which thread is this?
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	//each thread to increment consecutive elementes, wrappint at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	//g[i] = g[i] + 1;
	atomicAdd(&g[i], 1);
}


int main(int argc,char **argv)
{   
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
 
    // declare, allocate, and zero out GPU memory
	int * d_array;
	cudaMalloc((void**)&d_array,ARRAY_BYTES);
	cudaMemset((void*)d_array, 0, ARRAY_BYTES);
	//cudaMemcpy(d_array, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);

    timer.Start();

    // launch the kernel 
	increment <<<NUM_THREADS/BLOCK_WIDTH,BLOCK_WIDTH>>>(d_array);

    timer.Stop();
    
    // copy back the array of sums from GPU and print
	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed = %g ms\n", timer.Elapsed());
 
    // free GPU memory allocation and exit
	cudaFree(d_array);

    return 0;
}

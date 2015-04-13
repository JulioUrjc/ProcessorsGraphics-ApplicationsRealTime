#include <stdio.h>

#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 16
#define BLOCK_WIDTH 16 // num threads by block

__global__ void kernel(char * m){

	printf("%s\n",m);
   return;
}

int main (int argc, char ** argv) {


   int tamM = 8;
   char * m  = (char *) malloc(tamM);
   char * dm = NULL;

   _snprintf_s(m,8,8, "Hello!!\0");

   fprintf(stderr,"Al hacer la reserva: %s \n",cudaGetErrorString(cudaMalloc(&dm,tamM)));
   fprintf(stderr,"Al hacer la copia:   %s \n",cudaGetErrorString(cudaMemcpy(dm,m,tamM,cudaMemcpyHostToDevice)));
   fprintf(stderr,"Soy el HOST voy a mandar a dispositivo: %s\n",m);

   //Launch kernel
   kernel<<<NUM_THREADS/BLOCK_WIDTH,BLOCK_WIDTH>>>(dm);

   fprintf(stderr,"Mensaje: %s\n",m);

   cudaFree(dm);
   
   return 0;
}


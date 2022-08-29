#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string>

__global__ void myKernel(int *a){
    uint thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block[%d], thread[%d]: a[%d]=%d\n", blockIdx.x, threadIdx.x, thread_global_idx, a[thread_global_idx]);
}

int main(int argc, char* argv[]){
    int a[4] = {0,1,2,3};
    int *dev_a;
    uint size = 4*sizeof(int);

    cudaMalloc((void**)&dev_a, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    uint b = 2; // dim3 b(2,1,1);
    uint t = 2; //dim3 t(2,1,1);
    myKernel<<<b,t>>>(dev_a);
    cudaDeviceSynchronize();
    cudaFree(dev_a);
    return 0;
}

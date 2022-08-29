#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string>

__global__ void myKernel(int *a){
    uint thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    printf("thread[%d, %d]: a[%d]=%d\n", threadIdx.x, threadIdx.y, thread_idx, a[thread_idx]);
}

int main(int argc, char* argv[]){
    int a[4] = {0,1,2,3};
    int *dev_a;
    uint size = 4*sizeof(int);

    cudaMalloc((void**)&dev_a, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

    uint b = 1; // dim3 b(1,1,1);
    dim3 t(2, 2); //dim3 t(2,2,1);
    myKernel<<<b,t>>>(dev_a);
    cudaDeviceSynchronize();
    cudaFree(dev_a);
    return 0;
}

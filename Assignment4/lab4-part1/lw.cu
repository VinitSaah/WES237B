#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string>

__global__ void myKernel(int *m, int *v, int *r){
    // write your code here
    uint thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("block[%d], thread[%d]: m[%d]=%d\n", blockIdx.x, threadIdx.x, thread_global_idx, m[thread_global_idx]);
    r[thread_global_idx] = 0;
    for(int i =0; i < 3;i++)
    {
        r[thread_global_idx] += m[(thread_global_idx*3)+i]*v[i];
    }    
}

int main(int argc, char* argv[]){

    int m[9] = {0,1,2,
		2,3,4,
		4,5,6};

    int v[3] = {-1,0,1};
    int r[3];
    
    int *dev_m;
    int *dev_v;
    int *dev_r;

    uint size_m = 9*sizeof(int);
    uint size_v = 3*sizeof(int);

    cudaMalloc((void**)&dev_m, size_m);
    cudaMalloc((void**)&dev_v, size_v);
    cudaMalloc((void**)&dev_r, size_v);

    cudaMemcpy(dev_m, m, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, v, size_v, cudaMemcpyHostToDevice);

    uint b = 1; // dim3 b(1,1,1);
    uint t = 3; //dim3 t(3,1,1);
    myKernel<<<b,t>>>(dev_m, dev_v, dev_r);
    cudaDeviceSynchronize();
    
    cudaMemcpy(r, dev_r, size_v, cudaMemcpyDeviceToHost);
    int scs = 1;
    for(uint i=0; i<3; i++){
        if(r[i] != 2){
            printf("error! ");
            scs = 0;
        }
        printf("r[%d] = %d\n", i, r[i]);
    }
    if(scs == 1){
        printf("Done with no error\n");
    }
    cudaFree(dev_m);
    cudaFree(dev_v);
    cudaFree(dev_r);
    return 0;
}

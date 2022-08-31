#include <stdio.h>
#include <stdlib.h>

#include "matrixmul.h"
#include "timer.h"

#define BLOCK_SIZE 16

__global__ void block_mm_kernel(const float* A, const float* B, float* output, int M, int N) 
{
	// TODO: complete the block matrix kernel function
	const unsigned int tidx =  threadIdx.x;
	const unsigned int tidy =  threadIdx.y;
	const unsigned int bidx =  blockIdx.x;
	const unsigned int bidy =  blockIdx.y;

	uint64_t col = bidx*blockDim.x +tidx;
    uint64_t row = bidy*blockDim.y +tidy;

	__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    if(col < N && row < N)
	{
		float reg = 0;
		for(int k = 0; k < M/BLOCK_SIZE; k++)
		{
			a_shared[tidy][tidx] = A[(row*M)+(BLOCK_SIZE*k+tidx)];
			b_shared[tidy][tidx] = B[col+((k*BLOCK_SIZE)+tidy)*N];
			__syncthreads();
			for(int i = 0; i < BLOCK_SIZE; i++)
			{
				reg += a_shared[tidy][i]*b_shared[i][tidx];
			}
			__syncthreads();
		}
		output[row*N+col] = reg;
	}
}



inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}


float run_mm_gpu(const float* A, const float* B, float* C, int M, int N)
{
	Timer gpu_timer;
	gpu_timer.start();

	int block_size = BLOCK_SIZE;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(divup(N, dimBlock.x), divup(N, dimBlock.y));

	//TODO: launch the kernel function
	block_mm_kernel<<<dimGrid, dimBlock>>>(A, B, C, M, N);
	
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}



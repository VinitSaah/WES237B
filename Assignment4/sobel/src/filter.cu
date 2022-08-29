#include "filter.h"
#include "timer.h"

#include <iostream>

using namespace std;

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	const int sobel_kernel_x[3][3] = 
	{
    	{ 1,  0, -1},
    	{ 2,  0, -2},
    	{ 1,  0, -1}
	};

    const int sobel_kernel_y[3][3] = 
	{
    	{ 1,  2, 1},
    	{ 0,  0, 0},
    	{ -1, -2,-1}
	};

    //Gx
    for(int i = 0; i < height-2; i++)
    {
        for(int j = 0; j < width-2; j++)
        {
            int32_t reg1 = 0;
            int32_t reg2 = 0;
            reg1 += input[(i+0)*width+(j+0)] * sobel_kernel_x[0][0];
            reg1 += input[(i+0)*width+(j+1)] * sobel_kernel_x[0][1];
            reg1 += input[(i+0)*width+(j+2)] * sobel_kernel_x[0][2];
            reg1 += input[(i+1)*width+(j+0)] * sobel_kernel_x[1][0];
            reg1 += input[(i+1)*width+(j+1)] * sobel_kernel_x[1][1];
            reg1 += input[(i+1)*width+(j+2)] * sobel_kernel_x[1][2];
            reg1 += input[(i+2)*width+(j+0)] * sobel_kernel_x[2][0];
            reg1 += input[(i+2)*width+(j+1)] * sobel_kernel_x[2][1];
            reg1 += input[(i+2)*width+(j+2)] * sobel_kernel_x[2][2];
        
            reg2 += input[(i+0)*width+(j+0)] * sobel_kernel_y[0][0];
            reg2 += input[(i+0)*width+(j+1)] * sobel_kernel_y[0][1];
            reg2 += input[(i+0)*width+(j+2)] * sobel_kernel_y[0][2];
            reg2 += input[(i+1)*width+(j+0)] * sobel_kernel_y[1][0];
            reg2 += input[(i+1)*width+(j+1)] * sobel_kernel_y[1][1];
            reg2 += input[(i+1)*width+(j+2)] * sobel_kernel_y[1][2];
            reg2 += input[(i+2)*width+(j+0)] * sobel_kernel_y[2][0];
            reg2 += input[(i+2)*width+(j+1)] * sobel_kernel_y[2][1];
            reg2 += input[(i+2)*width+(j+2)] * sobel_kernel_y[2][2];

            double gxgy= sqrt(reg1*reg1+reg2*reg2);
            if(gxgy > 255)
            {
                gxgy = 255; //max value of uint8_t It corresponds to White
            }
            output[i*width+j] = (uint8_t)gxgy;  
        }
    }
}

// =================== GPU Kernel Functions ===================

__global__
void sobel_filter_kernel(const uchar * input, uchar * output, const uint height, const uint width)
{
    uint64_t col = blockIdx.x*blockDim.x +threadIdx.x;
    uint64_t row = blockIdx.y*blockDim.y +threadIdx.y;
    uint64_t stride_x = blockDim.x*gridDim.x;
    uint64_t stride_y = blockDim.y*gridDim.y;

    const int sobel_kernel_x[3][3] = 
	{
    	{ 1,  0, -1},
    	{ 2,  0, -2},
    	{ 1,  0, -1}
	};

    const int sobel_kernel_y[3][3] = 
	{
    	{ 1,  2, 1},
    	{ 0,  0, 0},
    	{ -1, -2,-1}
	};
    
    while(col < (width-2) && row < (height-2))
    {
        int32_t reg1 = 0;
        int32_t reg2 = 0;

        reg1 += input[(row+0)*width+(col+0)] * sobel_kernel_x[0][0];
        reg1 += input[(row+0)*width+(col+1)] * sobel_kernel_x[0][1];
        reg1 += input[(row+0)*width+(col+2)] * sobel_kernel_x[0][2];
        reg1 += input[(row+1)*width+(col+0)] * sobel_kernel_x[1][0];
        reg1 += input[(row+1)*width+(col+1)] * sobel_kernel_x[1][1];
        reg1 += input[(row+1)*width+(col+2)] * sobel_kernel_x[1][2];
        reg1 += input[(row+2)*width+(col+0)] * sobel_kernel_x[2][0];
        reg1 += input[(row+2)*width+(col+1)] * sobel_kernel_x[2][1];
        reg1 += input[(row+2)*width+(col+2)] * sobel_kernel_x[2][2];
        
        reg2 += input[(row+0)*width+(col+0)] * sobel_kernel_y[0][0];
        reg2 += input[(row+0)*width+(col+1)] * sobel_kernel_y[0][1];
        reg2 += input[(row+0)*width+(col+2)] * sobel_kernel_y[0][2];
        reg2 += input[(row+1)*width+(col+0)] * sobel_kernel_y[1][0];
        reg2 += input[(row+1)*width+(col+1)] * sobel_kernel_y[1][1];
        reg2 += input[(row+1)*width+(col+2)] * sobel_kernel_y[1][2];
        reg2 += input[(row+2)*width+(col+0)] * sobel_kernel_y[2][0];
        reg2 += input[(row+2)*width+(col+1)] * sobel_kernel_y[2][1];
        reg2 += input[(row+2)*width+(col+2)] * sobel_kernel_y[2][2];

        double gxgy= sqrt(float(reg1*reg1+reg2*reg2));
        if(gxgy > 255)
        {
            gxgy = 255; //max value of uint8_t It corresponds to White
        }
        output[row*width+col] = (uint8_t)gxgy;
        col += stride_x;
        row += stride_y; 
    }

}

// =================== GPU Host Functions ===================
void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	//TODO: launch kernel function
	if(NULL != input && NULL != output)
    {
        int block_size = 32;
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid(divup(width, dimBlock.x), divup(height, dimBlock.y));
        sobel_filter_kernel<<<dimGrid, dimBlock>>>(input, output, height, width);
        cudaDeviceSynchronize();
    }
}

#include "img_proc.h"
#include <stdint.h>
#include <iostream>
#define NUM_THREADS 128
#define NUM_BLOCKS 64
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================

void img_rgb2gray_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    //TODO: Convert a 3 channel RGB image to grayscale
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint img_idx = (i*width)+j;
            uint rgb_i = img_idx*channels;
            int sum = 0;
            for(int chan = 0; chan< channels; chan++)
            {
                sum += in[rgb_i+chan];
            }
            out[img_idx] = sum/channels;
        }
    }

}

void img_invert_cpu(uchar* out, const uchar* in, const uint width, const uint height)
{
    //TODO: Invert a 8bit image
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint img_idx = (i*width)+j;
            out[img_idx] = 255-in[img_idx];
        }
    }
}

void img_blur_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int blur_size)
{
    //TODO: Average out blur_size pixels
    //create the filter kernel
    int num_kernel = blur_size*blur_size;
    size_t kernel_size = sizeof(int)*num_kernel;
    int* blur_kernel = (int*)malloc(kernel_size);
    for(int i = 0; i < blur_size; i++)
    {
        for(int j = 0; j < blur_size; j++)
        {
            blur_kernel[(i*blur_size)+j] = 1;
        }
    }
    //int* ptr_blur_kernel = blur_kernel.ptr<int>();
    for(int i = 0; i < height-(blur_size-1); i++)
    {
        for(int j = 0; j < width-(blur_size-1); j++)
        {
            float reg1 = 0;
            for(int k = 0; k< blur_size; k++)
            {
                for(int l = 0; l< blur_size; l++)
                {
                    reg1 += in[(i+k)*width+(j+l)] * blur_kernel[(k*blur_size)+l];
                }
                
            }
            out[i*width+j] = (uchar)(reg1/num_kernel);  
        }
    }
    free(blur_kernel);
}

// =================== GPU Kernel Functions ===================
/*
TODO: Write GPU kernel functions for the above functions
   */
__global__
void img_rgb2gray_kernel(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    uint64_t col = blockIdx.x*blockDim.x +threadIdx.x;
    uint64_t row = blockIdx.y*blockDim.y +threadIdx.y;

    if(col < width && row < height)
    {
        uint64_t offset = row*width+col;
        int sum  = 0;
        for (int i  = 0; i < channels; i++)
        {
            sum += in[(offset+i)*channels];
        }
        sum = sum/3;
        out[offset] = sum;
    }
}

// =================== GPU Host Functions ===================
/* 
TODO: Write GPU host functions that launch the kernel functions above
   */

void img_rgb2gray_host(uchar* out, const uchar* in, uchar* gray_ptr, uchar* rgb_ptr, const uint width, const uint height, const int channels)
{
    if(NULL != in && NULL != rgb_ptr && NULL != gray_ptr && NULL != rgb_ptr)
    {
        size_t size_img = width*height;
        cudaMemcpy(rgb_ptr, in, size_img*channels, cudaMemcpyHostToDevice);

        dim3 dimGrid(64, 64,1);
        dim3 dimBlock(divup(width, dimGrid.x), divup(width, dimGrid.y),1);

        img_rgb2gray_kernel<<<dimGrid, dimBlock>>>(gray_ptr, rgb_ptr, width, height, channels);

        cudaDeviceSynchronize();
        cudaMemcpy(out, gray_ptr, size_img, cudaMemcpyDeviceToHost);
    #if 0
        cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":" << __LINE__
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
    #endif
    }

}
__global__
void img_invert_kernel(uchar* out, const uchar* in, const uint width, const uint height)
{
    uint64_t col = blockIdx.x*blockDim.x +threadIdx.x;
    uint64_t row = blockIdx.y*blockDim.y +threadIdx.y;

    if(col < width && row < height)
    {
        uint64_t offset = row*width+col;
        out[offset] = 255-in[offset];
    }
}
void img_invert(uchar* out, const uchar* in, const uint height, const uint width)
{
    if(NULL != in && NULL != out )
    {
        dim3 dimGrid(64, 64,1);
        dim3 dimBlock(divup(width, dimGrid.x), divup(width, dimGrid.y),1);
        img_invert_kernel<<<dimGrid, dimBlock>>>(out, in, width, height);
        cudaDeviceSynchronize();
    }

}

__global__
void img_blur_kernel(uchar* out, const uchar* in, const uint height, const uint width, const int* k_blur, const int blur_size)
{
    uint64_t col = blockIdx.x*blockDim.x +threadIdx.x;
    uint64_t row = blockIdx.y*blockDim.y +threadIdx.y;
    size_t kernel_size = blur_size*blur_size;
     if(col < (width-(blur_size-1)) && row < (height-(blur_size-1)))
     {
        uint64_t offset = row*width+col;
        float reg1 = 0.0;
        for(int k = 0; k< blur_size; k++)
        {
            for(int l = 0; l< blur_size; l++)
            {
                reg1 += (float)in[(row+k)*width+(col+l)] * k_blur[(k*blur_size)+l];
            }   
        }
        out[offset] = (uchar)(reg1/kernel_size);
     }

}

void img_blur(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size)
{
     if(NULL != in && NULL != out )
     {
        int* blur_kernel = NULL;
        int* blur_kernel_k = NULL;
        size_t size_kernel = blur_size*blur_size*sizeof(int);

        cudaMalloc((void**)&blur_kernel_k,size_kernel);

        blur_kernel = (int*)malloc(size_kernel);

        for(int i = 0; i < blur_size; i++)
        {
            for(int j = 0; j < blur_size; j++)
            {
                blur_kernel[(i*blur_size)+j] = 1;
            }
        }
        
        cudaMemcpy(blur_kernel_k, blur_kernel, size_kernel, cudaMemcpyHostToDevice);
        dim3 dimGrid(64, 64,1);
        dim3 dimBlock(divup(width, dimGrid.x), divup(width, dimGrid.y),1);
        img_blur_kernel<<<dimGrid, dimBlock>>>(out, in, height, width, blur_kernel_k, blur_size);
        cudaDeviceSynchronize();
        free(blur_kernel);
     }
}
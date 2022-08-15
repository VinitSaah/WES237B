
#include "sobel.h"
#include "mat.h"
using namespace std;
using namespace cv;

#define LOOP_UNROLL_1
//#define DEBUG_NEON
const int sobel_kernel_x[3][3] = {
    { 1,  0, -1},
    { 2,  0, -2},
    { 1,  0, -1}};

const int sobel_kernel_y[3][3] = {
    { 1,  2, 1},
    { 0,  0, 0},
    { -1, -2,-1}
};



void sobel(const Mat& src, Mat& dst)
{
    const int HEIGHT = src.rows;
    const int WIDTH  = src.cols;

    //Gx//Gy square, sqrt
    for(int i = 0; i < HEIGHT-2; i++)
    {
        for(int j = 0; j < WIDTH-2; j++)
        {
            int32_t reg1 = 0;
            int32_t reg2 = 0;
            for(int k = 0; k< 3; k++)
            {
                for(int l = 0; l< 3; l++)
                {
                    reg1 += src.at<uint8_t>((i+k),(j+l)) * sobel_kernel_x[k][l];
                    reg2 += src.at<uint8_t>((i+k),(j+l)) * sobel_kernel_y[k][l];
                }
                
            }
            uint32_t gxgy= sqrt(reg1*reg1+reg2*reg2);
            if(gxgy > 255)
            {
                gxgy = 255; //max value of uint8_t It corresponds to White
            }
            dst.at<uint8_t>(i,j) = (uint8_t)gxgy;  
        }
    }
}

void sobel_unroll(const Mat& src, Mat& dst)
{
    const int HEIGHT = src.rows;
    const int WIDTH  = src.cols;

    //Gx
    for(int i = 0; i < HEIGHT-2; i++)
    {
        for(int j = 0; j < WIDTH-2; j++)
        {
            int32_t reg1 = 0;
            int32_t reg2 = 0;
#ifdef LOOP_UNROLL_1
            for(int k = 0; k< 3; k++)
            {
                reg1 += src.at<uint8_t>((i+k),(j))   * sobel_kernel_x[k][0];
                reg1 += src.at<uint8_t>((i+k),(j+1)) * sobel_kernel_x[k][1];
                reg1 += src.at<uint8_t>((i+k),(j+2)) * sobel_kernel_x[k][2];

                reg2 += src.at<uint8_t>((i+k),(j))   * sobel_kernel_y[k][0];
                reg2 += src.at<uint8_t>((i+k),(j+1)) * sobel_kernel_y[k][1];
                reg2 += src.at<uint8_t>((i+k),(j+2)) * sobel_kernel_y[k][2];
            }
#else

                reg1 += src.at<uint8_t>((i+0),(j))   * sobel_kernel_x[0][0];
                reg1 += src.at<uint8_t>((i+0),(j+1)) * sobel_kernel_x[0][1];
                reg1 += src.at<uint8_t>((i+0),(j+2)) * sobel_kernel_x[0][2];
                reg1 += src.at<uint8_t>((i+1),(j))   * sobel_kernel_x[1][0];
                reg1 += src.at<uint8_t>((i+1),(j+1)) * sobel_kernel_x[1][1];
                reg1 += src.at<uint8_t>((i+1),(j+2)) * sobel_kernel_x[1][2];
                reg1 += src.at<uint8_t>((i+2),(j))   * sobel_kernel_x[2][0];
                reg1 += src.at<uint8_t>((i+2),(j+1)) * sobel_kernel_x[2][1];
                reg1 += src.at<uint8_t>((i+2),(j+2)) * sobel_kernel_x[2][2];
        
                reg2 += src.at<uint8_t>((i+0),(j))   * sobel_kernel_y[0][0];
                reg2 += src.at<uint8_t>((i+0),(j+1)) * sobel_kernel_y[0][1];
                reg2 += src.at<uint8_t>((i+0),(j+2)) * sobel_kernel_y[0][2];
                reg2 += src.at<uint8_t>((i+1),(j))   * sobel_kernel_y[1][0];
                reg2 += src.at<uint8_t>((i+1),(j+1)) * sobel_kernel_y[1][1];
                reg2 += src.at<uint8_t>((i+1),(j+2)) * sobel_kernel_y[1][2];
                reg2 += src.at<uint8_t>((i+2),(j))   * sobel_kernel_y[2][0];
                reg2 += src.at<uint8_t>((i+2),(j+1)) * sobel_kernel_y[2][1];
                reg2 += src.at<uint8_t>((i+2),(j+2)) * sobel_kernel_y[2][2];

#endif

            uint64_t gxgy= sqrt(reg1*reg1+reg2*reg2);
            if(gxgy > 255)
            {
                gxgy = 255; //max value of uint8_t It corresponds to White
            }
            dst.at<uint8_t>(i,j) = (uint8_t)gxgy;  
        }
    }
}

void sobel_neon(const Mat& src, Mat& dst)
{
    const int HEIGHT = src.rows;
    const int WIDTH  = src.cols;
    
    float32x4_t ne_gx;
    float32x4_t ne_gy;
    float32x4_t input_reg;
    float32x4_t kernel_reg_x;
    float32x4_t kernel_reg_y;

    uint32_t KERNEL_SIZE = 3*3;
    //Converting into float32 type as it is big enough to store int and uint8_t and avoids recast during sqrt
    float32_t ne_kernel_sobel_x[KERNEL_SIZE] = {0};
    float32_t ne_kernel_sobel_y[KERNEL_SIZE] = {0};

    float32_t input_chunk[KERNEL_SIZE] = {0};
    
    //Fill sobel filter kernel into 1D memory;
    for(uint8_t i = 0; i< 3; i++)
    {
        for(uint8_t j=0; j<3; j++)
        {
            ne_kernel_sobel_x[i*3+j] = (float32_t)sobel_kernel_x[i][j];
            ne_kernel_sobel_y[i*3+j] = (float32_t)sobel_kernel_y[i][j];
        }
    }
#ifdef DEBUG_NEON
    std::cout << "Linearized Sobel Filter X" << std::endl;
    for(uint8_t i = 0; i< 3; i++)
    {
        for(uint8_t j = 0; j< 3; j++)
        {
            std::cout << ne_kernel_sobel_x[i*3+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Linearized Sobel Filter Y" << std::endl;
    for(uint8_t i = 0; i< 3; i++)
    {
        for(uint8_t j = 0; j< 3; j++)
        {
            std::cout << ne_kernel_sobel_y[i*3+j] << " ";
        }
        std::cout << std::endl;
    }
#endif
    uint32_t BLOCK_SIZE = KERNEL_SIZE/4;
    uint8_t REM_SIZE = KERNEL_SIZE%4;
    for(int i = 0; i < HEIGHT-2; i++)
    {
        for(int j = 0; j < WIDTH-2; j++)
        {
            float32_t gx = 0;
            float32_t gy = 0;
            //fill kernal size data 3*3 size worth into float32 type arr;
            for(int k = 0; k < 3; k++)
            {
                for(int l = 0; l< 3; l++)
                {
                    input_chunk[k*3+l] = (float32_t)src.at<uint8_t>((i+k),(j+l));
                }
            }
            //Now we have float32 type data use neon intrinsics in block of 4 (float32x4), process remaining in other loop
            for(uint32_t block_idx = 0; block_idx< BLOCK_SIZE; block_idx++)
            {
                input_reg = vld1q_f32(&input_chunk[block_idx]+4*block_idx);
                kernel_reg_x = vld1q_f32(&ne_kernel_sobel_x[block_idx]+4*block_idx);
                kernel_reg_y = vld1q_f32(&ne_kernel_sobel_y[block_idx]+4*block_idx);
                ne_gx = vmulq_f32(input_reg, kernel_reg_x);
                ne_gy = vmulq_f32(input_reg, kernel_reg_y);
                float32_t ne_gx_arr[4] = {0};
                float32_t ne_gy_arr[4] = {0};
                vst1q_f32(&ne_gx_arr[0], ne_gx);
                vst1q_f32(&ne_gy_arr[0], ne_gy);

                gx += ne_gx_arr[0]+ne_gx_arr[1]+ne_gx_arr[2]+ne_gx_arr[3];
                gy += ne_gy_arr[0]+ne_gy_arr[1]+ne_gy_arr[2]+ne_gy_arr[3]; 
            }

            //Process the remaining
            for(uint8_t rem = 0; rem< REM_SIZE; rem++)
            {
                gx +=  input_chunk[(4*BLOCK_SIZE)+rem]*ne_kernel_sobel_x[(4*BLOCK_SIZE)+rem];
                gy +=  input_chunk[(4*BLOCK_SIZE)+rem]*ne_kernel_sobel_y[(4*BLOCK_SIZE)+rem];
            }
            uint64_t gxgy= sqrt(int32_t(gx*gx+gy*gy));
            if(gxgy > 255)
            {
                gxgy = 255;
            }
            dst.at<uint8_t>(i,j) = (uint8_t)gxgy;  
        }
    }
}
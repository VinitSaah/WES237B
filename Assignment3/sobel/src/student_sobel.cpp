
#include "sobel.h"
#include "mat.h"
using namespace std;
using namespace cv;


int sobel_kernel_x[3][3] = {
	{ -1,  0, 1},
	{ -2,  0, 2},
	{ -1,  0, 1}};

int sobel_kernel_y[3][3] = {
	{ -1,  -2, -1},
	{ 0,  0, 0},
	{ 1, 2, 1}
};



void sobel(const Mat& src, Mat& dst)
{
	// TODO
	const int HEIGHT = src.rows;
	const int WIDTH  = src.cols;

	const uint8_t* src_ptr  = src.ptr<uint8_t>();
    uint8_t* result_ptr = dst.ptr<uint8_t>();

    //Gx
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
// TODO
	const int HEIGHT = src.rows;
	const int WIDTH  = src.cols;

	const uint8_t* src_ptr  = src.ptr<uint8_t>();
    uint8_t* result_ptr = dst.ptr<uint8_t>();

    //Gx
	for(int i = 0; i < HEIGHT-2; i++)
	{
		int64_t reg3 = 0;
		int64_t reg4 = 0;
		for(int j = 0; j < WIDTH-2; j++)
		{
			int32_t reg1[3] = {0};
			int32_t reg2[3] = {0};
            for(int k = 0; k< 3; k++)
			{
				reg1[0] += src.at<uint8_t>((i+k),(j))   * sobel_kernel_x[k][0];
				reg1[1] += src.at<uint8_t>((i+k),(j+1)) * sobel_kernel_x[k][1];
				reg1[2] += src.at<uint8_t>((i+k),(j+2)) * sobel_kernel_x[k][2];

				reg2[0] += src.at<uint8_t>((i+k),(j))   * sobel_kernel_y[k][0];
				reg2[1] += src.at<uint8_t>((i+k),(j+1)) * sobel_kernel_y[k][1];
				reg2[2] += src.at<uint8_t>((i+k),(j+2)) * sobel_kernel_y[k][2];
			}
			reg3 = reg1[0]+reg1[1]+reg1[2];
			reg4 = reg2[0]+reg2[1]+reg2[2];
			uint64_t gxgy= sqrt(reg3*reg3+reg4*reg4);
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
	// TODO
}


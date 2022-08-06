#include "main.h"
#include "mat.h"

using namespace cv;

#define USE_LUT //option to enable look up table

/* options to enable specific algo dct */
#define USE_NAIVE_DCT 
//#define USE_1D_SEPARABLE
//#define USE_NAIVE_MM
//#define USE_BMM

Mat LUT_w;
Mat LUT_h;

// Helper function
float sf(int in){
	if (in == 0)
		return 0.70710678118; // = 1 / sqrt(2)
	return 1.;
}

// Initialize LUT
void initDCT(int WIDTH, int HEIGHT)
{
	if(WIDTH != HEIGHT)
	{
		std::cerr<<"DCT only supported for square matrix" << std::endl;
		return;
	}
	
	LUT_h = Mat(HEIGHT, WIDTH, CV_32FC1);
	LUT_w = Mat(HEIGHT, WIDTH, CV_32FC1);

	float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_w_ptr  = LUT_w.ptr<float>();
    
	for(int x = 0; x < HEIGHT; x++)
	{
		for(int y = 0; y < WIDTH; y++)
		{
			LUT_h_ptr[x * WIDTH + y] = sf(x)*sqrt(2./HEIGHT)*cos(M_PI/((float)HEIGHT)*(y+1./2.)*(float)x);
            LUT_w_ptr[x * WIDTH + y] = LUT_h_ptr[x * WIDTH + y]; //both the LUT table is same.
		}
	}
}

#ifdef USE_NAIVE_DCT// Baseline: O(N^4)
Mat student_dct(Mat input)
{
	const int HEIGHT = input.rows;
	const int WIDTH  = input.cols;

	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

	// Note: Using pointers is faster than Mat.at<float>(x,y)
	// Try to use pointers for your LUT as well
	float* result_ptr = result.ptr<float>();
	float* input_ptr  = input.ptr<float>();
#ifdef USE_LUT
	float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_w_ptr  = LUT_w.ptr<float>();
	std::cout << "LUT is Enabled" << std::endl;
#else
    float scale = 2./sqrt(HEIGHT*WIDTH);
#endif
	for(int x = 0; x < HEIGHT; x++)
	{
		for(int y = 0; y < WIDTH; y++)
		{
			float value = 0.f;

			for(int i = 0; i < HEIGHT; i++)
			{
				for(int j = 0; j < WIDTH; j++)
				{
#ifdef USE_LUT
					value += input_ptr[i * WIDTH + j]
						* LUT_h_ptr[x * WIDTH + i]
						* LUT_w_ptr[y * WIDTH + j];
#else
                    value += cos(M_PI/((float)HEIGHT)*(i+1./2.)*(float)x)
					* cos(M_PI/((float)WIDTH)*(j+1./2.)*(float)y)*input_ptr[i * WIDTH + j];
#endif
				}
			}
#ifndef USE_LUT
            value = scale * sf(x) * sf(y) * value;
#endif
			result_ptr[x * WIDTH + y] = value;
		}
	}
	return result;
}
#endif

#ifdef USE_1D_SEPARABLE
Mat student_dct(Mat input)
{
	const int HEIGHT = input.rows;
	const int WIDTH = input.cols;

#ifndef USE_LUT 
	float scale = 2./sqrt(HEIGHT*WIDTH);
#else
    float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_w_ptr  = LUT_w.ptr<float>();
	std::cout << "LUT is Enabled" << std::endl;
#endif

	// Create the result matrix of the correct datatype
	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);
	Mat result_row = Mat(HEIGHT, WIDTH, CV_32FC1);

	float* result_ptr = result.ptr<float>();
	float* input_ptr = input.ptr<float>();

	// Less naive implementation.
	// Perform 2 1D DCTs, one for the rows and one for the columns
	float value;
	for(int k=0; k<HEIGHT; k++) 
	{
		for(int i=0; i<WIDTH; i++) 
		{
			value = 0.0;
			for(int j=0; j<WIDTH; j++) 
			{
#ifndef USE_LUT
				value += input.at<float>(k, j) * cos(M_PI/((float)WIDTH)*(j+1./2.)*(float)i);
#else
                value += input.at<float>(k, j) * LUT_h_ptr[i * WIDTH + j];
#endif
			}
#ifndef USE_LUT
			result_row.at<float>(k,i) = value * sf(i);
#else
            result_row.at<float>(k,i) = value;
#endif
		}
	}

	// Now perform the column transformation
	for(int k=0; k<WIDTH; k++) 
	{
		for(int i=0; i<HEIGHT; i++) 
		{
			value = 0.0;
			for (int j=0; j<HEIGHT; j++) 
			{
#ifndef USE_LUT
				value += result_row.at<float>(j,k) * cos(M_PI/((float)HEIGHT)*(j+1./2.)*(float)i);
#else
                value += result_row.at<float>(j,k) *  LUT_w_ptr[j * WIDTH + i];
#endif
			}
#ifndef USE_LUT
			result.at<float>(i, k) = value*sf(k)*scale;
#else
            result.at<float>(i, k) = value;
#endif
		}
	}

	return result;
}
#endif

// DCT as matrix multiplication
#ifdef USE_NAIVE_MM
Mat student_dct(Mat input)
{
	// -- Works only for WIDTH == HEIGHT
	assert(input.rows == input.cols);
	const int HEIGHT = input.rows;
	const int WIDTH  = input.cols;
	Mat LUT_h_t = LUT_h.t();
	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

	Matrix<float> mat_input(HEIGHT, WIDTH);
	Matrix<float> temp_mat(HEIGHT, WIDTH);
	Matrix<float> LUT_h_mat(HEIGHT, WIDTH);
	Matrix<float> result_mat(HEIGHT, WIDTH);
	Matrix<float> LUT_transpose(WIDTH,HEIGHT);

	float* input_ptr  = input.ptr<float>();
	float* result_ptr = result.ptr<float>();
    float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_h_ptr_t = LUT_h_t.ptr<float>();

	//copy the mat, would add func to directly do so.
	for(int i = 0; i < HEIGHT; i++)
	{
		for(int j = 0; j< WIDTH; j++)
		{
			mat_input.put(i,j, input_ptr[i*WIDTH+j]);
			LUT_h_mat.put(i,j, LUT_h_ptr[i*WIDTH+j]);
			LUT_transpose.put(i,j, LUT_h_ptr_t[i*WIDTH+j]);
		}
	}
	// -- Matrix multiply with OpenCV
	temp_mat = LUT_h_mat*mat_input;
	result_mat = temp_mat*LUT_transpose;
	
	for(int i = 0; i < HEIGHT; i++)
	{
		for(int j = 0; j< WIDTH; j++)
		{
			result_ptr[i*WIDTH+j] = result_mat.get(i,j);
		}
	}
	return result;
}
#endif

#ifdef USE_BMM
// DCT as block multiplication
Mat student_dct(Mat input)
{
    assert(input.rows == input.cols);
	const int HEIGHT = input.rows;
	const int WIDTH  = input.cols;
	Mat LUT_h_t = LUT_h.t();
	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

	Matrix<float> mat_input(HEIGHT, WIDTH);
	Matrix<float> temp_mat(HEIGHT, WIDTH);
	Matrix<float> LUT_h_mat(HEIGHT, WIDTH);
	Matrix<float> result_mat(HEIGHT, WIDTH);
	Matrix<float> LUT_transpose(WIDTH,HEIGHT);

	float* input_ptr  = input.ptr<float>();
	float* result_ptr = result.ptr<float>();
    float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_h_ptr_t = LUT_h_t.ptr<float>();
    
    std::cout<<"Block Matrix Multiplication Method" << std::endl;
	//copy the mat, would add func to directly do so.
	for(int i = 0; i < HEIGHT; i++)
	{
		for(int j = 0; j< WIDTH; j++)
		{
			mat_input.put(i,j, input_ptr[i*WIDTH+j]);
			LUT_h_mat.put(i,j, LUT_h_ptr[i*WIDTH+j]);
			LUT_transpose.put(i,j, LUT_h_ptr_t[i*WIDTH+j]);
		}
	}
	uint16_t block_size = 64; 
	temp_mat = block_matrix_multiply(LUT_h_mat, mat_input, block_size);
	result_mat = block_matrix_multiply(temp_mat, LUT_transpose, block_size);
	
	for(int i = 0; i < HEIGHT; i++)
	{
		for(int j = 0; j< WIDTH; j++)
		{
			result_ptr[i*WIDTH+j] = result_mat.get(i,j);
		}
	}
	return result;
}

#endif
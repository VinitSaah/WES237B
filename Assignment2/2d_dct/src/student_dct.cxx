#include "main.h"
#include "mat.h"

using namespace cv;


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

#if 0// Baseline: O(N^4)
Mat student_dct(Mat input)
{
	const int HEIGHT = input.rows;
	const int WIDTH  = input.cols;

	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

	// Note: Using pointers is faster than Mat.at<float>(x,y)
	// Try to use pointers for your LUT as well
	float* result_ptr = result.ptr<float>();
	float* input_ptr  = input.ptr<float>();
	float* LUT_h_ptr = LUT_h.ptr<float>();
	float* LUT_w_ptr  = LUT_w.ptr<float>();
	for(int x = 0; x < HEIGHT; x++)
	{
		for(int y = 0; y < WIDTH; y++)
		{
			float value = 0.f;

			for(int i = 0; i < HEIGHT; i++)
			{
				for(int j = 0; j < WIDTH; j++)
				{
					value += input_ptr[i * WIDTH + j]
						* LUT_h_ptr[x * WIDTH + i]
						* LUT_w_ptr[y * WIDTH + j];
				}
			}
			result_ptr[x * WIDTH + y] = value;
		}
	}

	return result;
}
#endif

// *****************
//   Hint
// *****************
//
// DCT as matrix multiplication
#if 0
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
	//Mat output = LUT_w * input * LUT_w.t();
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
	uint16_t block_size = 4; 
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
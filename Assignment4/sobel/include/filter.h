#ifndef _FILTER_H_
#define _FILTER_H_

#include <cuda_runtime.h>
#include <string>

typedef unsigned char uchar;
typedef unsigned int uint;

void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width);
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width);

template<class T>
double compute_rmse(const T * data1, const T * data2, size_t size)
{
	double mse = 0.0;
    for (uint i = 0; i < size; ++i)
	{
        double diff = abs((double)data1[i] - (double)data2[i]);
        mse += diff * diff;
    }

    return sqrt(mse);
}

inline int get_array_index(const int x, const int y, const int width) {
	return x + y * width;
}

inline std::string get_cuda_error() {
	return cudaGetErrorString(cudaGetLastError());
}

#endif // _FILTER_H_


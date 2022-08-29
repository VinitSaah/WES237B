#ifndef _FILTER_H_
#define _FILTER_H_

#include "cuda_runtime.h"

typedef unsigned char uchar;
typedef unsigned int uint;

// =================== CPU Functions ===================
void img_invert_cpu(uchar* out, const uchar* in, const uint height, const uint width);
void img_rgb2gray_cpu(uchar* out, const uchar* in, const uint height, const uint width, const int channels);
void img_blur_cpu(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size);
void img_rgb2gray_host(uchar* out, const uchar* in, uchar* gray_ptr, uchar* rgb_ptr, const uint width, const uint height, const int channels);

// =================== GPU Host Functions ===================
void img_rgb2gray(uchar* out, const uchar* in, const uint height, const uint width, const int channels);
void img_invert(uchar* out, const uchar* in, const uint height, const uint width);
void img_blur(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size);

#endif

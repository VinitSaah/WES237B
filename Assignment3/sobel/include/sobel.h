#ifndef __SOBEL_H_
#define __SOBEL_H_

#include <opencv2/imgproc/imgproc.hpp>


void sobel(const cv::Mat& src, cv::Mat& dst);
void sobel_unroll(const cv::Mat& src, cv::Mat& dst);
void sobel_neon(const cv::Mat& src, cv::Mat& dst);


#endif // __SOBEL_H_

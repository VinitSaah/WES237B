#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "img_proc.h"
#include "timer.h"

#define OPENCV 0
#define CPU 1
#define GPU 2

#define BLUR_SIZE 5

//#define UNIFIED_MEM 

using namespace std;
using namespace cv;

#define GRAY
#define INVERT
#define BLUR

int usage()
{
	cout << "Usage: ./lab4 <mode> <WIDTH> <HEIGHT>" <<endl;
	cout << "mode: 0 OpenCV" << endl;
	cout << "      1 CPU" << endl;
	cout << "      2 GPU" << endl;
	return 0;
}

int use_mode(int mode)
{
	string descr;
	switch(mode)
	{
		case OPENCV:
			descr = "OpenCV Functions";
			break;
		case CPU:
			descr = "CPU Implementations";
			break;
		case GPU:
			descr = "GPU Implementations";
			break;
		default:
			descr = "None";
			return usage();
	}	
	
	cout << "Using " << descr.c_str() <<endl;
	return 1;
}

int main(int argc, const char *argv[]) 
{

	int mode = 0;

	if(argc >= 2)
	{
		mode = atoi(argv[1]);	
	}
	
	if(use_mode(mode) == 0)
		return 0;

	VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;
	int CHANNELS = 3;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 3)
	{
		WIDTH = atoi(argv[2]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 4)
	{
		HEIGHT = atoi(argv[3]);
	}

	// Profiling framerate
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;
    
	uchar* rgb_ptr = NULL;
	uchar* gray_ptr = NULL;
	uchar* invert_ptr = NULL;
	uchar* blur_ptr = NULL;
	size_t size_img = WIDTH*HEIGHT*CHANNELS*sizeof(uchar); 
#ifndef UNIFIED_MEM
    //TODO: Allocate memory on the GPU device.
    cudaMalloc((void**)&rgb_ptr, size_img);
	//TODO: Declare the host image result matrices
	cudaMalloc((void**)&gray_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
	cudaMalloc((void**)&invert_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
	cudaMalloc((void**)&blur_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#else
    //TODO: Allocate unified memory for the necessary matrices
	cudaMallocManaged(&gray_ptr, WIDTH*HEIGHT*sizeof(uchar));
	cudaMallocManaged(&rgb_ptr, size_img);

    //TODO: Declare the image matrices which point to the unified memory
#endif
#ifndef UNIFIED_MEM
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U);
	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3);
#else
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3,rgb_ptr);
#endif

	//Matrix for OpenCV inversion
	Mat ones = Mat::ones(HEIGHT, WIDTH, CV_8U)*255;
	Mat invert = Mat::zeros(HEIGHT, WIDTH, CV_8U);
	Mat blr = Mat(HEIGHT,WIDTH,CV_8U);
	Mat frame;	
	char key=0;
	int count = 0;

	while (key != 'q')
	{
		cap >> frame;
		if(frame.empty())
		{
			waitKey();
			break;
		}

		resize(frame, rgb, Size(WIDTH, HEIGHT));

		imshow("Original", rgb);

		timer.start();
		switch(mode)
		{
			case OPENCV:
#ifdef OPENCV4
				cvtColor(rgb, gray, COLOR_BGR2GRAY);
#else
				cvtColor(rgb, gray, CV_BGR2GRAY);
				
#endif
				//cv::subtract(ones,rgb,invert);
				invert = ones-gray;
				blur(gray,blr,Size(BLUR_SIZE,BLUR_SIZE));
				break;
			case CPU:
                // TODO: 1) Call the CPU functions
				img_rgb2gray_cpu(gray.ptr<uchar>(),rgb.ptr<uchar>(),HEIGHT, WIDTH, CHANNELS);
				img_invert_cpu(invert.ptr<uchar>(),gray.ptr<uchar>(),HEIGHT, WIDTH);
				img_blur_cpu(blr.ptr<uchar>(), gray.ptr<uchar>(),HEIGHT, WIDTH, BLUR_SIZE);
				break;

			case GPU:
#ifndef UNIFIED_MEM
                /* TODO: 1) Copy data from host to device
                 *       2) Call GPU host function with device data
                 *       3) Copy data from device to host
                */
			   //Memcpy input matrix
			   img_rgb2gray_host(gray.ptr<uchar>(), rgb.ptr<uchar>(), gray_ptr, rgb_ptr, HEIGHT, WIDTH, CHANNELS);
			   img_invert(invert_ptr, gray_ptr, HEIGHT, WIDTH);
			   cudaMemcpy(invert.ptr<uchar>(), invert_ptr, HEIGHT*WIDTH, cudaMemcpyDeviceToHost);
			   img_blur(blur_ptr, gray_ptr, HEIGHT, WIDTH, BLUR_SIZE);
			   cudaMemcpy(blr.ptr<uchar>(), blur_ptr, HEIGHT*WIDTH, cudaMemcpyDeviceToHost);

#else
                /* TODO: 1) Call GPU host function with unified memory allocated data
                */
			   img_rgb2gray_host(gray.ptr<uchar>(), rgb.ptr<uchar>(), gray_ptr, rgb_ptr, HEIGHT, WIDTH, CHANNELS);
#endif
				break;

		}
		timer.stop();

		size_t time_rgb2gray = timer.getElapsed();
		
		count++;
		time_elapsed += (timer.getElapsed())/10000000000.0;

		if (count % 10 == 0)
		{
			cout << "Execution Time (s) = " << time_elapsed << endl;
			time_elapsed = 0;
		}

		imshow("Gray", gray);
		imshow("Invert", invert);
		imshow("Blurred", blr);

		key = waitKey(1);
	}
	cudaFree(rgb_ptr);
	cudaFree(gray_ptr);
	cudaFree(invert_ptr);
	cudaFree(blur_ptr);
}

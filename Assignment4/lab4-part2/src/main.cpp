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

#define BLUR_SIZE 10

#define UNIFIED_MEM 

using namespace std;
using namespace cv;

#define GRAY
//#define INVERT
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
#ifdef INVERT
	uchar* invert_ptr = NULL;
#endif
#ifdef BLUR
	uchar* blur_ptr = NULL;
#endif
	size_t size_img = WIDTH*HEIGHT*CHANNELS*sizeof(uchar); 
#ifndef UNIFIED_MEM
    //TODO: Allocate memory on the GPU device.
    cudaMalloc((void**)&rgb_ptr, size_img);
	//TODO: Declare the host image result matrices
	cudaMalloc((void**)&gray_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#ifdef INVERT
	cudaMalloc((void**)&invert_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#endif
#ifdef BLUR
	cudaMalloc((void**)&blur_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#endif
#else
    //TODO: Allocate unified memory for the necessary matrices
	cudaMallocManaged((void**)&gray_ptr, WIDTH*HEIGHT*sizeof(uchar));
	cudaMallocManaged((void**)&rgb_ptr, size_img);
#ifdef INVERT
	cudaMallocManaged((void**)&invert_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#endif
#ifdef BLUR
	cudaMallocManaged((void**)&blur_ptr, WIDTH*HEIGHT*sizeof(unsigned char));
#endif
    //TODO: Declare the image matrices which point to the unified memory
#endif
#ifndef UNIFIED_MEM
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U);
	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3);
#ifdef INVERT
	Mat invert = Mat(HEIGHT, WIDTH, CV_8U);
#endif
#ifdef BLUR
	Mat blr = Mat(HEIGHT,WIDTH,CV_8U);
#endif
#else
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U, gray_ptr);
	Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3,rgb_ptr);
#ifdef INVERT
	Mat invert = Mat(HEIGHT, WIDTH, CV_8U, invert_ptr);
#endif
#ifdef BLUR
	Mat blr = Mat(HEIGHT,WIDTH,CV_8U, blur_ptr);
#endif
#endif

	//Matrix for OpenCV inversion
	Mat ones = Mat::ones(HEIGHT, WIDTH, CV_8U)*255;
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
#ifdef INVERT
				invert = ones-gray;
#endif
#ifdef BLUR
				blur(gray,blr,Size(BLUR_SIZE,BLUR_SIZE));
#endif
				break;
			case CPU:
                // TODO: 1) Call the CPU functions
				img_rgb2gray_cpu(gray.ptr<uchar>(),rgb.ptr<uchar>(),HEIGHT, WIDTH, CHANNELS);
#ifdef INVERT
				img_invert_cpu(invert.ptr<uchar>(),gray.ptr<uchar>(),HEIGHT, WIDTH);
#endif
#ifdef BLUR
				img_blur_cpu(blr.ptr<uchar>(), gray.ptr<uchar>(),HEIGHT, WIDTH, BLUR_SIZE);
#endif
				break;

			case GPU:
				size_t size_img_1c = WIDTH*HEIGHT*sizeof(uchar);
#ifndef UNIFIED_MEM
                /* TODO: 1) Copy data from host to device
                 *       2) Call GPU host function with device data
                 *       3) Copy data from device to host
                */
			   //Memcpy input matrix
				cudaMemcpy(rgb_ptr, rgb.ptr<uchar>(), size_img, cudaMemcpyHostToDevice);
				img_rgb2gray_host(gray.ptr<uchar>(), rgb.ptr<uchar>(), gray_ptr, rgb_ptr, HEIGHT, WIDTH, CHANNELS);
				cudaMemcpy(gray.ptr<uchar>(), gray_ptr, size_img_1c, cudaMemcpyDeviceToHost);
#ifdef INVERT
				img_invert(invert_ptr, gray_ptr, HEIGHT, WIDTH);
				cudaMemcpy(invert.ptr<uchar>(), invert_ptr, HEIGHT*WIDTH, cudaMemcpyDeviceToHost);
#endif
#ifdef BLUR
				img_blur(blur_ptr, gray_ptr, HEIGHT, WIDTH, BLUR_SIZE);
				cudaMemcpy(blr.ptr<uchar>(), blur_ptr, HEIGHT*WIDTH, cudaMemcpyDeviceToHost);
#endif
#else
                /* TODO: 1) Call GPU host function with unified memory allocated data
                */
			   img_rgb2gray_host(gray.ptr<uchar>(), rgb.ptr<uchar>(), gray_ptr, rgb_ptr, HEIGHT, WIDTH, CHANNELS);
#ifdef INVERT
			   img_invert(invert_ptr, gray_ptr, HEIGHT, WIDTH);
#endif
#ifdef BLUR
			   img_blur(blur_ptr, gray_ptr, HEIGHT, WIDTH, BLUR_SIZE);
#endif
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
#ifdef INVERT
		imshow("Invert", invert);
#endif
#ifdef BLUR
		imshow("Blurred", blr);
#endif

		key = waitKey(1);
	}
	cudaFree(rgb_ptr);
	cudaFree(gray_ptr);
#ifdef INVERT
	cudaFree(invert_ptr);
#endif
#ifdef BLUR
	cudaFree(blur_ptr);
#endif
}

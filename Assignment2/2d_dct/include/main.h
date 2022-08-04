//
//  main.h
//  Lab2
//
//  Created by Alireza on 7/6/16.
//  Copyright © 2016 Alireza. All rights reserved.
//

#ifndef main_h
#define main_h

#include <iostream>
#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>

//#define WIDTH   64
//#define HEIGHT  64

/* student DCT */
cv::Mat student_dct(cv::Mat input);
void initDCT(int WIDTH, int HEIGHT);

#endif /* main_h */

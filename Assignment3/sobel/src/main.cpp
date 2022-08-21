#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

#include "sobel.h"
#include "timer.h"

#define FRAME_NUMBER 1
#define NAIVE 1
#define UNROLLED 2
#define NEON 3

using namespace std;
using namespace cv;

void usage()
{
        cout << "Usage: ./hw3 <MODE> <WIDTH> <HEIGHT>" << endl;
        cout << endl;
        cout << "Mode: " << NAIVE << " - naive sobel" << endl;
        cout << "      " << UNROLLED << " - unrolled sobel" << endl;
        cout << "      " << NEON << " - neon sobel" << endl;
        cout << "WIDTH: 384 default" << endl;
        cout << "HEIGHT: 384 default" << endl;
        cout << "Without HEIGHT, HEIGHT=WIDTH" << endl;
}

int main(int argc, const char * argv[])
{

    int mode = 0;
    LinuxTimer timer;
    double time_elapsed = 0;

    if (argc < 2){
        usage();
        return 0;
    } else {
        mode = stoi(argv[1]);
    }

	cout << "WES237B hw3\n";

	VideoCapture cap("input.raw");

	int WIDTH  = 384;
	int HEIGHT = 384;

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

	Mat frame, gray;
	char key=0;

    for (int f=0; f<FRAME_NUMBER; f++)
	{
		cap >> frame;
		if(frame.empty())
		{
			waitKey();
			break;
		}

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		resize(gray, gray, Size(WIDTH, HEIGHT));
        imwrite("./image_outputs/image_gray.tif", gray);

        //OpenCV sobel filter
        timer.start();
        Mat s_x, s_y, cv_sobel_out;
        Mat s_x_abs, s_y_abs, mag_squared;
        Sobel(gray, s_x, CV_32F, 1, 0, 3, 1, 0, BORDER_ISOLATED);
        Sobel(gray, s_y, CV_32F, 0, 1, 3, 1, 0, BORDER_ISOLATED);
        pow(s_x, 2, s_x_abs);
        pow(s_y, 2, s_y_abs);
        add(s_x_abs , s_y_abs, mag_squared);
        sqrt(mag_squared, cv_sobel_out);
        cv_sobel_out.convertTo(cv_sobel_out, CV_8U);
        timer.stop();
        time_elapsed += (timer.getElapsed())/10000000000.0;
        cout << "Execution Time (s) for OpenCV= " << time_elapsed << endl;
        imwrite("./image_outputs/sobel_opencv.tif", cv_sobel_out);

        Mat sobel_out;
        sobel_out = Mat::zeros(gray.size(), CV_8U);
        char descr[128];
		// Apply filter
        if (mode == NAIVE){
            sobel(gray, sobel_out);
            sprintf(descr, "sobel_naive");
        }
		
        else if (mode == UNROLLED){
            sobel_unroll(gray, sobel_out);
            sprintf(descr, "sobel_unrolled");
        }	

        else if (mode == NEON){
            sobel_neon(gray, sobel_out);
            sprintf(descr, "sobel_neon");
        }

        //Write the current output
        char filename[128];
        sprintf(filename, "./image_outputs/%s.tif", descr);
        imwrite(filename, sobel_out);

		printf("\n");
	}

	return 0;
}

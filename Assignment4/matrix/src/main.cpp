#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>

#include "matrixmul.h"
#include "timer.h"


using namespace std;
using namespace cv;


// ***********************************************
int main(int argc, char const *argv[])
// ***********************************************
{
	if(argc < 3)
	{
		cout << "Usage: " << argv[0] << " M N" << endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);

	srand(time(0));

	float* h_A;
	float* h_B;
	float* h_C;
	float* h_Ccpu;

	//TODO: allocate the unified memory for the input/output matrices. The program will result in a segfault until you complete this line.
    cudaMallocManaged((void**)&h_A, sizeof(float)*M*N);
	cudaMallocManaged((void**)&h_B, sizeof(float)*M*N);
	cudaMallocManaged((void**)&h_C, sizeof(float)*N*N);
	h_Ccpu = (float*)malloc(sizeof(float)*N*N);
	// Initialize matrices
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < M; j++)
		{
			h_A[i * M + j] = rand() / (float)RAND_MAX;
		}
	}

	for(int i = 0; i < M; i++)
	{
		for(int j = 0; j < N; j++)
		{
			h_B[i * N + j] = rand() / (float)RAND_MAX;
		}
	}
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			h_C[i * N + j] = 0;
		}
	}
	// MM GPU
	float time_gpu = -1.f;
	//TODO: call the GPU host wrapper function
	time_gpu = run_mm_gpu(h_A, h_B, h_C, M, N);

	// Profiling
	float time_cpu;

	Timer cpu_timer;
	cpu_timer.start();
	
	Mat cv_A = Mat(N, M, CV_32F, h_A);
	Mat cv_B = Mat(M, N, CV_32F, h_B);
	Mat cv_C = Mat(N, N, CV_32F, h_Ccpu);

	cv_C = cv_A * cv_B;

	cpu_timer.stop();
	time_cpu = cpu_timer.getElapsed();
	cpu_timer.end();

	// Check for errors
	float mse = 0.f;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			float diff = abs(h_C[i*N + j] - h_Ccpu[i*N + j]);
			mse += diff * diff;
		}
	}
	mse /= N*N;

	float rmse = sqrt(mse);

	stringstream ss;
	ss << fixed;
	ss << setprecision(2);
	ss << "Time CPU = " << time_cpu << "ms, Time GPU = " << time_gpu << "ms, Speedup = " << time_cpu/time_gpu << "x, RMSE = ";
	ss << setprecision(5);
	ss << rmse;

	cout << ss.str() << endl;

	// Free memory
	//TODO: free the device memory
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
	free(h_Ccpu);

	return 0;
}

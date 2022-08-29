#ifndef __TIMER_H_
#define __TIMER_H_

#include "cuda_error.h"

class Timer
{

public: 
	Timer() 
	{ 
			CudaSafeCall(cudaEventCreate(&start_));
			CudaSafeCall(cudaEventCreate(&stop_));
	}

	~Timer() 
	{ 
			CudaSafeCall(cudaEventDestroy(start_));
			CudaSafeCall(cudaEventDestroy(stop_));
	}

	void start() { CudaSafeCall(cudaEventRecord(start_, 0)); }

	void stop() { CudaSafeCall(cudaEventRecord(stop_, 0)); CudaSafeCall(cudaEventSynchronize(stop_));}
	
	float getElapsed()
	{
		float time_elapsed;

		CudaSafeCall(cudaEventElapsedTime(&time_elapsed, start_, stop_));

		return time_elapsed;
	}

private:
	cudaEvent_t start_;
	cudaEvent_t stop_;

};

#endif


#ifndef __CUDA_TIMER_H__
#define __CUDA_TIMER_H__


class CudaSynchronizedTimer
{

public:
	CudaSynchronizedTimer()
	{
		cudaEventCreate(&event_start);
		cudaEventCreate(&event_stop);
	}

	~CudaSynchronizedTimer()
	{
		cudaEventDestroy(event_start);
		cudaEventDestroy(event_stop);
	}

	void start(){ cudaEventRecord(event_start); }

	void stop(){ cudaEventRecord(event_stop); }

	float getElapsed()
	{
		cudaEventSynchronize(event_stop);
		float milliseconds;
		cudaEventElapsedTime(&milliseconds, event_start, event_stop);
		return milliseconds;
	}

private:
	cudaEvent_t event_start, event_stop;
};


#endif // __CUDA_TIMER_H__


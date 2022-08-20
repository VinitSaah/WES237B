#include "main.h"

// ----------------------------------------------
// Run a FIR filter on the given input data
// ----------------------------------------------
void fir(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    //TODO
    for(int i = 0; i< (length-filterLength); i++)
    {
        float acc = 0;
	for(int j =0; j< filterLength; j++)
	{
		acc += input[i+j]*coeffs[j];
	}
	output[i] = acc;
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using Loop Unrolling
// ----------------------------------------------
void fir_opt(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    //TODO
    int i = 0;
    for(i = 0; i< (length-filterLength); i++)
    {
	    float acc[4] = {0,0,0,0};
	    for(int j = 0; j< filterLength; j+=4)
	    {
		    acc[0] += input[i+j]*coeffs[j];
		    acc[1] += input[i+1+j]*coeffs[j+1];
		    acc[2] += input[i+2+j]*coeffs[j+2];
		    acc[3] += input[i+3+j]*coeffs[j+3];
	    }

         output[i] = acc[0]+acc[1]+acc[2]+acc[3];	    
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using NEON
// ----------------------------------------------
void fir_neon(float *coeffs, float *input, float *output, int length, int filterLength)
{
	// Apply the filter to each input sample
	    float32x4_t co;
        float32x4_t in;
        float32x4_t ac;
	for (int i = 0; i < length-filterLength; i++)
	{
		float32_t acc1[4] = {0};
		ac = vmovq_n_f32(0);
		for (int j = 0; j <filterLength; j+=4)
	    {
			in = vld1q_f32(&input[i+j]);
			co = vld1q_f32(&coeffs[j]);
			ac = vaddq_f32(ac, vmulq_f32(in, co));
		}
		vst1q_f32(&acc1[0], ac);
		output[i] =  acc1[0] + acc1[1] + acc1[2] + acc1[3];
	}
}

// ----------------------------------------------
// Create filter coefficients
// ----------------------------------------------
void designLPF(float* coeffs, int filterLength, float Fs, float Fx)
// ----------------------------------------------
{
	float lambda = M_PI * Fx / (Fs/2);

	for(int n = 0; n < filterLength; n++)
	{
		float mm = n - (filterLength - 1.0) / 2.0;
		if( mm == 0.0 ) coeffs[n] = lambda / M_PI;
		else coeffs[n] = sin( mm * lambda ) / (mm * M_PI);
	}
}

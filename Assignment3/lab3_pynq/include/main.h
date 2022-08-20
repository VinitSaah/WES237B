#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <cmath>
#include <fstream>

#include <arm_neon.h>

#include <iostream>


// ----------------------------------------------
template<class T>
void printArray(T* data, size_t length, bool doflush = true)
// ----------------------------------------------
{
	for(size_t i = 0; i < length; i++)
	{
		std::cout << data[i] << " ";
	}
	std::cout << "\n";
	if(doflush){ std::cout << std::flush; }
}

void fir(float *coeffs, float *input, float *output, int length, int filterLength);

void fir_opt(float *coeffs, float *input, float *output, int length, int filterLength);

void fir_neon(float *coeffs, float *input, float *output, int length, int filterLength);

void designLPF(float* coeffs, int filterLength, float Fs, float Fx);


#endif // __MAIN_H__


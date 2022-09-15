#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

struct Shape{
	size_t rows, cols;
	
	Shape(size_t rows, size_t cols) : rows(rows), cols(cols) {}

	void transpose()
	{
		size_t tmp = this->rows;
		this->rows = this->cols;
		this->cols = tmp;
	}
};

class Matrix {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Shape shape;

	float* data_device;
	Mat data_host;

	Matrix(size_t rows = 1, size_t cols = 1);
	Matrix(Shape shape);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();
};

#endif

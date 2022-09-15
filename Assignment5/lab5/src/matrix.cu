#include "matrix.h"
#include "nn_exception.h"

Matrix::Matrix(size_t rows, size_t cols) :
	shape(rows, cols), data_device(NULL), data_host(Mat(rows, cols, CV_32F)),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.rows, shape.cols)
{ }

void Matrix::allocateCudaMemory()
{
	if (!device_allocated) 
	{
		cudaMalloc((void **)&data_device, shape.rows * shape.cols * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocated CUDA memory for Tensor3D.");
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory()
{
	if (!host_allocated)
	{
		data_host = Mat(shape.rows, shape.cols, CV_32F);
		host_allocated = true;
	}
}

void Matrix::allocateMemory()
{
	allocateCudaMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape)
{
	if (!device_allocated && !host_allocated)
	{
		this->shape = shape;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice()
{
	if (device_allocated && host_allocated)
	{
		cudaMemcpy(data_device, data_host.ptr<float>(), shape.rows * shape.cols * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	} else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Matrix::copyDeviceToHost()
{
	if (device_allocated && host_allocated)
	{
		cudaMemcpy(data_host.ptr<float>(), data_device, shape.rows * shape.cols * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	} else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

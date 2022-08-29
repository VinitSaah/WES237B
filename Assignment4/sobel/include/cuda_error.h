// Define CUDA_ERROR_CHECK to turn on error checking

#include <stdexcept>
#include <sstream>

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
		std::stringstream ss;
		ss << "cudaSafeCall() failed at " << file << ":" << line << " : " << cudaGetErrorString(err);
		throw std::runtime_error(ss.str());
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
		std::stringstream ss;
		ss << "cudaCheckError() failed at " << file << ":" << line << " : " << cudaGetErrorString(err);
		throw std::runtime_error(ss.str());
    }

    // More careful checking. However, this will affect performance.
    // Uncomment if needed.
    //err = cudaDeviceSynchronize();
    //if( cudaSuccess != err )
    //{
	//	std::stringstream ss;
	//	ss << "cudaCheckError() with sync failed at " << file << ":" << line << " : " << cudaGetErrorString(err);
	//	throw std::runtime_error(ss.str());
    //}
#endif

    return;
}

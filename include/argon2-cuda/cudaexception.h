#ifndef ARGON2_CUDA_CUDAEXCEPTION_H
#define ARGON2_CUDA_CUDAEXCEPTION_H

#include <cuda_runtime.h>

#include <exception>

namespace argon2 {
namespace cuda {

class CudaException : public std::exception {
private:
    cudaError_t res;

public:
    CudaException(cudaError_t res) : res(res) { }

    const char *what() const noexcept override
    {
        return cudaGetErrorString(res);
    }

    static void check(cudaError_t res)
    {
        if (res != cudaSuccess) {
            throw CudaException(res);
        }
    }
};

} // namespace cuda
} // namespace argon2

#endif // ARGON2_CUDA_CUDAEXCEPTION_H

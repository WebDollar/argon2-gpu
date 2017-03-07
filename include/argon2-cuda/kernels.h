#ifndef ARGON2_CUDA_KERNELS_H
#define ARGON2_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

/* workaround weird CMake/CUDA bug: */
#ifdef argon2
#undef argon2
#endif

namespace argon2 {
namespace cuda {

class Argon2KernelRunner
{
private:
    std::uint32_t type, version;
    std::uint32_t passes, lanes, segmentBlocks;
    std::uint32_t batchSize;
    bool bySegment;

    cudaEvent_t start, end;
    cudaStream_t stream;
    void *memory;

    void runKernelSegment(std::uint32_t blockSize,
                          std::uint32_t pass, std::uint32_t slice);
    void runKernelOneshot(std::uint32_t blockSize);

    static uint32_t checkPowerOf2(uint32_t v)
    {
        return (v & (v - 1)) == 0 ? v : 1;
    }

public:
    std::uint32_t getMaxBlockSize() const
    {
        return checkPowerOf2(bySegment ? lanes : batchSize);
    }

    std::uint32_t getBatchSize() const { return batchSize; }
    void *getMemory() const { return memory; }

    Argon2KernelRunner(std::uint32_t type, std::uint32_t version,
                       std::uint32_t passes, std::uint32_t lanes,
                       std::uint32_t segmentBlocks, std::uint32_t batchSize,
                       bool bySegment);
    ~Argon2KernelRunner();

    void run(std::uint32_t blockSize);
    float finish();
};

} // cuda
} // argon2

#endif // ARGON2_CUDA_KERNELS_H

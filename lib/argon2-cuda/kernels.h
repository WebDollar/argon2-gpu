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

void argon2_run_kernel_segment(
        std::uint32_t type, std::uint32_t version, std::uint32_t batchSize,
        cudaStream_t stream, void *memory, std::uint32_t passes,
        std::uint32_t lanes, std::uint32_t segment_blocks, std::uint32_t pass,
        std::uint32_t slice);

void argon2_run_kernel_oneshot(
        std::uint32_t type, std::uint32_t version, std::uint32_t batchSize,
        cudaStream_t stream, void *memory, std::uint32_t passes,
        std::uint32_t lanes, std::uint32_t segment_blocks);

} // cuda
} // argon2

#endif // ARGON2_CUDA_KERNELS_H

#ifndef ARGON2_CUDA_KERNELS_H
#define ARGON2_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

void argon2_run_kernel_segment(
        uint32_t type, uint32_t version, uint32_t batchSize,
        cudaStream_t stream, void *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t pass, uint32_t slice);

#endif // ARGON2_CUDA_KERNELS_H

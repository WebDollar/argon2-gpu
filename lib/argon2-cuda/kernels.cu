/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "kernels.h"
#include "cudaexception.h"

#include <stdexcept>
#ifndef NDEBUG
#include <iostream>
#endif

#define ARGON2_D 0
#define ARGON2_I 1

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

namespace argon2 {
namespace cuda {

using namespace std;

__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

struct block_g {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_l {
    uint32_t lo[ARGON2_QWORDS_IN_BLOCK];
    uint32_t hi[ARGON2_QWORDS_IN_BLOCK];
};

__device__ void move_block(struct block_l *dst, const struct block_l *src,
                           uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        dst->lo[pos_l] = src->lo[pos_l];
        dst->hi[pos_l] = src->hi[pos_l];
    }
}

__device__ void xor_block(struct block_l *dst, const struct block_l *src,
                          uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        dst->lo[pos_l] ^= src->lo[pos_l];
        dst->hi[pos_l] ^= src->hi[pos_l];
    }
}

__device__ void load_block(struct block_l *dst, const struct block_g *src,
                           uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t in = src->data[i * THREADS_PER_LANE + thread];
        dst->lo[i * THREADS_PER_LANE + pos_l] = u64_lo(in);
        dst->hi[i * THREADS_PER_LANE + pos_l] = u64_hi(in);
    }
}

__device__ void load_block_xor(struct block_l *dst, const struct block_g *src,
                               uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t in = src->data[i * THREADS_PER_LANE + thread];
        dst->lo[i * THREADS_PER_LANE + pos_l] ^= u64_lo(in);
        dst->hi[i * THREADS_PER_LANE + pos_l] ^= u64_hi(in);
    }
}

__device__ void store_block(struct block_g *dst, const struct block_l *src,
                            uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t out = u64_build(src->hi[i * THREADS_PER_LANE + pos_l],
                                 src->lo[i * THREADS_PER_LANE + pos_l]);
        dst->data[i * THREADS_PER_LANE + thread] = out;
    }
}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

template<uint32_t bw, uint32_t bh, uint32_t dx, uint32_t dy, uint32_t offset>
__device__ void g(struct block_l *block, uint32_t subblock, uint32_t hash_lane)
{
    uint32_t index[4];
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t bpos = (hash_lane + i * offset) % 4;
        uint32_t x = (subblock * dy + i * dx) * bw + bpos % bw;
        uint32_t y = (subblock * dx + i * dy) * bh + bpos / bw;

        index[i] = y * 16 + (x + (y / 2) * 4) % 16;
    }

    uint64_t a, b, c, d;
    a = u64_build(block->hi[index[0]], block->lo[index[0]]);
    b = u64_build(block->hi[index[1]], block->lo[index[1]]);
    c = u64_build(block->hi[index[2]], block->lo[index[2]]);
    d = u64_build(block->hi[index[3]], block->lo[index[3]]);

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->lo[index[0]] = u64_lo(a);
    block->lo[index[1]] = u64_lo(b);
    block->lo[index[2]] = u64_lo(c);
    block->lo[index[3]] = u64_lo(d);

    block->hi[index[0]] = u64_hi(a);
    block->hi[index[1]] = u64_hi(b);
    block->hi[index[2]] = u64_hi(c);
    block->hi[index[3]] = u64_hi(d);
}

__device__ void shuffle_block(uint32_t thread, struct block_l *block)
{
    uint32_t subblock = (thread >> 2) & 0x7;
    uint32_t hash_lane = (thread >> 0) & 0x3;

    g<4, 1, 1, 0, 0>(block, subblock, hash_lane);

    __syncthreads();

    g<4, 1, 1, 0, 1>(block, subblock, hash_lane);

    __syncthreads();

    g<2, 2, 0, 1, 0>(block, subblock, hash_lane);

    __syncthreads();

    g<2, 2, 0, 1, 1>(block, subblock, hash_lane);
}

__device__ void next_addresses(uint32_t thread,
                               struct block_l *addr, struct block_l *tmp,
                               uint32_t thread_input)
{
    addr->lo[thread] = thread_input;
    addr->hi[thread] = 0;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos = i * THREADS_PER_LANE + thread;
        addr->hi[pos] = addr->lo[pos] = 0;
    }

    __syncthreads();

    shuffle_block(thread, addr);

    __syncthreads();

    addr->lo[thread] ^= thread_input;
    move_block(tmp, addr, thread);

    __syncthreads();

    shuffle_block(thread, addr);

    __syncthreads();

    xor_block(addr, tmp, thread);

    __syncthreads();
}

__device__ void compute_ref_pos(
        uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t lane, uint32_t slice, uint32_t offset,
        uint32_t *ref_lane, uint32_t *ref_index)
{
    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    *ref_lane = *ref_lane % lanes;

    uint32_t base;
    if (pass != 0) {
        base = lane_blocks - segment_blocks;
    } else {
        if (slice == 0) {
            *ref_lane = lane;
        }
        base = slice * segment_blocks;
    }

    uint32_t ref_area_size = base + offset - 1;
    if (*ref_lane != lane) {
        ref_area_size = min(ref_area_size, base);
    }

    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

    if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
        *ref_index += (slice + 1) * segment_blocks;
        if (*ref_index >= lane_blocks) {
            *ref_index -= lane_blocks;
        }
    }
}

struct ref {
    uint32_t ref_lane;
    uint32_t ref_index;
};

struct shmem_precompute {
    struct block_l addr, tmp;
};

/*
 * Refs hierarchy:
 * lanes -> passes -> slices -> blocks
 */
__global__ void argon2i_precompute_kernel(
        struct ref *refs, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks)
{
    uint32_t block_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t warp = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t segment_addr_blocks = (segment_blocks + ARGON2_QWORDS_IN_BLOCK - 1)
            / ARGON2_QWORDS_IN_BLOCK;
    uint32_t block = block_id % segment_addr_blocks;
    uint32_t segment = block_id / segment_addr_blocks;

    uint32_t slice = segment % ARGON2_SYNC_POINTS;
    uint32_t pass_id = segment / ARGON2_SYNC_POINTS;

    uint32_t pass = pass_id % passes;
    uint32_t lane = pass_id / passes;

    extern __shared__ struct shmem_precompute shared_mem2[];

    struct block_l *addr = &shared_mem2[warp].addr;
    struct block_l *tmp = &shared_mem2[warp].tmp;

    uint32_t thread_input;
    switch (thread) {
    case 0:
        thread_input = pass;
        break;
    case 1:
        thread_input = lane;
        break;
    case 2:
        thread_input = slice;
        break;
    case 3:
        thread_input = lanes * segment_blocks * ARGON2_SYNC_POINTS;
        break;
    case 4:
        thread_input = passes;
        break;
    case 5:
        thread_input = ARGON2_I;
        break;
    case 6:
        thread_input = block + 1;
        break;
    default:
        thread_input = 0;
        break;
    }

    next_addresses(thread, addr, tmp, thread_input);

    refs += segment * segment_blocks;

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint32_t ref_index = addr->lo[i * THREADS_PER_LANE + pos_l];
        uint32_t ref_lane  = addr->hi[i * THREADS_PER_LANE + pos_l];

        uint32_t pos = i * THREADS_PER_LANE + thread;
        uint32_t offset = block * ARGON2_QWORDS_IN_BLOCK + pos;
        if (offset < segment_blocks) {
            compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                            &ref_lane, &ref_index);

            refs[offset].ref_index = ref_index;
            refs[offset].ref_lane  = ref_lane;
        }
    }
}

template<uint32_t version>
__device__ void argon2_core(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_l *prev, struct block_l *tmp,
        uint32_t lane_blocks, uint32_t thread, uint32_t pass,
        uint32_t ref_index, uint32_t ref_lane)
{
    struct block_g *mem_ref = memory + ref_lane * lane_blocks + ref_index;

    if (version != ARGON2_VERSION_10 && pass != 0) {
        load_block(tmp, mem_curr, thread);
        load_block_xor(prev, mem_ref, thread);
        xor_block(tmp, prev, thread);
    } else {
        load_block_xor(prev, mem_ref, thread);
        move_block(tmp, prev, thread);
    }

    __syncthreads();

    shuffle_block(thread, prev);

    __syncthreads();

    xor_block(prev, tmp, thread);

    store_block(mem_curr, prev, thread);
}

template<uint32_t version>
__global__ void argon2i_kernel_segment_precompute(
        struct block_g *memory, const struct ref *refs,
        uint32_t passes, uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t slice)
{
    extern __shared__ struct block_l shared_mem[];
    struct block_l *shared = shared_mem;

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select warp's shared memory buffer: */
    shared += threadIdx.y * 2;

    struct block_l *prev = &shared[0];
    struct block_l *tmp  = &shared[1];

    struct block_g *mem_segment =
            memory + lane * lane_blocks + slice * segment_blocks;
    struct block_g *mem_prev, *mem_curr;
    uint32_t start_offset = 0;
    if (pass == 0) {
        if (slice == 0) {
            mem_prev = mem_segment + 1;
            mem_curr = mem_segment + 2;
            start_offset = 2;
        } else {
            mem_prev = mem_segment - 1;
            mem_curr = mem_segment;
        }
    } else {
        mem_prev = mem_segment + (slice == 0 ? lane_blocks : 0) - 1;
        mem_curr = mem_segment;
    }

    load_block(prev, mem_prev, thread);

    refs += (lane * passes + pass) * lane_blocks + slice * segment_blocks;
    refs += start_offset;

    for (uint32_t offset = start_offset; offset < segment_blocks; ++offset) {
        argon2_core<version>(memory, mem_curr, prev, tmp, lane_blocks,
                             thread, pass, refs->ref_index, refs->ref_lane);

        ++mem_curr;
        ++refs;
    }
}

template<uint32_t version>
__global__ void argon2i_kernel_oneshot_precompute(
        struct block_g *memory, const struct ref *refs, uint32_t passes,
        uint32_t lanes, uint32_t segment_blocks)
{
    extern __shared__ struct block_l shared_mem[];
    struct block_l *shared = shared_mem;

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select lane's shared memory buffer: */
    shared += lane * 2;

    struct block_l *prev = &shared[0];
    struct block_l *tmp  = &shared[1];

    struct block_g *mem_lane = memory + lane * lane_blocks;
    struct block_g *mem_prev = mem_lane + 1;
    struct block_g *mem_curr = mem_lane + 2;

    load_block(prev, mem_prev, thread);

    refs += lane * passes * lane_blocks + 2;

    uint32_t skip = 2;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                argon2_core<version>(memory, mem_curr, prev, tmp,
                                     lane_blocks, thread, pass,
                                     refs->ref_index, refs->ref_lane);

                ++mem_curr;
                ++refs;
            }

            __syncthreads();
        }

        mem_curr = mem_lane;
    }
}

template<uint32_t type, uint32_t version>
__device__ void argon2_step(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_l *prev, struct block_l *tmp, struct block_l *addr,
        uint32_t lanes, uint32_t segment_blocks, uint32_t lane_blocks,
        uint32_t thread, uint32_t *thread_input,
        uint32_t lane, uint32_t pass, uint32_t slice, uint32_t offset)
{
    uint32_t ref_index, ref_lane;

    if (type == ARGON2_I) {
        uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
        if (addr_index == 0) {
            if (thread == 6) {
                ++*thread_input;
            }
            next_addresses(thread, addr, tmp, *thread_input);
        }
        uint32_t addr_index_x = addr_index % 16;
        uint32_t addr_index_y = addr_index / 16;
        addr_index = addr_index_y * 16 +
                (addr_index_x + (addr_index_y / 2) * 4) % 16;
        ref_index = addr->lo[addr_index];
        ref_lane = addr->hi[addr_index];
    } else {
        ref_index = prev->lo[0];
        ref_lane = prev->hi[0];
    }

    compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                    &ref_lane, &ref_index);

    argon2_core<version>(memory, mem_curr, prev, tmp, lane_blocks,
                         thread, pass, ref_index, ref_lane);
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_segment(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t pass, uint32_t slice)
{
    extern __shared__ struct block_l shared_mem[];
    struct block_l *shared = shared_mem;

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select warp's shared memory buffer: */
    shared += threadIdx.y * (type == ARGON2_I ? 3 : 2);

    uint32_t thread_input;
    struct block_l *prev = &shared[0];
    struct block_l *tmp  = &shared[1];
    struct block_l *addr;

    if (type == ARGON2_I) {
        addr = &shared[2];

        switch (thread) {
        case 0:
            thread_input = pass;
            break;
        case 1:
            thread_input = lane;
            break;
        case 2:
            thread_input = slice;
            break;
        case 3:
            thread_input = lanes * lane_blocks;
            break;
        case 4:
            thread_input = passes;
            break;
        case 5:
            thread_input = ARGON2_I;
            break;
        default:
            thread_input = 0;
            break;
        }

        if (pass == 0 && slice == 0 && segment_blocks > 2) {
            if (thread == 6) {
                ++thread_input;
            }
            next_addresses(thread, addr, tmp, thread_input);
        }
    }

    struct block_g *mem_segment =
            memory + lane * lane_blocks + slice * segment_blocks;
    struct block_g *mem_prev, *mem_curr;
    uint32_t start_offset = 0;
    if (pass == 0) {
        if (slice == 0) {
            mem_prev = mem_segment + 1;
            mem_curr = mem_segment + 2;
            start_offset = 2;
        } else {
            mem_prev = mem_segment - 1;
            mem_curr = mem_segment;
        }
    } else {
        mem_prev = mem_segment + (slice == 0 ? lane_blocks : 0) - 1;
        mem_curr = mem_segment;
    }

    load_block(prev, mem_prev, thread);

    for (uint32_t offset = start_offset; offset < segment_blocks; ++offset) {
        argon2_step<type, version>(
                    memory, mem_curr, prev, tmp, addr,
                    lanes, segment_blocks, lane_blocks,
                    thread, &thread_input,
                    lane, pass, slice, offset);

        ++mem_curr;
    }
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_oneshot(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks)
{
    extern __shared__ struct block_l shared_mem[];
    struct block_l *shared = shared_mem;

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select lane's shared memory buffer: */
    shared += lane * (type == ARGON2_I ? 3 : 2);

    struct block_l *prev = &shared[0];
    struct block_l *tmp  = &shared[1];
    struct block_l *addr;
    uint32_t thread_input;

    if (type == ARGON2_I) {
        addr = &shared[2];

        switch (thread) {
        case 1:
            thread_input = lane;
            break;
        case 3:
            thread_input = lanes * lane_blocks;
            break;
        case 4:
            thread_input = passes;
            break;
        case 5:
            thread_input = ARGON2_I;
            break;
        default:
            thread_input = 0;
            break;
        }

        if (segment_blocks > 2) {
            if (thread == 6) {
                ++thread_input;
            }
            next_addresses(thread, addr, tmp, thread_input);
        }
    }

    struct block_g *mem_lane = memory + lane * lane_blocks;
    struct block_g *mem_prev = mem_lane + 1;
    struct block_g *mem_curr = mem_lane + 2;

    load_block(prev, mem_prev, thread);

    uint32_t skip = 2;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                argon2_step<type, version>(
                            memory, mem_curr, prev, tmp, addr,
                            lanes, segment_blocks, lane_blocks,
                            thread, &thread_input,
                            lane, pass, slice, offset);

                ++mem_curr;
            }

            __syncthreads();

            if (type == ARGON2_I) {
                if (thread == 2) {
                    ++thread_input;
                }
                if (thread == 6) {
                    thread_input = 0;
                }
            }
        }
        if (type == ARGON2_I) {
            if (thread == 0) {
                ++thread_input;
            }
            if (thread == 2) {
                thread_input = 0;
            }
        }
        mem_curr = mem_lane;
    }
}

Argon2KernelRunner::Argon2KernelRunner(
        uint32_t type, uint32_t version, uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, uint32_t batchSize, bool bySegment,
        bool precompute)
    : type(type), version(version), passes(passes), lanes(lanes),
      segmentBlocks(segmentBlocks), batchSize(batchSize), bySegment(bySegment),
      precompute(precompute), stream(nullptr), memory(nullptr),
      refs(nullptr), start(nullptr), end(nullptr)
{
    // FIXME: check overflow:
    uint32_t memorySize = lanes * segmentBlocks * ARGON2_SYNC_POINTS
            * ARGON2_BLOCK_SIZE * batchSize;

    CudaException::check(cudaMallocManaged(&memory, memorySize,
                                           cudaMemAttachHost));

    CudaException::check(cudaEventCreate(&start));
    CudaException::check(cudaEventCreate(&end));

    CudaException::check(cudaStreamCreate(&stream));
    CudaException::check(cudaStreamAttachMemAsync(stream, memory));
    CudaException::check(cudaStreamSynchronize(stream));

    if (type == ARGON2_I && precompute) {
        uint32_t segments = passes * lanes * ARGON2_SYNC_POINTS;

        uint32_t refsSize = segments * segmentBlocks * sizeof(struct ref);

#ifndef NDEBUG
        std::cerr << "[INFO] Allocating " << refsSize << " bytes for refs..."
                  << std::endl;
#endif

        CudaException::check(cudaMallocManaged(&refs, refsSize,
                                               cudaMemAttachHost));

        CudaException::check(cudaStreamAttachMemAsync(stream, refs));
        CudaException::check(cudaStreamSynchronize(stream));

        precomputeRefs();
        CudaException::check(cudaStreamSynchronize(stream));
    }
}

void Argon2KernelRunner::precomputeRefs()
{
    struct ref *refs = (struct ref *)this->refs;

    uint32_t segmentAddrBlocks = (segmentBlocks + ARGON2_QWORDS_IN_BLOCK - 1)
            / ARGON2_QWORDS_IN_BLOCK;
    uint32_t segments = passes * lanes * ARGON2_SYNC_POINTS;

    dim3 blocks = dim3(1, segments * segmentAddrBlocks);
    dim3 threads = dim3(THREADS_PER_LANE);

    uint32_t shmemSize = sizeof(struct shmem_precompute);
    argon2i_precompute_kernel<<<blocks, threads, shmemSize, stream>>>(
            refs, passes, lanes, segmentBlocks);
}

Argon2KernelRunner::~Argon2KernelRunner()
{
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (end != nullptr) {
        cudaEventDestroy(end);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (memory != nullptr) {
        cudaFree(memory);
    }
    if (refs != nullptr) {
        cudaFree(refs);
    }
}

void Argon2KernelRunner::runKernelSegment(uint32_t lanesPerBlock,
                                          uint32_t jobsPerBlock,
                                          uint32_t pass, uint32_t slice)
{
    if (lanesPerBlock > lanes || lanes % lanesPerBlock != 0) {
        throw std::logic_error("Invalid lanesPerBlock!");
    }

    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
        throw std::logic_error("Invalid jobsPerBlock!");
    }

    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(1, lanes / lanesPerBlock, batchSize / jobsPerBlock);
    dim3 threads = dim3(THREADS_PER_LANE, lanesPerBlock, jobsPerBlock);
    uint32_t blockSize = lanesPerBlock * jobsPerBlock;
    if (type == ARGON2_I) {
        if (precompute) {
            uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 2;
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2i_kernel_segment_precompute<ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2i_kernel_segment_precompute<ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        } else {
            uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 3;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            } else {
                argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks,
                            pass, slice);
            }
        }
    } else {
        uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 2;
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_10>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks,
                        pass, slice);
        } else {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_13>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks,
                        pass, slice);
        }
    }
}

void Argon2KernelRunner::runKernelOneshot(uint32_t lanesPerBlock,
                                          uint32_t jobsPerBlock)
{
    if (lanesPerBlock != lanes) {
        throw std::logic_error("Invalid lanesPerBlock!");
    }

    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
        throw std::logic_error("Invalid jobsPerBlock!");
    }

    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(1, 1, batchSize / jobsPerBlock);
    dim3 threads = dim3(THREADS_PER_LANE, lanes, jobsPerBlock);
    uint32_t blockSize = lanesPerBlock * jobsPerBlock;
    if (type == ARGON2_I) {
        if (precompute) {
            uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 2;
            struct ref *refs = (struct ref *)this->refs;
            if (version == ARGON2_VERSION_10) {
                argon2i_kernel_oneshot_precompute<ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            } else {
                argon2i_kernel_oneshot_precompute<ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, refs, passes, lanes, segmentBlocks);
            }
        } else {
            uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 3;
            if (version == ARGON2_VERSION_10) {
                argon2_kernel_oneshot<ARGON2_I, ARGON2_VERSION_10>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            } else {
                argon2_kernel_oneshot<ARGON2_I, ARGON2_VERSION_13>
                        <<<blocks, threads, shared_size, stream>>>(
                            memory_blocks, passes, lanes, segmentBlocks);
            }
        }
    } else {
        uint32_t shared_size = blockSize * ARGON2_BLOCK_SIZE * 2;
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_oneshot<ARGON2_D, ARGON2_VERSION_10>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks);
        } else {
            argon2_kernel_oneshot<ARGON2_D, ARGON2_VERSION_13>
                    <<<blocks, threads, shared_size, stream>>>(
                        memory_blocks, passes, lanes, segmentBlocks);
        }
    }
}

void Argon2KernelRunner::run(uint32_t lanesPerBlock, uint32_t jobsPerBlock)
{
    CudaException::check(cudaEventRecord(start, stream));

    if (bySegment) {
        for (uint32_t pass = 0; pass < passes; pass++) {
            for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
                runKernelSegment(lanesPerBlock, jobsPerBlock, pass, slice);
            }
        }
    } else {
        runKernelOneshot(lanesPerBlock, jobsPerBlock);
    }

    CudaException::check(cudaGetLastError());

    CudaException::check(cudaEventRecord(end, stream));
}

float Argon2KernelRunner::finish()
{
    CudaException::check(cudaStreamSynchronize(stream));

    float time = 0.0;
    CudaException::check(cudaEventElapsedTime(&time, start, end));
    return time;
}

} // cuda
} // argon2

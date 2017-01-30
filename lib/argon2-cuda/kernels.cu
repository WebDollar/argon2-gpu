#include "kernels.h"

#include <cuda_runtime.h>

#define ARGON2_D 0
#define ARGON2_I 1

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t upsample(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = (uint32_t)x;
    uint32_t ylo = (uint32_t)y;
    return x + y + 2 * upsample(__umulhi(xlo, ylo), xlo * ylo);
}

struct block_g {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_l {
    uint32_t lo[ARGON2_QWORDS_IN_BLOCK];
    uint32_t hi[ARGON2_QWORDS_IN_BLOCK];
};

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
    a = upsample(block->hi[index[0]], block->lo[index[0]]);
    b = upsample(block->hi[index[1]], block->lo[index[1]]);
    c = upsample(block->hi[index[2]], block->lo[index[2]]);
    d = upsample(block->hi[index[3]], block->lo[index[3]]);

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->lo[index[0]] = (uint32_t)a;
    block->lo[index[1]] = (uint32_t)b;
    block->lo[index[2]] = (uint32_t)c;
    block->lo[index[3]] = (uint32_t)d;

    block->hi[index[0]] = (uint32_t)(a >> 32);
    block->hi[index[1]] = (uint32_t)(b >> 32);
    block->hi[index[2]] = (uint32_t)(c >> 32);
    block->hi[index[3]] = (uint32_t)(d >> 32);
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

__device__ void fill_block(uint32_t thread, struct block_g *ref_block,
                           struct block_l *prev_block,
                           struct block_l *next_block)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t in = ref_block->data[i * THREADS_PER_LANE + thread];
        next_block->lo[pos_l] = prev_block->lo[pos_l] ^= (uint32_t)in;
        next_block->hi[pos_l] = prev_block->hi[pos_l] ^= (uint32_t)(in >> 32);
    }

    __syncthreads();

    shuffle_block(thread, prev_block);

    __syncthreads();

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        next_block->lo[pos_l] ^= prev_block->lo[pos_l];
        next_block->hi[pos_l] ^= prev_block->hi[pos_l];
    }
}

__device__ void fill_block_xor(uint32_t thread, struct block_g *ref_block,
                               struct block_l *prev_block,
                               struct block_l *next_block)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t in = ref_block->data[i * THREADS_PER_LANE + thread];
        next_block->lo[pos_l] ^= prev_block->lo[pos_l] ^= (uint32_t)in;
        next_block->hi[pos_l] ^= prev_block->hi[pos_l] ^= (uint32_t)(in >> 32);
    }

    __syncthreads();

    shuffle_block(thread, prev_block);

    __syncthreads();

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        next_block->lo[pos_l] ^= prev_block->lo[pos_l];
        next_block->hi[pos_l] ^= prev_block->hi[pos_l];
    }
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

    tmp->lo[thread] = addr->lo[thread] ^= thread_input;
    tmp->hi[thread] = addr->hi[thread];
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos = i * THREADS_PER_LANE + thread;
        tmp->lo[pos] = addr->lo[pos];
        tmp->hi[pos] = addr->hi[pos];
    }

    __syncthreads();

    shuffle_block(thread, addr);

    __syncthreads();

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos = i * THREADS_PER_LANE + thread;
        addr->lo[pos] ^= tmp->lo[pos];
        addr->hi[pos] ^= tmp->hi[pos];
    }

    __syncthreads();
}


template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_segment(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t pass, uint32_t slice)
{
    uint32_t job_id = blockIdx.x;
    uint32_t lane   = blockIdx.y;
    uint32_t thread = threadIdx.z;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;

    uint32_t thread_input;
    __shared__ struct block_l local_curr, local_prev, local_addr;
    struct block_l *curr = &local_curr;
    struct block_l *prev = &local_prev;

    if (type == ARGON2_I) {
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
            next_addresses(thread, &local_addr, curr, thread_input);
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

    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        uint64_t in = mem_prev->data[i * THREADS_PER_LANE + thread];
        prev->lo[i * THREADS_PER_LANE + pos_l] = (uint32_t)in;
        prev->hi[i * THREADS_PER_LANE + pos_l] = (uint32_t)(in >> 32);
    }

    for (uint32_t offset = start_offset; offset < segment_blocks; ++offset) {
        uint32_t pseudo_rand_lo, pseudo_rand_hi;

        if (type == ARGON2_I) {
            uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
            if (addr_index == 0) {
                if (thread == 6) {
                    ++thread_input;
                }
                next_addresses(thread, &local_addr, curr, thread_input);
            }
            uint32_t addr_index_x = addr_index % 16;
            uint32_t addr_index_y = addr_index / 16;
            addr_index = addr_index_y * 16 +
                    (addr_index_x + (addr_index_y / 2) * 4) % 16;
            pseudo_rand_lo = local_addr.lo[addr_index];
            pseudo_rand_hi = local_addr.hi[addr_index];
        } else {
                pseudo_rand_lo = prev->lo[0];
                pseudo_rand_hi = prev->hi[0];
        }

        uint32_t ref_lane = pseudo_rand_hi % lanes;

        uint32_t base;
        if (pass != 0) {
            base = lane_blocks - segment_blocks;
        } else {
            if (slice == 0) {
                ref_lane = lane;
            }
            base = slice * segment_blocks;
        }

        uint32_t ref_area_size = base + offset - 1;
        if (ref_lane != lane) {
            ref_area_size = min(ref_area_size, base);
        }

        uint32_t ref_index = pseudo_rand_lo;
        ref_index = __mulhi(ref_index, ref_index);
        ref_index = ref_area_size - 1 - __mulhi(ref_area_size, ref_index);

        if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
            ref_index += (slice + 1) * segment_blocks;
            ref_index %= lane_blocks;
        }

        struct block_g *mem_ref = (struct block_g *)(
                    memory + ref_lane * lane_blocks + ref_index);

        /* NOTE: no need to wrap fill_block in barriers, since
         * it starts & ends in 'nicely parallel' memory operations
         * like we do in this loop (IOW: this thread only depends on
         * its own data w.r.t. these boundaries) */
        if (version == ARGON2_VERSION_10) {
            fill_block(thread, mem_ref, prev, curr);
        } else {
            if (pass != 0) {
                for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
                    uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
                    uint64_t in = mem_curr->data[i * THREADS_PER_LANE + thread];
                    curr->lo[i * THREADS_PER_LANE + pos_l] = (uint32_t)in;
                    curr->hi[i * THREADS_PER_LANE + pos_l] = (uint32_t)(in >> 32);
                }

                fill_block_xor(thread, mem_ref, prev, curr);
            } else {
                fill_block(thread, mem_ref, prev, curr);
            }
        }

        for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
            uint32_t pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
            uint64_t out = upsample(curr->hi[i * THREADS_PER_LANE + pos_l],
                                 curr->lo[i * THREADS_PER_LANE + pos_l]);
            mem_curr->data[i * THREADS_PER_LANE + thread] = out;
        }

        /* swap curr and prev buffers: */
        struct block_l *tmp = curr;
        curr = prev;
        prev = tmp;

        ++mem_curr;
    }
}

void argon2_run_kernel_segment(
        uint32_t type, uint32_t version, uint32_t batchSize,
        cudaStream_t stream, void *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t pass, uint32_t slice)
{
    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(batchSize, lanes);
    dim3 threads = dim3(1, 1, THREADS_PER_LANE);
    if (type == ARGON2_I) {
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_10>
                    <<<blocks, threads, 0, stream>>>(memory_blocks, passes, lanes,
                                                     segment_blocks, pass, slice);
        } else {
            argon2_kernel_segment<ARGON2_I, ARGON2_VERSION_13>
                    <<<blocks, threads, 0, stream>>>(memory_blocks, passes, lanes,
                                                     segment_blocks, pass, slice);
        }
    } else {
        if (version == ARGON2_VERSION_10) {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_10>
                    <<<blocks, threads, 0, stream>>>(memory_blocks, passes, lanes,
                                                     segment_blocks, pass, slice);
        } else {
            argon2_kernel_segment<ARGON2_D, ARGON2_VERSION_13>
                    <<<blocks, threads, 0, stream>>>(memory_blocks, passes, lanes,
                                                     segment_blocks, pass, slice);
        }
    }
}

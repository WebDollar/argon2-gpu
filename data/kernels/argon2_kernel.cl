/* C compatibility For dumb IDEs: */
#ifndef __OPENCL_VERSION__
#ifndef __cplusplus
typedef int bool;
#endif
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef size_t uintptr_t;
typedef ptrdiff_t intptr_t;
#ifndef __kernel
#define __kernel
#endif
#ifndef __global
#define __global
#endif
#ifndef __private
#define __private
#endif
#ifndef __local
#define __local
#endif
#ifndef __constant
#define __constant const
#endif
#endif /* __OPENCL_VERSION__ */

#define ARGON2_D  0
#define ARGON2_I  1
#define ARGON2_ID 2

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

#ifndef ARGON2_VERSION
#define ARGON2_VERSION ARGON2_VERSION_13
#endif

#ifndef ARGON2_TYPE
#define ARGON2_TYPE ARGON2_I
#endif

#define F(x, y) ((x) + (y) + 2 * upsample( \
    mul_hi((uint)(x), (uint)(y)), \
    (uint)(x) * (uint)(y) \
    ))

#define rotr64(x, n) rotate(x, (ulong)(64 - (n)))

struct block_g {
    ulong data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_l {
    uint lo[ARGON2_QWORDS_IN_BLOCK];
    uint hi[ARGON2_QWORDS_IN_BLOCK];
};

void g(__local struct block_l *block, uint subblock, uint hash_lane,
       uint bw, uint bh, uint dx, uint dy, uint offset)
{
    uint index[4];
    for (uint i = 0; i < 4; i++) {
        uint bpos = (hash_lane + i * offset) % 4;
        uint x = (subblock * dy + i * dx) * bw + bpos % bw;
        uint y = (subblock * dx + i * dy) * bh + bpos / bw;

        index[i] = y * 16 + (x + (y / 2) * 4) % 16;
    }

    ulong a, b, c, d;
    a = upsample(block->hi[index[0]], block->lo[index[0]]);
    b = upsample(block->hi[index[1]], block->lo[index[1]]);
    c = upsample(block->hi[index[2]], block->lo[index[2]]);
    d = upsample(block->hi[index[3]], block->lo[index[3]]);

    a = F(a, b);
    d = rotr64(d ^ a, 32);
    c = F(c, d);
    b = rotr64(b ^ c, 24);
    a = F(a, b);
    d = rotr64(d ^ a, 16);
    c = F(c, d);
    b = rotr64(b ^ c, 63);

    block->lo[index[0]] = (uint)a;
    block->lo[index[1]] = (uint)b;
    block->lo[index[2]] = (uint)c;
    block->lo[index[3]] = (uint)d;

    block->hi[index[0]] = (uint)(a >> 32);
    block->hi[index[1]] = (uint)(b >> 32);
    block->hi[index[2]] = (uint)(c >> 32);
    block->hi[index[3]] = (uint)(d >> 32);
}

void shuffle_block(__local struct block_l *block, uint thread)
{
    uint subblock = (thread >> 2) & 0x7;
    uint hash_lane = (thread >> 0) & 0x3;

    g(block, subblock, hash_lane, 4, 1, 1, 0, 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g(block, subblock, hash_lane, 4, 1, 1, 0, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    g(block, subblock, hash_lane, 2, 2, 0, 1, 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    g(block, subblock, hash_lane, 2, 2, 0, 1, 1);
}

void fill_block(__global const struct block_g *restrict ref_block,
                __local struct block_l *restrict prev_block,
                __local struct block_l *restrict next_block,
                uint thread)
{
    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        ulong in = ref_block->data[i * THREADS_PER_LANE + thread];
        next_block->lo[pos_l] = prev_block->lo[pos_l] ^= (uint)in;
        next_block->hi[pos_l] = prev_block->hi[pos_l] ^= (uint)(in >> 32);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        prev_block->lo[pos_l] ^= next_block->lo[pos_l];
        prev_block->hi[pos_l] ^= next_block->hi[pos_l];
    }
}

#if ARGON2_VERSION != ARGON2_VERSION_10
void fill_block_xor(__global const struct block_g *restrict ref_block,
                    __local struct block_l *restrict prev_block,
                    __local struct block_l *restrict next_block,
                    uint thread)
{
    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        ulong in = ref_block->data[i * THREADS_PER_LANE + thread];
        next_block->lo[pos_l] ^= prev_block->lo[pos_l] ^= (uint)in;
        next_block->hi[pos_l] ^= prev_block->hi[pos_l] ^= (uint)(in >> 32);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(prev_block, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = i * THREADS_PER_LANE +
                (thread & 0x10) + ((thread + i * 4) & 0xf);
        prev_block->lo[pos_l] ^= next_block->lo[pos_l];
        prev_block->hi[pos_l] ^= next_block->hi[pos_l];
    }
}
#endif

#if ARGON2_TYPE == ARGON2_I || ARGON2_TYPE == ARGON2_ID
void next_addresses(uint thread_input,
                    __local struct block_l *restrict addr,
                    __local struct block_l *restrict tmp,
                    uint thread)
{
    addr->lo[thread] = thread_input;
    addr->hi[thread] = 0;
    for (uint i = 1; i < QWORDS_PER_THREAD; i++) {
        uint pos = i * THREADS_PER_LANE + thread;
        addr->hi[pos] = addr->lo[pos] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    tmp->lo[thread] = addr->lo[thread] ^= thread_input;
    tmp->hi[thread] = addr->hi[thread];
    for (uint i = 1; i < QWORDS_PER_THREAD; i++) {
        uint pos = i * THREADS_PER_LANE + thread;
        tmp->lo[pos] = addr->lo[pos];
        tmp->hi[pos] = addr->hi[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    shuffle_block(addr, thread);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos = i * THREADS_PER_LANE + thread;
        addr->lo[pos] ^= tmp->lo[pos];
        addr->hi[pos] ^= tmp->hi[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

__kernel void argon2_kernel_segment(
        __global struct block_g *memory, uint passes, uint lanes,
        uint segment_blocks, uint pass, uint slice)
{
    size_t job_id = get_global_id(2);
    uint lane = (uint)get_global_id(1);
    uint thread = (uint)get_global_id(0);

    uint lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;

    __local struct block_l local_curr, local_prev, local_addr;
    __local struct block_l *curr = &local_curr;
    __local struct block_l *prev = &local_prev;

    uint thread_input;
#if ARGON2_TYPE == ARGON2_I || ARGON2_TYPE == ARGON2_ID
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
        thread_input = ARGON2_TYPE;
        break;
    default:
        thread_input = 0;
        break;
    }

    if (pass == 0 && slice == 0 && segment_blocks > 2) {
        if (thread == 6) {
            ++thread_input;
        }
        next_addresses(thread_input, &local_addr, curr, thread);
    }
#endif

    __global struct block_g *mem_segment = memory
            + lane * lane_blocks + slice * segment_blocks;
    __global struct block_g *mem_prev, *mem_curr;
    uint start_offset = 0;
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

    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        ulong in = mem_prev->data[i * THREADS_PER_LANE + thread];
        prev->lo[i * THREADS_PER_LANE + pos_l] = (uint)in;
        prev->hi[i * THREADS_PER_LANE + pos_l] = (uint)(in >> 32);
    }

    for (uint offset = start_offset; offset < segment_blocks; ++offset) {
        uint pseudo_rand_lo, pseudo_rand_hi;
        if (ARGON2_TYPE == ARGON2_I || (ARGON2_TYPE == ARGON2_ID && pass == 0 &&
                                        slice < ARGON2_SYNC_POINTS / 2)) {
            uint addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
            if (addr_index == 0) {
                if (thread == 6) {
                    ++thread_input;
                }
                next_addresses(thread_input, &local_addr, curr, thread);
            }
            uint addr_index_x = addr_index % 16;
            uint addr_index_y = addr_index / 16;
            addr_index = addr_index_y * 16 +
                    (addr_index_x + (addr_index_y / 2) * 4) % 16;
            pseudo_rand_lo = local_addr.lo[addr_index];
            pseudo_rand_hi = local_addr.hi[addr_index];
        } else {
            pseudo_rand_lo = prev->lo[0];
            pseudo_rand_hi = prev->hi[0];
        }

        uint ref_lane = pseudo_rand_hi % lanes;

        uint base;
        if (pass != 0) {
            base = lane_blocks - segment_blocks;
        } else {
            if (slice == 0) {
                ref_lane = lane;
            }
            base = slice * segment_blocks;
        }

        uint ref_area_size = base + offset - 1;
        if (ref_lane != lane) {
            ref_area_size = min(ref_area_size, base);
        }

        uint ref_index = pseudo_rand_lo;
        ref_index = mul_hi(ref_index, ref_index);
        ref_index = ref_area_size - 1 - mul_hi(ref_area_size, ref_index);

        if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
            ref_index += (slice + 1) * segment_blocks;
            if (ref_index >= lane_blocks) {
                ref_index -= lane_blocks;
            }
        }

        __global struct block_g *mem_ref = memory +
                ref_lane * lane_blocks + ref_index;

        /* NOTE: no need to wrap fill_block in barriers, since
         * it starts & ends in 'nicely parallel' memory operations
         * like we do in this loop (IOW: this thread only depends on
         * its own data w.r.t. these boundaries) */
#if ARGON2_VERSION == ARGON2_VERSION_10
        fill_block(mem_ref, prev, curr, thread);
#else
        if (pass != 0) {
            for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
                uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
                ulong in = mem_curr->data[i * THREADS_PER_LANE + thread];
                curr->lo[i * THREADS_PER_LANE + pos_l] = (uint)in;
                curr->hi[i * THREADS_PER_LANE + pos_l] = (uint)(in >> 32);
            }

            fill_block_xor(mem_ref, prev, curr, thread);
        } else {
            fill_block(mem_ref, prev, curr, thread);
        }
#endif

        for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
            uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
            ulong out = upsample(prev->hi[i * THREADS_PER_LANE + pos_l],
                                 prev->lo[i * THREADS_PER_LANE + pos_l]);
            mem_curr->data[i * THREADS_PER_LANE + thread] = out;
        }

        ++mem_curr;
    }
}

#if ARGON2_TYPE == ARGON2_I || ARGON2_TYPE == ARGON2_ID
#define SHARED_BLOCKS 3
#else
#define SHARED_BLOCKS 2
#endif

__kernel void argon2_kernel_oneshot(
        __global struct block_g *memory, __local struct block_l *shared,
        uint passes, uint lanes, uint segment_blocks)
{
    size_t job_id = get_global_id(2);
    uint lane = (uint)get_global_id(1);
    uint thread = (uint)get_global_id(0);

    uint lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += job_id * lanes * lane_blocks;
    /* select lane's shared memory buffer: */
    shared += lane * SHARED_BLOCKS;

    __local struct block_l *restrict curr = &shared[0];
    __local struct block_l *restrict prev = &shared[1];
    __local struct block_l *restrict addr;
    uint thread_input;

#if ARGON2_TYPE == ARGON2_I || ARGON2_TYPE == ARGON2_ID
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
        thread_input = ARGON2_TYPE;
        break;
    default:
        thread_input = 0;
        break;
    }

    if (segment_blocks > 2) {
        if (thread == 6) {
            ++thread_input;
        }
        next_addresses(thread_input, addr, curr, thread);
    }
#endif

    __global struct block_g *mem_lane = memory + lane * lane_blocks;
    __global struct block_g *mem_prev = mem_lane + 1;
    __global struct block_g *mem_curr = mem_lane + 2;

    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
        uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
        ulong in = mem_prev->data[i * THREADS_PER_LANE + thread];
        prev->lo[i * THREADS_PER_LANE + pos_l] = (uint)in;
        prev->hi[i * THREADS_PER_LANE + pos_l] = (uint)(in >> 32);
    }
    uint skip = 2;
    for (uint pass = 0; pass < passes; ++pass) {
        for (uint slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                uint pseudo_rand_lo, pseudo_rand_hi;
                if (ARGON2_TYPE == ARGON2_I || (ARGON2_TYPE == ARGON2_ID &&
                        pass == 0 && slice < ARGON2_SYNC_POINTS / 2)) {
                    uint addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
                    if (addr_index == 0) {
                        if (thread == 6) {
                            ++thread_input;
                        }
                        next_addresses(thread_input, addr, curr, thread);
                    }
                    uint addr_index_x = addr_index % 16;
                    uint addr_index_y = addr_index / 16;
                    addr_index = addr_index_y * 16 +
                            (addr_index_x + (addr_index_y / 2) * 4) % 16;
                    pseudo_rand_lo = addr->lo[addr_index];
                    pseudo_rand_hi = addr->hi[addr_index];
                } else {
                    pseudo_rand_lo = prev->lo[0];
                    pseudo_rand_hi = prev->hi[0];
                }

                uint ref_lane = pseudo_rand_hi % lanes;

                uint base;
                if (pass != 0) {
                    base = lane_blocks - segment_blocks;
                } else {
                    if (slice == 0) {
                        ref_lane = lane;
                    }
                    base = slice * segment_blocks;
                }

                uint ref_area_size = base + offset - 1;
                if (ref_lane != lane) {
                    ref_area_size = min(ref_area_size, base);
                }

                uint ref_index = pseudo_rand_lo;
                ref_index = mul_hi(ref_index, ref_index);
                ref_index = ref_area_size - 1 - mul_hi(ref_area_size, ref_index);

                if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
                    ref_index += (slice + 1) * segment_blocks;
                    if (ref_index >= lane_blocks) {
                        ref_index -= lane_blocks;
                    }
                }

                __global struct block_g *mem_ref = memory +
                        ref_lane * lane_blocks + ref_index;

                /* NOTE: no need to wrap fill_block in barriers, since
                 * it starts & ends in 'nicely parallel' memory operations
                 * like we do in this loop (IOW: this thread only depends on
                 * its own data w.r.t. these boundaries) */
#if ARGON2_VERSION == ARGON2_VERSION_10
                fill_block(mem_ref, prev, curr, thread);
#else
                if (pass != 0) {
                    for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
                        uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
                        ulong in = mem_curr->data[i * THREADS_PER_LANE + thread];
                        curr->lo[i * THREADS_PER_LANE + pos_l] = (uint)in;
                        curr->hi[i * THREADS_PER_LANE + pos_l] = (uint)(in >> 32);
                    }

                    fill_block_xor(mem_ref, prev, curr, thread);
                } else {
                    fill_block(mem_ref, prev, curr, thread);
                }
#endif

                for (uint i = 0; i < QWORDS_PER_THREAD; i++) {
                    uint pos_l = (thread & 0x10) + ((thread + i * 4) & 0xf);
                    ulong out = upsample(prev->hi[i * THREADS_PER_LANE + pos_l],
                                         prev->lo[i * THREADS_PER_LANE + pos_l]);
                    mem_curr->data[i * THREADS_PER_LANE + thread] = out;
                }

                ++mem_curr;
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
#if ARGON2_TYPE == ARGON2_I || ARGON2_TYPE == ARGON2_ID
            if (thread == 2) {
                ++thread_input;
            }
            if (thread == 6) {
                thread_input = 0;
            }
#endif
        }
#if ARGON2_TYPE == ARGON2_I
        if (thread == 0) {
            ++thread_input;
        }
        if (thread == 2) {
            thread_input = 0;
        }
#endif
        mem_curr = mem_lane;
    }
}

# argon2-gpu

A proof-of-concept GPU password cracker for Argon2 hashes.

[Argon2](https://github.com/P-H-C/phc-winner-argon2) is a password hashing function created by Alex Biryukov, Daniel Dinu, and Dmitry Khovratovich. It was designed to be resistant against brute-force attacks using specialized hardware, such as GPUs, ASICs, or FPGAs. In July 2015, it was announced as the winner of the [Password Hashing Competition](https://password-hashing.net).

The main goal of this project is to provide an efficient GPU implementation of Argon2 that can be used to estimate the speed and efficiency of Argon2 GPU cracking, in order to support or refute claims of its GPU cracking resistance.

## Backends

Currently, the project implements two backends -- one that uses the NVIDIA's [CUDA](https://www.nvidia.com/object/cuda_home_new.html) framework and another one that uses the [OpenCL](https://www.khronos.org/opencl/) API. Note, however, that the OpenCL backend is currently out-of-sync with the CUDA backend and is very inefficient.

## Argon2 variants

Argon2-gpu supports all Argon2 variants (Argon2i, Argon2d, and Argon2id) and versions (1.3 and 1.0).

## Performance

The CUDA implementation can reach about 40-60 GiB/s (divide by time cost * memory cost to get hashes per second) on an NVIDIA Tesla K20X. For comparison, a fast Intel Xeon processor can only reach about 10 GiB/s.

## CUDA kernel variants

The CUDA implementation has three variants, which are currently implemented in separate branches:

 * `master` -- uses only shared memory operations; is much slower than other two
 * `warp-shuffle` -- uses warp shuffle instructions; doesn't use shared memory at all
 * `warp-shuffle-shared` -- like `warp-shuffle`, but uses less regsters (compensated by using shared memory); this one is about as fast as `warp-shuffle`, but can be a little slower or faster in some edge cases

In addition, Argon2i and Argon2id implementations support a special 'precompute' mode, which makes them as fast as Argon2d, but uses a bit more memory (depending on time cost and memory cost). This mode can be enabled/disabled at runtime.

# Tutorial for Node-WebDollar to mine with GPU (unfriendly)

## Install cmake new version

```
    cmake
    https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu
```

## Install opencl headers

```
apt-get install opencl-headers
sudo apt-get install ocl-icd-libopencl1

sudo ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

```




## CUDA Installation:

1.
    https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8

2. There is a mistake in the tutorial

    sudo nano /etc/environment
    Be sure you have at the end of the file **:/usr/local/cuda/bin** (it is important to have `:` at the beginning something exists before)


## Final compile

0. `git submodule update --init`

1. `make`
2. copy the `argon2-gpu-test` and `data` folder to Node-WebDollar **/dist_bundle/GPU/**


### Tutorial how to use for manual tests

ns_per_hash=$(./argon2-gpu-bench \
                                    -t $type -v $version \
                                    $precompute_flag \
                                    -m $mode -d $device -k $kernel \
                                    -b $batch_size -s $samples \
                                    -T $t_cost -M $m_cost -L $lanes \
                                    -o ns-per-hash --output-mode mean)
                                ret=$?


./argon2-gpu-bench -t d -v 1.3 -m cuda -d 0 -k by-segment -b 50 -s 5 -T 2 -M 256 -L 2 -o ns-per-hash --output-mode mean


./argon2 Satoshi_is_Finney -d -t 2 -m 8 -p 2 -e -l 32


./argon2-gpu-test -m opencl -d 0




# argon2-gpu [![build status](https://gitlab.com/omos/argon2-gpu/badges/master/build.svg)](https://gitlab.com/omos/argon2-gpu/commits/master)

A proof-of-concept GPU password cracker for Argon2 hashes.

[Argon2](https://github.com/P-H-C/phc-winner-argon2) is a password hashing function created by Alex Biryukov, Daniel Dinu, and Dmitry Khovratovich. It was designed to be resistant against brute-force attacks using specialized hardware, such as GPUs, ASICs, or FPGAs. In July 2015, it was announced as the winner of the [Password Hashing Competition](https://password-hashing.net).

The main goal of this project is to provide an efficient GPU implementation of Argon2 that can be used to estimate the speed and efficiency of Argon2 GPU cracking, in order to support or refute claims of its GPU cracking resistance.

## Backends

Currently, the project implements two backends -- one that uses the NVIDIA's [CUDA](https://www.nvidia.com/object/cuda_home_new.html) framework and another one that uses the [OpenCL](https://www.khronos.org/opencl/) API.

## Argon2 variants

Argon2-gpu supports all Argon2 variants (Argon2i, Argon2d, and Argon2id) and versions (1.3 and 1.0).

## Performance

The CUDA implementation can reach about 40-60 GiB/s (divide by time cost * memory cost  * 1024 B to get hashes per second) on an NVIDIA Tesla K20X. For comparison, a fast Intel Xeon processor can only reach about 10 GiB/s.

## Building

This project uses the [CMake](https://cmake.org/) build system.

First, if you haven't cloned the repository using `git clone --recursive`, you need to run:

```bash
git submodule update --init
```

Then, to prepare build:

```bash
cmake -DCMAKE_BUILD_TYPE=Release .
```

Finally, just run `make` to build the code. Note that to use the OpenCL backend, you need to have the `data` subdirectory in the working directory (if you have the binaries in a different directory, just create a symlink using `ln -s <path_to_repo>/data data`).

### Building without CUDA support

To build without CUDA support, just add `-DNO_CUDA=TRUE` to the CMake command line. In this case the CUDA backend will always report 0 devices.

If CMake fails to find a usable CUDA installation, the project will be automatically built without CUDA support.

## CUDA kernel variants

The CUDA implementation has three variants, which are currently implemented in separate branches:

 * `master` -- uses only shared memory operations; is somewhat slower than the other two
 * `warp-shuffle` -- uses warp shuffle instructions; doesn't use shared memory at all
 * `warp-shuffle-shared` -- like `warp-shuffle`, but uses less regsters (compensated by using shared memory); this one is about as fast as `warp-shuffle`, but can be a little slower or faster in some edge cases

In addition, Argon2i and Argon2id implementations support a special 'precompute' mode, which makes them as fast as Argon2d, but uses a bit more memory (depending on time cost and memory cost). This mode is also supported by the OpenCL backend and can be enabled/disabled at runtime.

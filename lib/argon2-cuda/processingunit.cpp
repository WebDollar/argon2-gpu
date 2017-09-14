#include "processingunit.h"

#include "cudaexception.h"

#include <limits>
#ifndef NDEBUG
#include <iostream>
#endif

namespace argon2 {
namespace cuda {

static void setCudaDevice(int deviceIndex)
{
    int currentIndex = -1;
    CudaException::check(cudaGetDevice(&currentIndex));
    if (currentIndex != deviceIndex) {
        CudaException::check(cudaSetDevice(deviceIndex));
    }
}

static bool isPowerOfTwo(std::uint32_t x)
{
    return (x & (x - 1)) == 0;
}

ProcessingUnit::ProcessingUnit(
        const ProgramContext *programContext, const Argon2Params *params,
        const Device *device, std::size_t batchSize, bool bySegment,
        bool precomputeRefs)
    : programContext(programContext), params(params), device(device),
      runner(programContext->getArgon2Type(),
             programContext->getArgon2Version(), params->getTimeCost(),
             params->getLanes(), params->getSegmentBlocks(), batchSize,
             bySegment, precomputeRefs),
      bestLanesPerBlock(runner.getMinLanesPerBlock()),
      bestJobsPerBlock(runner.getMinJobsPerBlock())
{
    setCudaDevice(device->getDeviceIndex());

    /* pre-fill first blocks with pseudo-random data: */
    for (std::size_t i = 0; i < batchSize; i++) {
        setPassword(i, NULL, 0);
    }

    if (runner.getMaxLanesPerBlock() > runner.getMinLanesPerBlock()
            && isPowerOfTwo(runner.getMaxLanesPerBlock())) {
#ifndef NDEBUG
        std::cerr << "[INFO] Tuning lanes per block..." << std::endl;
#endif

        float bestTime = std::numeric_limits<float>::infinity();
        for (std::uint32_t lpb = 1; lpb <= runner.getMaxLanesPerBlock();
             lpb *= 2)
        {
            float time;
            try {
                runner.run(lpb, bestJobsPerBlock);
                time = runner.finish();
            } catch(CudaException &ex) {
#ifndef NDEBUG
                std::cerr << "[WARN]   CUDA error on " << lpb
                          << " lanes per block: " << ex.what() << std::endl;
#endif
                break;
            }

#ifndef NDEBUG
            std::cerr << "[INFO]   " << lpb << " lanes per block: "
                      << time << " ms" << std::endl;
#endif

            if (time < bestTime) {
                bestTime = time;
                bestLanesPerBlock = lpb;
            }
        }
#ifndef NDEBUG
        std::cerr << "[INFO] Picked " << bestLanesPerBlock
                  << " lanes per block." << std::endl;
#endif
    }

    /* Only tune jobs per block if we hit maximum lanes per block: */
    if (bestLanesPerBlock == runner.getMaxLanesPerBlock()
            && runner.getMaxJobsPerBlock() > runner.getMinJobsPerBlock()
            && isPowerOfTwo(runner.getMaxJobsPerBlock())) {
#ifndef NDEBUG
        std::cerr << "[INFO] Tuning jobs per block..." << std::endl;
#endif

        float bestTime = std::numeric_limits<float>::infinity();
        for (std::uint32_t jpb = 1; jpb <= runner.getMaxJobsPerBlock();
             jpb *= 2)
        {
            float time;
            try {
                runner.run(bestLanesPerBlock, jpb);
                time = runner.finish();
            } catch(CudaException &ex) {
#ifndef NDEBUG
                std::cerr << "[WARN]   CUDA error on " << jpb
                          << " jobs per block: " << ex.what() << std::endl;
#endif
                break;
            }

#ifndef NDEBUG
            std::cerr << "[INFO]   " << jpb << " jobs per block: "
                      << time << " ms" << std::endl;
#endif

            if (time < bestTime) {
                bestTime = time;
                bestJobsPerBlock = jpb;
            }
        }
#ifndef NDEBUG
        std::cerr << "[INFO] Picked " << bestJobsPerBlock
                  << " jobs per block." << std::endl;
#endif
    }
}

void ProcessingUnit::setPassword(std::size_t index, const void *pw,
                                 std::size_t pwSize)
{
    std::size_t size = params->getLanes() * 2 * ARGON2_BLOCK_SIZE;
    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    params->fillFirstBlocks(buffer.get(), pw, pwSize,
                            programContext->getArgon2Type(),
                            programContext->getArgon2Version());
    runner.writeInputMemory(index, buffer.get());
}

void ProcessingUnit::getHash(std::size_t index, void *hash)
{
    std::size_t size = params->getLanes() * ARGON2_BLOCK_SIZE;
    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    runner.readOutputMemory(index, buffer.get());
    params->finalize(hash, buffer.get());
}

void ProcessingUnit::beginProcessing()
{
    setCudaDevice(device->getDeviceIndex());
    runner.run(bestLanesPerBlock, bestJobsPerBlock);
}

void ProcessingUnit::endProcessing()
{
    runner.finish();
}

} // namespace cuda
} // namespace argon2

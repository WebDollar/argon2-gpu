#include "processingunit.h"

#include "cudaexception.h"
#include "kernels.h"

#include <limits>
#ifndef NDEBUG
#include <iostream>
#endif

namespace argon2 {
namespace cuda {

ProcessingUnit::ProcessingUnit(
        const ProgramContext *programContext, const Argon2Params *params,
        const Device *device, std::size_t batchSize, bool bySegment,
        bool precomputeRefs)
    : programContext(programContext), params(params), device(device),
      runner(programContext->getArgon2Type(),
             programContext->getArgon2Version(), params->getTimeCost(),
             params->getLanes(), params->getSegmentBlocks(), batchSize,
             bySegment, precomputeRefs),
      bestBlockSize(1)
{
    CudaException::check(cudaSetDevice(device->getDeviceIndex()));

    if (runner.getMaxBlockSize() > 1) {
#ifndef NDEBUG
        std::cerr << "[INFO] Benchmarking block size..." << std::endl;
#endif

        float bestTime = std::numeric_limits<float>::infinity();
        for (std::uint32_t blockSize = 1; blockSize <= runner.getMaxBlockSize();
             blockSize *= 2)
        {
            float time;
            try {
                runner.run(blockSize);
                time = runner.finish();
            } catch(CudaException &ex) {
#ifndef NDEBUG
                std::cerr << "[WARN]   Exception on block size " << blockSize
                          << ": " << ex.what() << std::endl;
#endif
                break;
            }

#ifndef NDEBUG
            std::cerr << "[INFO]   Block size " << blockSize << ": "
                      << time << " ms" << std::endl;
#endif

            if (time < bestTime) {
                bestTime = time;
                bestBlockSize = blockSize;
            }
        }
#ifndef NDEBUG
        std::cerr << "[INFO] Picked block size: " << bestBlockSize << std::endl;
#endif
    }
}

ProcessingUnit::PasswordWriter::PasswordWriter(
        ProcessingUnit &parent, std::size_t index)
    : params(parent.params),
      type(parent.programContext->getArgon2Type()),
      version(parent.programContext->getArgon2Version()),
      dest(static_cast<std::uint8_t *>(parent.runner.getMemory()))
{
    dest += index * params->getMemorySize();
}

void ProcessingUnit::PasswordWriter::moveForward(std::size_t offset)
{
    dest += offset * params->getMemorySize();
}

void ProcessingUnit::PasswordWriter::moveBackwards(std::size_t offset)
{
    dest -= offset * params->getMemorySize();
}

void ProcessingUnit::PasswordWriter::setPassword(
        const void *pw, std::size_t pwSize) const
{
    params->fillFirstBlocks(dest, pw, pwSize, type, version);
}

ProcessingUnit::HashReader::HashReader(
        ProcessingUnit &parent, std::size_t index)
    : params(parent.params),
      src(static_cast<const std::uint8_t *>(parent.runner.getMemory())),
      buffer(new std::uint8_t[params->getOutputLength()])
{
    src += index * params->getMemorySize();
}

void ProcessingUnit::HashReader::moveForward(std::size_t offset)
{
    src += offset * params->getMemorySize();
}

void ProcessingUnit::HashReader::moveBackwards(std::size_t offset)
{
    src -= offset * params->getMemorySize();
}

const void *ProcessingUnit::HashReader::getHash() const
{
    params->finalize(buffer.get(), src);
    return buffer.get();
}

void ProcessingUnit::beginProcessing()
{
    CudaException::check(cudaSetDevice(device->getDeviceIndex()));
    runner.run(bestBlockSize);
}

void ProcessingUnit::endProcessing()
{
    runner.finish();
}

} // namespace cuda
} // namespace argon2

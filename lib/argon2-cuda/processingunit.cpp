#include "processingunit.h"

#include "cudaexception.h"
#include "kernels.h"

namespace argon2 {
namespace cuda {

ProcessingUnit::ProcessingUnit(
        const ProgramContext *programContext, const Argon2Params *params,
        const Device *device, std::size_t batchSize,
        bool bySegment)
    : programContext(programContext), params(params),
      device(device), batchSize(batchSize), bySegment(bySegment),
      stream(nullptr), memoryBuffer(nullptr)
{
    // FIXME: check memSize out of bounds
    CudaException::check(cudaStreamCreate(&stream));

    memorySize = params->getMemorySize() * batchSize;

    CudaException::check(cudaMallocManaged(&memoryBuffer, memorySize,
                                           cudaMemAttachHost));

    CudaException::check(cudaStreamAttachMemAsync(stream, memoryBuffer));
    CudaException::check(cudaStreamSynchronize(stream));

    /*
    if (bySegment) {
        kernel = cl::Kernel(programContext->getProgram(),
                            "argon2_kernel_segment");
        kernel.setArg<cl::Buffer>(0, memoryBuffer);
        kernel.setArg<cl_uint>(1, params->getTimeCost());
        kernel.setArg<cl_uint>(2, lanes);
        kernel.setArg<cl_uint>(3, params->getSegmentBlocks());
    } else {
        auto localMemSize = (std::size_t)lanes * ARGON2_BLOCK_SIZE;
        if (programContext->getArgon2Type() == ARGON2_I) {
            localMemSize *= 3;
        } else {
            localMemSize *= 2;
        }

        kernel = cl::Kernel(programContext->getProgram(),
                            "argon2_kernel_oneshot");
        kernel.setArg<cl::Buffer>(0, memoryBuffer);
        kernel.setArg<cl::LocalSpaceArg>(1, { localMemSize });
        kernel.setArg<cl_uint>(2, params->getTimeCost());
        kernel.setArg<cl_uint>(3, lanes);
        kernel.setArg<cl_uint>(4, params->getSegmentBlocks());
    }
    */
}

ProcessingUnit::~ProcessingUnit()
{
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (memoryBuffer != nullptr) {
        cudaFree(memoryBuffer);
    }
}

ProcessingUnit::PasswordWriter::PasswordWriter(
        ProcessingUnit &parent, std::size_t index)
    : params(parent.params),
      type(parent.programContext->getArgon2Type()),
      version(parent.programContext->getArgon2Version()),
      dest(static_cast<std::uint8_t *>(parent.memoryBuffer))
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
      src(static_cast<const std::uint8_t *>(parent.memoryBuffer)),
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
    if (bySegment) {
        for (unsigned int pass = 0; pass < params->getTimeCost(); pass++) {
            for (unsigned int slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
                argon2_run_kernel_segment(
                            programContext->getArgon2Type(),
                            programContext->getArgon2Version(),
                            batchSize, stream, (unsigned long *)memoryBuffer,
                            params->getTimeCost(),
                            params->getLanes(),
                            params->getSegmentBlocks(),
                            pass, slice);
            }
        }
    } else {
        // TODO
    }
}

void ProcessingUnit::endProcessing()
{
    CudaException::check(cudaStreamSynchronize(stream));
}

} // namespace cuda
} // namespace argon2

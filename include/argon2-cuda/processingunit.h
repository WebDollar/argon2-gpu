#ifndef ARGON2_CUDA_PROCESSINGUNIT_H
#define ARGON2_CUDA_PROCESSINGUNIT_H

#include <memory>

#include "programcontext.h"
#include "kernels.h"
#include "argon2-gpu-common/argon2params.h"

namespace argon2 {
namespace cuda {

class ProcessingUnit
{
private:
    const ProgramContext *programContext;
    const Argon2Params *params;
    const Device *device;

    Argon2KernelRunner runner;
    uint32_t bestLanesPerBlock;
    uint32_t bestJobsPerBlock;

public:
    class PasswordWriter
    {
    private:
        const Argon2Params *params;
        Type type;
        Version version;
        std::uint8_t *dest;

    public:
        PasswordWriter(ProcessingUnit &parent, std::size_t index = 0);

        void moveForward(std::size_t offset);
        void moveBackwards(std::size_t offset);

        void setPassword(const void *pw, std::size_t pwSize) const;
    };

    class HashReader
    {
    private:
        const Argon2Params *params;
        const std::uint8_t *src;
        std::unique_ptr<uint8_t[]> buffer;

    public:
        HashReader(ProcessingUnit &parent, std::size_t index = 0);

        void moveForward(std::size_t offset);
        void moveBackwards(std::size_t offset);

        const void *getHash() const;
    };

    std::size_t getBatchSize() const { return runner.getBatchSize(); }

    ProcessingUnit(
            const ProgramContext *programContext, const Argon2Params *params,
            const Device *device, std::size_t batchSize,
            bool bySegment = true, bool precomputeRefs = false);

    void beginProcessing();
    void endProcessing();
};

} // namespace cuda
} // namespace argon2

#endif // ARGON2_CUDA_PROCESSINGUNIT_H

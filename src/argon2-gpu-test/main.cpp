#include "argon2-gpu-common/argon2params.h"
#include "argon2-opencl/processingunit.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/cudaexception.h"

#include "testcase.h"
#include "testvectors.h"

#include <iostream>
#include <array>
#include <cstdint>
#include <cstring>

using namespace argon2;

template<class Device, class GlobalContext, class ProgramContext,
         class ProcessingUnit>
std::size_t runTests(const GlobalContext &global, const Device &device,
                     Type type, Version version,
                     const TestCase *casesFrom, const TestCase *casesTo)
{
    std::cout << "Running tests for Argon2";
    if (type == ARGON2_I) {
        std::cout << "i";
    } else if (type == ARGON2_D) {
        std::cout << "d";
    } else if (type == ARGON2_ID) {
        std::cout << "id";
    }
    std::cout << " v" << (version == ARGON2_VERSION_10 ? "1.0" : "1.3")
              << "..." << std::endl;

    std::size_t failures = 0;
    ProgramContext progCtx(&global, { device }, type, version);
    for (auto bySegment : {true, false}) {
        const std::array<bool, 2> precomputeOpts = { false, true };
        auto precBegin = precomputeOpts.begin();
        auto precEnd = precomputeOpts.end();
        if (type == ARGON2_D) {
            precEnd--;
        }
        for (auto precIt = precBegin; precIt != precEnd; precIt++) {
            for (auto tc = casesFrom; tc < casesTo; ++tc) {
                bool precompute = *precIt;
                std::cout << "  "
                          << (bySegment  ? "[by-segment] " : "[oneshot]    ")
                          << (precompute ? "[precompute] " : "[in-place]   ");
                tc->dump(std::cout);
                std::cout << "... ";

                auto &params = tc->getParams();
                ProcessingUnit pu(&progCtx, &params, &device, 1, bySegment,
                                  precompute);
                {
                    typename ProcessingUnit::PasswordWriter writer(pu);
                    writer.setPassword(tc->getInput(), tc->getInputLength());
                }
                pu.beginProcessing();
                pu.endProcessing();

                typename ProcessingUnit::HashReader hash(pu);
                bool res = std::memcmp(tc->getOutput(), hash.getHash(),
                                       params.getOutputLength()) == 0;
                if (!res) {
                    ++failures;
                    std::cout << "FAIL" << std::endl;
                } else {
                    std::cout << "PASS" << std::endl;
                }
            }
        }
    }
    if (!failures) {
        std::cout << "  ALL PASSED" << std::endl;
    }
    return failures;
}

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define ARRAY_BEGIN(a) (a)
#define ARRAY_END(a) ((a) + ARRAY_SIZE(a))

template<class Device, class GlobalContext,
         class ProgramContext, class ProcessingUnit>
void runAllTests(std::size_t &failures)
{
    GlobalContext global;
    auto &devices = global.getAllDevices();
    auto &device = devices[0];

    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_I, ARGON2_VERSION_10,
             ARRAY_BEGIN(CASES_I_10), ARRAY_END(CASES_I_10));
    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_I, ARGON2_VERSION_13,
             ARRAY_BEGIN(CASES_I_13), ARRAY_END(CASES_I_13));
    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_D, ARGON2_VERSION_10,
             ARRAY_BEGIN(CASES_D_10), ARRAY_END(CASES_D_10));
    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_D, ARGON2_VERSION_13,
             ARRAY_BEGIN(CASES_D_13), ARRAY_END(CASES_D_13));
    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_ID, ARGON2_VERSION_10,
             ARRAY_BEGIN(CASES_ID_10), ARRAY_END(CASES_ID_10));
    failures += runTests<Device, GlobalContext, ProgramContext, ProcessingUnit>
            (global, device, ARGON2_ID, ARGON2_VERSION_13,
             ARRAY_BEGIN(CASES_ID_13), ARRAY_END(CASES_ID_13));
}

int main(void) {
    std::size_t failures = 0;

    std::cout << "Running CUDA tests..." << std::endl;
    try {
        runAllTests<cuda::Device, cuda::GlobalContext, cuda::ProgramContext,
                cuda::ProcessingUnit>(failures);
    } catch (cuda::CudaException &err) {
        std::cerr << "CUDA ERROR: " << err.what() << std::endl;
        return 2;
    }

    std::cout << "Running OpenCL tests..." << std::endl;
    try {
        runAllTests<opencl::Device, opencl::GlobalContext, opencl::ProgramContext,
                opencl::ProcessingUnit>(failures);
    } catch (cl::Error &err) {
        std::cerr << "OpenCL ERROR: " << err.err() << ": "
                  << err.what() << std::endl;
        return 2;
    }

    if (failures) {
        std::cout << failures << " TESTS FAILED!" << std::endl;
        return 1;
    }
    return 0;
}

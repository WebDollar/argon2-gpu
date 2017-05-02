#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

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

                auto buffer = std::unique_ptr<std::uint8_t[]>(
                            new std::uint8_t[params.getOutputLength()]);

                ProcessingUnit pu(&progCtx, &params, &device, 1, bySegment,
                                  precompute);
                pu.setPassword(0, tc->getInput(), tc->getInputLength());
                pu.beginProcessing();
                pu.endProcessing();
                pu.getHash(0, buffer.get());

                bool res = std::memcmp(tc->getOutput(), buffer.get(),
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
int runAllTests(const char *progname, const char *name, std::size_t deviceIndex,
                bool listDevices, std::size_t &failures)
{
    GlobalContext global;
    auto &devices = global.getAllDevices();

    if (listDevices) {
        for (std::size_t i = 0; i < devices.size(); i++) {
            auto &device = devices[i];
            std::cout << "Device #" << i << ": " << device.getInfo()
                      << std::endl;
        }
        return 0;
    }

    if (deviceIndex >= devices.size()) {
        std::cerr << progname << ": Device index out of range!" << std::endl;
        return 2;
    }

    auto &device = devices[deviceIndex];
    std::cout << "Running " << name << " tests..." << std::endl;
    std::cout << "Using device #" << deviceIndex << ": " << device.getName()
              << std::endl;

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
    return 0;
}

using namespace libcommandline;

struct Arguments
{
    bool showHelp = false;
    bool listDevices = false;

    std::string mode = "cuda";

    std::size_t deviceIndex = 0;
};

static CommandLineParser<Arguments> buildCmdLineParser()
{
    static const auto positional = PositionalArgumentHandler<Arguments>(
                [] (Arguments &, const std::string &) {});

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode; },
            "mode", 'm', "mode in which to run ('cuda' for CUDA or 'opencl' for OpenCL)", "cuda", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = (std::size_t)index;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true; },
            "help", '?', "show this help and exit")
    };

    return CommandLineParser<Arguments>(
        "A tool for testing the argon2-opencl and argon2-cuda libraries.",
        positional, options);
}

int main(int, const char * const *argv) {
    CommandLineParser<Arguments> parser = buildCmdLineParser();

    Arguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }

    std::size_t failures = 0;
    if (args.mode == "cuda") {
        try {
            ret = runAllTests<cuda::Device, cuda::GlobalContext,
                    cuda::ProgramContext, cuda::ProcessingUnit>(
                        argv[0], "CUDA", args.deviceIndex, args.listDevices,
                        failures);
        } catch (cuda::CudaException &err) {
            std::cerr << argv[0] << ": CUDA ERROR: " << err.what() << std::endl;
            return 2;
        }
    } else if (args.mode == "opencl") {
        try {
            ret = runAllTests<opencl::Device, opencl::GlobalContext,
                    opencl::ProgramContext, opencl::ProcessingUnit>(
                        argv[0], "OpenCL", args.deviceIndex, args.listDevices,
                        failures);
        } catch (cl::Error &err) {
            std::cerr << argv[0] << ": OpenCL ERROR: " << err.err() << ": "
                      << err.what() << std::endl;
            return 2;
        }
    } else {
        std::cerr << argv[0] << ": invalid mode: " << args.mode << std::endl;
        return 2;
    }
    if (ret)
        return ret;

    if (failures) {
        std::cout << failures << " TESTS FAILED!" << std::endl;
        return 1;
    }
    return 0;
}

#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "argon2-gpu-common/argon2params.h"
#include "argon2-opencl/processingunit.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/cudaexception.h"

#include "argon2-functions.h"
#include "main-functions.h"

#include "global.cpp"

//#include "argon2-worker.cpp"
//Argon2Worker worker;

//#include "server.cpp"

using namespace libcommandline;

struct Arguments{

    bool showHelp = false;
    bool listDevices = false;


    std::string mode = "cuda";
    std::string run = "normal";
    std::string filename = "input.txt";
    int batch = 3000;

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
                    [] (Arguments &state, const std::string &run) { state.run = run; },
                    "run", 'r', "runing normal or 'realtime'", "normal", "RUN"),

            new ArgumentOption<Arguments>(
                    [] (Arguments &state, const std::string &filename) { state.filename = filename; },
                    "filename", 'f', "filename", "", "FILENAME"),

            new ArgumentOption<Arguments>(
                    makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t)index;
                    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

            new ArgumentOption<Arguments>(
                    makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t batch) {
                        state.batch = (std::size_t)batch;
                    }), "batch", 'b', "batch", "3000", "BATCH"),


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
    if (ret != 0)
        return ret;

    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }

#ifdef ARGON2_SELECTABLE_IMPL
    argon2_select_impl(nullptr, "[libargon2] ");
#endif


    std::size_t failures = 0;

    if (args.mode == "cuda")
        try {

            if (args.run == "socket") ret = initializeSystem<cuda::Device, cuda::GlobalContext, cuda::ProgramContext, cuda::ProcessingUnit>( argv[0], "CUDA", args.deviceIndex, args.listDevices, failures, args.filename, args.batch); else
            if (args.run == "realtime") ret = runRealTime<cuda::Device, cuda::GlobalContext, cuda::ProgramContext, cuda::ProcessingUnit>( argv[0], "CUDA", args.deviceIndex, args.listDevices, failures); else
                ret = runAllTests<cuda::Device, cuda::GlobalContext, cuda::ProgramContext, cuda::ProcessingUnit>( argv[0], "CUDA", args.deviceIndex, args.listDevices, failures, args.filename);

        } catch (cuda::CudaException &err) {
            std::cerr << argv[0] << ": CUDA ERROR: " << err.what() << std::endl;
            return 2;
        }
    else if (args.mode == "opencl")
        try {

            if (args.run == "socket") ret = initializeSystem<opencl::Device, opencl::GlobalContext, opencl::ProgramContext, opencl::ProcessingUnit>( argv[0], "OpenCL", args.deviceIndex, args.listDevices, failures, args.filename, args.batch); else
            if (args.run == "realtime") ret = runRealTime<opencl::Device, opencl::GlobalContext, opencl::ProgramContext, opencl::ProcessingUnit>( argv[0], "OpenCL", args.deviceIndex, args.listDevices, failures); else
                ret = runAllTests<opencl::Device, opencl::GlobalContext, opencl::ProgramContext, opencl::ProcessingUnit>( argv[0], "OpenCL", args.deviceIndex, args.listDevices, failures, args.filename);

        } catch (cl::Error &err) {
            std::cerr << argv[0] << ": OpenCL ERROR: " << err.err() << ": "
                      << err.what() << std::endl;
            return 2;
        }
    else {
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




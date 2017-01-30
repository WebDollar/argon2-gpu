#include "argon2-gpu-common/argon2params.h"

#include <iostream>
#include <cstdint>
#include <cstring>

using namespace argon2;

static char toHex(std::uint8_t digit) {
    return digit >= 10 ? 'a' + (digit - 10) : '0' + digit;
}

static void dumpBytes(std::ostream &out, const void *data, std::size_t size)
{
    auto bdata = static_cast<const std::uint8_t *>(data);
    for (std::size_t i = 0; i < size; i++) {
        auto val = bdata[i];
        out << toHex((val >> 4) & 0xf) << toHex(val & 0xf);
    }
}

class TestCase
{
private:
    Argon2Params params;
    const void *output;
    const void *input;
    std::size_t inputLength;

public:
    const Argon2Params &getParams() const { return params; }
    const void *getOutput() const { return output; }
    const void *getInput() const { return input; }
    std::size_t getInputLength() const { return inputLength; }

    TestCase(const Argon2Params &params, const void *output,
             const void *input, std::size_t inputLength)
        : params(params), output(output),
          input(input), inputLength(inputLength)
    {
    }

    void dump(std::ostream &out) const
    {
        out << "t=" << params.getTimeCost()
            << " m=" << params.getMemoryCost()
            << " p=" << params.getLanes()
            << " pass=";
        dumpBytes(out, input, inputLength);

        if (params.getSaltLength()) {
            out << " salt=";
            dumpBytes(out, params.getSalt(), params.getSaltLength());
        }

        if (params.getAssocDataLength()) {
            out << " ad=";
            dumpBytes(out, params.getAssocData(), params.getAssocDataLength());
        }

        if (params.getSecretLength()) {
            out << " secret=";
            dumpBytes(out, params.getSecret(), params.getSecretLength());
        }
    }
};

template<class Device, class GlobalContext, class ProgramContext,
         class ProcessingUnit>
std::size_t runTests(const GlobalContext &global, const Device &device,
                     Type type, Version version,
                     const TestCase *casesFrom, const TestCase *casesTo)
{
    std::cerr << "Running tests for Argon2"
              << (type == ARGON2_I ? "i" : "d")
              << " v" << (version == ARGON2_VERSION_10 ? "1.0" : "1.3")
              << "..." << std::endl;

    std::size_t failures = 0;
    ProgramContext progCtx(&global, { device }, type, version);
    for (auto bySegment : {true, false}) {
        for (auto tc = casesFrom; tc < casesTo; ++tc) {
            std::cerr << "  " << (bySegment ? "[by-segment] " : "[oneshot] ");
            tc->dump(std::cerr);
            std::cerr << "... ";

            auto &params = tc->getParams();
            ProcessingUnit pu(&progCtx, &params, &device, 1, bySegment);

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
                std::cerr << "FAIL" << std::endl;
            } else {
                std::cerr << "PASS" << std::endl;
            }
        }
    }
    if (!failures) {
        std::cerr << "  ALL PASSED" << std::endl;
    }
    return failures;
}

const TestCase CASES_I_10[] = {
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xf6\xc4\xdb\x4a\x54\xe2\xa3\x70"
        "\x62\x7a\xff\x3d\xb6\x17\x6b\x94"
        "\xa2\xa2\x09\xa6\x2c\x8e\x36\x15"
        "\x27\x11\x80\x2f\x7b\x30\xc6\x94",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 1
        },
        "\xfd\x4d\xd8\x3d\x76\x2c\x49\xbd"
        "\xea\xf5\x7c\x47\xbd\xcd\x0c\x2f"
        "\x1b\xab\xf8\x63\xfd\xeb\x49\x0d"
        "\xf6\x3e\xde\x99\x75\xfc\xcf\x06",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 2
        },
        "\xb6\xc1\x15\x60\xa6\xa9\xd6\x1e"
        "\xac\x70\x6b\x79\xa2\xf9\x7d\x68"
        "\xb4\x46\x3a\xa3\xad\x87\xe0\x0c"
        "\x07\xe2\xb0\x1e\x90\xc5\x64\xfb",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            1, UINT32_C(1) << 16, 1
        },
        "\x81\x63\x05\x52\xb8\xf3\xb1\xf4"
        "\x8c\xdb\x19\x92\xc4\xc6\x78\x64"
        "\x3d\x49\x0b\x2b\x5e\xb4\xff\x6c"
        "\x4b\x34\x38\xb5\x62\x17\x24\xb2",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            4, UINT32_C(1) << 16, 1
        },
        "\xf2\x12\xf0\x16\x15\xe6\xeb\x5d"
        "\x74\x73\x4d\xc3\xef\x40\xad\xe2"
        "\xd5\x1d\x05\x24\x68\xd8\xc6\x94"
        "\x40\xa3\xa1\xf2\xc1\xc2\x84\x7b",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xe9\xc9\x02\x07\x4b\x67\x54\x53"
        "\x1a\x3a\x0b\xe5\x19\xe5\xba\xf4"
        "\x04\xb3\x0c\xe6\x9b\x3f\x01\xac"
        "\x3b\xf2\x12\x29\x96\x01\x09\xa3",
        "differentpassword", 17
    },
    {
        {
            32, "diffsalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\x79\xa1\x03\xb9\x0f\xe8\xae\xf8"
        "\x57\x0c\xb3\x1f\xc8\xb2\x22\x59"
        "\x77\x89\x16\xf8\x33\x6b\x7b\xda"
        "\xc3\x89\x25\x69\xd4\xf1\xc4\x97",
        "password", 8
    },
};

const TestCase CASES_I_13[] = {
    {
        {
            32,
            "\x02\x02\x02\x02\x02\x02\x02\x02"
            "\x02\x02\x02\x02\x02\x02\x02\x02", 16,
            "\x03\x03\x03\x03\x03\x03\x03\x03", 8,
            "\x04\x04\x04\x04\x04\x04\x04\x04"
            "\x04\x04\x04\x04", 12,
            3, 32, 4
        },
        "\xc8\x14\xd9\xd1\xdc\x7f\x37\xaa"
        "\x13\xf0\xd7\x7f\x24\x94\xbd\xa1"
        "\xc8\xde\x6b\x01\x6d\xd3\x88\xd2"
        "\x99\x52\xa4\xc4\x67\x2b\x6c\xe8",
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01", 32
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xc1\x62\x88\x32\x14\x7d\x97\x20"
        "\xc5\xbd\x1c\xfd\x61\x36\x70\x78"
        "\x72\x9f\x6d\xfb\x6f\x8f\xea\x9f"
        "\xf9\x81\x58\xe0\xd7\x81\x6e\xd0",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 1
        },
        "\x89\xe9\x02\x9f\x46\x37\xb2\x95"
        "\xbe\xb0\x27\x05\x6a\x73\x36\xc4"
        "\x14\xfa\xdd\x43\xf6\xb2\x08\x64"
        "\x52\x81\xcb\x21\x4a\x56\x45\x2f",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 2
        },
        "\x4f\xf5\xce\x27\x69\xa1\xd7\xf4"
        "\xc8\xa4\x91\xdf\x09\xd4\x1a\x9f"
        "\xbe\x90\xe5\xeb\x02\x15\x5a\x13"
        "\xe4\xc0\x1e\x20\xcd\x4e\xab\x61",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            1, UINT32_C(1) << 16, 1
        },
        "\xd1\x68\x07\x5c\x4d\x98\x5e\x13"
        "\xeb\xea\xe5\x60\xcf\x8b\x94\xc3"
        "\xb5\xd8\xa1\x6c\x51\x91\x6b\x6f"
        "\x4a\xc2\xda\x3a\xc1\x1b\xbe\xcf",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            4, UINT32_C(1) << 16, 1
        },
        "\xaa\xa9\x53\xd5\x8a\xf3\x70\x6c"
        "\xe3\xdf\x1a\xef\xd4\xa6\x4a\x84"
        "\xe3\x1d\x7f\x54\x17\x52\x31\xf1"
        "\x28\x52\x59\xf8\x81\x74\xce\x5b",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\x14\xae\x8d\xa0\x1a\xfe\xa8\x70"
        "\x0c\x23\x58\xdc\xef\x7c\x53\x58"
        "\xd9\x02\x12\x82\xbd\x88\x66\x3a"
        "\x45\x62\xf5\x9f\xb7\x4d\x22\xee",
        "differentpassword", 17
    },
    {
        {
            32, "diffsalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xb0\x35\x7c\xcc\xfb\xef\x91\xf3"
        "\x86\x0b\x0d\xba\x44\x7b\x23\x48"
        "\xcb\xef\xec\xad\xaf\x99\x0a\xbf"
        "\xe9\xcc\x40\x72\x6c\x52\x12\x71",
        "password", 8
    },
};

const TestCase CASES_D_13[] = {
    {
        {
            32,
            "\x02\x02\x02\x02\x02\x02\x02\x02"
            "\x02\x02\x02\x02\x02\x02\x02\x02", 16,
            "\x03\x03\x03\x03\x03\x03\x03\x03", 8,
            "\x04\x04\x04\x04\x04\x04\x04\x04"
            "\x04\x04\x04\x04", 12,
            3, 32, 4
        },
        "\x51\x2b\x39\x1b\x6f\x11\x62\x97"
        "\x53\x71\xd3\x09\x19\x73\x42\x94"
        "\xf8\x68\xe3\xbe\x39\x84\xf3\xc1"
        "\xa1\x3a\x4d\xb9\xfa\xbe\x4a\xcb",
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01", 32
    },
};

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
            (global, device, ARGON2_D, ARGON2_VERSION_13,
             ARRAY_BEGIN(CASES_D_13), ARRAY_END(CASES_D_13));
}

#include "argon2-opencl/processingunit.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/cudaexception.h"

int main(void) {
    std::size_t failures = 0;

    std::cerr << "Running CUDA tests..." << std::endl;
    try {
        runAllTests<cuda::Device, cuda::GlobalContext, cuda::ProgramContext,
                cuda::ProcessingUnit>(failures);
    } catch (cuda::CudaException &err) {
        std::cerr << "CUDA ERROR: " << err.what() << std::endl;
        return 2;
    }

    std::cerr << "Running OpenCL tests..." << std::endl;
    try {
        runAllTests<opencl::Device, opencl::GlobalContext, opencl::ProgramContext,
                opencl::ProcessingUnit>(failures);
    } catch (cl::Error &err) {
        std::cerr << "OpenCL ERROR: " << err.err() << ": "
                  << err.what() << std::endl;
        return 2;
    }

    if (failures) {
        std::cerr << failures << " TESTS FAILED!" << std::endl;
        return 1;
    }
    return 0;
}

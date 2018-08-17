#include "argon2.h"

#include "testcase.h"
#include "testvectors.h"
#include "testparams.h"

#include <iostream>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include <stdio.h>
#include <chrono>
#include <ctime>



#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "argon2-gpu-common/argon2params.h"
#include "argon2-opencl/processingunit.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/cudaexception.h"

#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "global.cpp"

using namespace argon2;

argon2::Argon2Params myParam = { 32, "Satoshi_is_Finney", 17, nullptr, 0, nullptr, 0, 2, 256,  2 };

auto buffer = std::unique_ptr<std::uint8_t[]>(new std::uint8_t[32]);


//with batches
template<class Device, class GlobalContext, class ProgramContext, class ProcessingUnit>
std::size_t runParamsVsRef(ProcessingUnit & pu, const GlobalContext &global, const Device &device,
                           Type type, Version version, unsigned char * pwd, int pwdLen, unsigned char * difficulty, unsigned long start, unsigned long batch )
{

	std::size_t failures = 0;

	int index;

	for (unsigned long i = 0; i < batch; ++ i) {

		index = i + start;


		data[ i ][ pwdLen + 3 ] = index & 0xff;
		data[ i ][ pwdLen + 2 ] = index >> 8 & 0xff;
		data[ i ][ pwdLen + 1 ] = index >> 16 & 0xff;
		data[ i ][ pwdLen ] = index >> 24 & 0xff;

		pu.setPassword (i, data[ i ].data (), pwdLen + 4);

//		if (g_debug) {
//			for (std::size_t q = 0; q < pwdLen + 4; q ++) {
//				d2base ((int) data[ i ][ q ], 16);
//				std::cout << " ";
//			}
//
//			std::cout << "\n";
//		}


	}


	pu.beginProcessing ();
	pu.endProcessing ();


	for (std::size_t i = 0; i < batch; ++ i) {

		pu.getHash (i, buffer.get ());

		auto x = buffer.get ();


		if (g_debug) {

			for (std::size_t q = 0; q < 32; q ++) {
				d2base ((int) x[ q ], 16);
				std::cout << " ";
			}

			std::cout << "\n";
		}


		bool change = false;
		for (std::size_t q = 0; q < 32; ++ q) {
			if (bestHash[ q ] == x[ q ]) continue;
			else if (bestHash[ q ] < x[ q ]) break;
			else if (bestHash[ q ] > x[ q ]) {

				change = true;
				break;

			}
		}


		if (change) {

			for (auto q = 0; q < 32; q ++)
				bestHash[ q ] = x[ q ];

			bestHashNonce = start + i;


			for (std::size_t q = 0; q < 32; ++ q) {
				if (difficulty[ q ] == bestHash[ q ]) continue;
				else if (difficulty[ q ] < bestHash[ q ]) break;
				else if (difficulty[ q ] > bestHash[ q ]) {


					if (g_debug) {
						for (auto w = 0; w < 32; w ++) {
							std::cout << (int) difficulty[ w ];
						}

						std::cout << "SOLUTIE   " << q << (int) difficulty[ q ] << " " << (int) bestHash[ q ] << "\n";
					}

					return true;

				}
			}


		}

	}

    if (g_debug) {
        if (failures)
            std::cout << "  ERROR " << std::endl;
    }

	return false;
}

template<class Device, class GlobalContext, class ProgramContext, class ProcessingUnit>
std::size_t runTestCases(const GlobalContext &global, const Device &device,
                         Type type, Version version,
                         unsigned char * pwd, int length)
{

	int result, i, start, end;


	auto &params = myParam;

	std::size_t failures = 0;
	ProgramContext progCtx (&global, { device }, type, version);
	for (auto bySegment : { true, false }) {
		const std::array<bool, 2> precomputeOpts = { false, true };
		auto precBegin = precomputeOpts.begin ();
		auto precEnd = precomputeOpts.end ();
		if (type == ARGON2_D)
			precEnd --;

		for (auto precIt = precBegin; precIt != precEnd; precIt ++) {


			bool precompute = *precIt;


			auto buffer = std::unique_ptr<std::uint8_t[]> (
					new std::uint8_t[32]);

			ProcessingUnit pu (&progCtx, &params, &device, 1, bySegment, precompute);

			pu.setPassword (0, pwd, length);
			pu.beginProcessing ();
			pu.endProcessing ();
			pu.getHash (0, buffer.get ());


			if (1 == 1) {
				auto x = buffer.get ();
				for (auto q = 0; q < 32; q ++)
					d2base ((int) x[ q ], 16);
				std::cout << "\n";
			}


		}
	}
	if (failures)
		std::cout << "  ERROR " << std::endl;

	return failures;
}

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define ARRAY_BEGIN(a) (a)
#define ARRAY_END(a) ((a) + ARRAY_SIZE(a))




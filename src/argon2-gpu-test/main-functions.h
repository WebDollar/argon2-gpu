#include <unistd.h>
#include <iostream>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include <stdio.h>
#include <chrono>
#include <ctime>

#include "global.cpp"




template<class Device, class GlobalContext, class ProgramContext, class ProcessingUnit>
int initializeSystem ( const char *progname, const char *name, std::size_t deviceIndex, bool listDevices, std::size_t &failures, std:: string filename, unsigned long batch ){

    int i;

    GlobalContext global;
    auto &devices = global.getAllDevices();

    std::cout << "{ \"t\": \"devices\", \"devices\": " << devices.size() << " } \n " ;


    if (listDevices) {
        for (std::size_t i = 0; i < devices.size(); i++) {
            auto &device = devices[i];
            std::cout << "Device #" << i << ": " << device.getInfo()
                      << std::endl;
        }
        return 0;
    }

    if (deviceIndex >= devices.size()) {
        std::cout << "{ \"t\": \"error\", \"msg\": \"Device index out of range! "<<deviceIndex<<" \",  } \n" ;
        return 2;
    }

    auto &device = devices[deviceIndex];
    //std::cout << "Running " << name << " tests..." << std::endl;
    std::cout << "{ \"t\": \"device\", \"name\": " <<  device.getName() << " } \n";

    bool bySegment = false;
    bool precompute = false;

    int answer;


    ProgramContext progCtx(&global, { device }, ARGON2_D, argon2::ARGON2_VERSION_13 );
    ProcessingUnit pu(&progCtx, &myParam, &device, batch, bySegment, precompute);

    while ( 1 ){


        usleep(50);

        answer = readData( filename.c_str() );

        if (answer == -1) break;// finished
        else if (answer == 0) continue; //next time

        // Test chrono system_clock
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        for (int i=0; i < batch; i++){

            std::string input;

            for (int q = 0; q<length; q++)
                input.push_back(  pwd[q] );

            input.push_back(0);
            input.push_back(0);
            input.push_back(0);
            input.push_back(0);

            data[i] = input;
        }

        bool solution = false;

        for (i=0; i <32; i++)
            bestHash[i] = 255;

        i = start;
        while ( i < end ){

            solution = runParamsVsRef<Device, GlobalContext, ProgramContext, ProcessingUnit>
                    (pu, global, device, ARGON2_D, argon2::ARGON2_VERSION_13, pwd, length,  difficulty, i, ( end - i >= batch ) ? batch : (end-i));

            if (solution)
                break;

            i += batch;

        }

        std::string hash;


        for (auto q=0; q < 32; q++){

            hash.push_back(   arr[ bestHash[q]/16 ] );
            hash.push_back(   arr[ bestHash[q]%16 ]);

        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);

        int hs = print_timediff(t_start, t_end, end-start);

/*
        if (solution) std::cout << "{ \"type\": \"s\", \"hash\": \""<< hash <<"\", \"nonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";
        else std::cout << "{ \"type\": \"b\", \"bestHash\": \""<< hash <<"\", \"bestNonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";
*/

        std::ofstream ofs (filename+"output", std::ofstream::out);
        if (solution) ofs << "{ \"type\": \"s\", \"hash\": \""<< hash <<"\", \"nonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";
        else  ofs << "{ \"type\": \"b\", \"bestHash\": \""<< hash <<"\", \"bestNonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";
        ofs.close();

    }


    return 0;
}


/***
 * OLD VERSIONS
 */


template<class Device, class GlobalContext, class ProgramContext, class ProcessingUnit>
int runRealTime(const char *progname, const char *name, std::size_t deviceIndex, bool listDevices, std::size_t &failures){
    
    int i;
    
    GlobalContext global;
    auto &devices = global.getAllDevices();

    std::cout << "{ \"t\": \"devices\", \"devices\": " << devices.size() << " } \n " ;

    if (listDevices) {
        for (std::size_t i = 0; i < devices.size(); i++) {
            auto &device = devices[i];
            std::cout << "Device #" << i << ": " << device.getInfo()
                      << std::endl;
        }
        return 0;
    }

    if (deviceIndex >= devices.size()) {
         std::cout << "{ \"t\": \"error\", \"msg\": \"Device index out of range! "<<deviceIndex<<" \",  } \n" ;
        return 2;
    }

    auto &device = devices[deviceIndex];
    std::cout << "{ \"t\": \"device\", \"name\": " <<  device.getName() << " } \n";


    usleep(2000);

    while ( 1  ){

        usleep (10);

        scanf ("%d", &length);
        scanf ("%s", &pwd);
        scanf ("%s", &difficulty);
        scanf ("%d", &start);
        scanf ("%d", &batch);

        std::cout << "{ 't': 'msg', 'length': " << length << ", 'start': " << start << ", 'batch':" << batch << " } \n";


        std::cout << "length: " << length << "\n";

        for (i = 0; i < 32; i ++)
            bestHash[i] = 255;

        i = start;


        // Test chrono system_clock
        clock_gettime (CLOCK_MONOTONIC, &t_start);

        for (int i = 0; i < batch; i ++) {

            std::string input;

            for (int q = 0; q < length; q ++)
                input.push_back (pwd[q]);

            input.push_back (0);
            input.push_back (0);
            input.push_back (0);
            input.push_back (0);

            data[i] = input;
        }

        bool bySegment = false;
        bool precompute = false;

        ProgramContext progCtx(&global, { device }, ARGON2_D, argon2::ARGON2_VERSION_13 );
        ProcessingUnit pu(&progCtx, &myParam, &device, batch, bySegment, precompute);


        failures += runParamsVsRef<Device, GlobalContext, ProgramContext, ProcessingUnit>
                 (pu, global, device, ARGON2_D, argon2::ARGON2_VERSION_13, pwd, length,  difficulty, i, batch);

        std::cout << "DONE\n";
        for (auto q=0; q < 32; q++){
            d2base( bestHash[q], 16);
                std::cout << " ";
	    }

        std::cout << "nonce    " << bestHashNonce << "\n";


        std::cout << "done\n \n \n \n";

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        print_timediff(t_start, t_end, end-start);

    }
		
    return 0;
}



template<class Device, class GlobalContext, class ProgramContext, class ProcessingUnit>
int runAllTests(const char *progname, const char *name, std::size_t deviceIndex, bool listDevices, std::size_t &failures, std::string filename){
    
    int i;
    
    GlobalContext global;
    auto &devices = global.getAllDevices();
    

    std::cout << "{ \"t\": \"devices\", \"devices\": " << devices.size() << " } \n " ;

    if (listDevices) {
        for (std::size_t i = 0; i < devices.size(); i++) {
            auto &device = devices[i];
            std::cout << "Device #" << i << ": " << device.getInfo()
                      << std::endl;
        }
        return 0;
    }

    if (deviceIndex >= devices.size()) {
         std::cout << "{ \"t\": \"error\", \"msg\": \"Device index out of range! "<<deviceIndex<<" \",  } \n" ;
        return 2;
    }

    auto &device = devices[deviceIndex];
    //std::cout << "Running " << name << " tests..." << std::endl;
    std::cout << "{ \"t\": \"device\", \"name\": " <<  device.getName() << " } \n";

    std::cout << "{ \"t\": \"filename\", \"name\": " <<  filename << " } \n";

    readData(filename.c_str());


    for (i=0; i <32; i++)
	bestHash[i] = 255;

    i = start;

    // Test chrono system_clock
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int i=0; i < batch; i++){
	   
    	std::string input; 
    
    	for (int q = 0; q<length; q++)
    	   input.push_back(  pwd[q] );
    
    	input.push_back(0);	
    	input.push_back(0);	
    	input.push_back(0);	
    	input.push_back(0);	
    
    	data[i] = input;	
	
    }
        

    bool bySegment = false;
    bool precompute = false;

    ProgramContext progCtx(&global, { device }, ARGON2_D, argon2::ARGON2_VERSION_13 );
    ProcessingUnit pu(&progCtx, &myParam, &device, batch, bySegment, precompute);

    
    bool solution = false;
    while ( i < end ){

//	    failures += runTestCases<Device, GlobalContext, ProgramContext, ProcessingUnit>
//        	    (global, device, ARGON2_D, argon2::ARGON2_VERSION_13,
//        	     pwd, length);

    	solution = runParamsVsRef<Device, GlobalContext, ProgramContext, ProcessingUnit>
	      (pu, global, device, ARGON2_D, argon2::ARGON2_VERSION_13, pwd, length,  difficulty, i, ( end - i >= batch ) ? batch : (end-i));

    	if (solution)
    		break;
    
    	i += batch;

    }

    std::string hash;


    for (auto q=0; q < 32; q++){
	
         hash.push_back(   arr[ bestHash[q]/16] );
         hash.push_back(   arr [ bestHash[q]%16 ]);

    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
  
    int hs = print_timediff(t_start, t_end, end-start);  

    if (solution)
	    std::cout << "{ \"type\": \"s\", \"hash\": \""<< hash <<"\", \"nonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";
    else 
	    std::cout << "{ \"type\": \"b\", \"bestHash\": \""<< hash <<"\", \"bestNonce\": " << bestHashNonce << " , \"h\": "<< hs <<" }";

  

    return 0;
}







#include <uWS/uWS.h>
#include <uWS/WebSocket.h>
#include <uWS/Node.h>

#include "global.cpp"

using namespace uWS;

void openServer(int port = 3000){

    Hub h;
    std::string response = "Hello!";

    h.onMessage([](WebSocket<SERVER> *ws, char *message, size_t length, OpCode opCode) {

	int index = 0;
	length = message[index]*255*255*255 + message[index+1]*255*255 + message[index+2]*255 + message[index+3];
	index += 4;
	 
	//std::cout << "hash:   " << " \n";   
	int a;
	for ( i=0; i<length ; i++){
	   pwd[i] = message[index];
	   index++;
	}

	//std::cout << "DIFFICULTY:   " << " \n";   
	for ( i=0; i<32; i++){

 	    difficulty[i] = message[index];
	    index++;
        }

	start = message[index]*255*255*255 + message[index+1]*255*255 + message[index+2]*255 + message[index+3];
	index += 4;
	end = message[index]*255*255*255 + message[index+1]*255*255 + message[index+2]*255 + message[index+3];
	index += 4;
	batch = message[index]*255*255*255 + message[index+1]*255*255 + message[index+2]*255 + message[index+3];
	index += 4;


   	for (i=0; i <32; i++)
	   bestHash[i] = 255;


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
        

        //ws->send(message, length, opCode);


    });

    h.onHttpRequest([&](HttpResponse *res, HttpRequest req, char *data, size_t length, size_t remainingBytes) {

        res->end(response.data(), response.length());

    });

    if (h.listen(port)) {
        h.run();
    }

}

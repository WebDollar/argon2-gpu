#ifndef GLOBAL2_H
#define GLOBAL2_H

#include <fstream>

char arr[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

 void d2base(int n, int b){

    std::cout << arr[n/b];   
    std::cout << arr[n%b];   
}


bool g_debug = false;

std::string data[1000000];

long unsigned g_id = 0;


long unsigned startPrev = 0, start = 0, length = 0, end = 0, batch;
unsigned char pwd[1024*1027], difficulty[34];
unsigned char _pwd[1024*1027],  _difficulty[34];

unsigned char bestHash[34];
long unsigned bestHashNonce = 0;
long unsigned hashesTotal = 0;

char * filename;
char filenameOutput[50] ;

int  print_timediff(const struct timespec& start, const  struct timespec& end, int count){

    double milliseconds = (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3;

    int hs = (int) (count / milliseconds *1000);

    printf(" %lf milliseconds \n", milliseconds);
    printf("%d h/s \n", hs);


    return hs;
}

struct timespec t_start, t_end;



 unsigned char encode(char x) {     /* Function to encode a hex character */
/****************************************************************************
 * these offsets should all be decimal ..x validated for hex..              *
 ****************************************************************************/
    if (x >= '0' && x <= '9')         /* 0-9 is offset by hex 30 */
        return (x - 0x30);
    else if (x >= 'a' && x <= 'f')    /* a-f offset by hex 57 */
        return(x - 0x57);
    else if (x >= 'A' && x <= 'F')    /* A-F offset by hex 37 */
        return(x - 0x37);
}

void print(char * data ){

}


bool fileExists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else
        return false;
    
}






int readData(const char * filename){

    int i, ok;
    long unsigned _start, _length, security;


    if (fileExists(filename) == 0) return 0;

    FILE *fin = fopen(filename, "rb");

    fscanf(fin, "%lu", &_start);
    fscanf(fin, "%lu", &_length);

    if (_start == 0 && _length == 0) {
        fclose(fin);
        return -1;
    }


    //printf("hash:   \n");
    for (i = 0; i < _length; i++) {

        fscanf(fin, "%hhu", &_pwd[i] );

        if (feof(fin)) {
            fclose(fin);
            return 0;
        }
        //std::cout << a << " ";
    }

    //printf("DIFFICULTY:   \n");
    for (i = 0; i < 32; i++) {

        fscanf(fin, "%hhu", &_difficulty[i]);

        if (feof(fin)) {
            fclose(fin);
            return 0;
        }
        //std::cout << a << " ";
    }

    //check if it is identical
    if (_start == startPrev && _length == length) {
        for (i=0; i < _length; i++)
            if (pwd[i] != _pwd[i]){
                ok = 0;
                break;
            }
        for (i=0; i < 32; i++)
            if ( difficulty[i] != _difficulty[i]){
                ok = 0;
                break;
            }
        if (ok == 1){
            fclose(fin);
            return 0;
        }
    }
    for (i=0; i < _length; i++)
        pwd[i] = _pwd[i];

    for (i=0; i < 32; i++)
        difficulty[i] = _difficulty[i];


    //fin >> pwdHex;
    //fin >> difficultyHex;
    fscanf(fin, "%lu", &end);
    fscanf(fin, "%lu", &batch);

    fscanf(fin, "%lu", &security);

    //std::cout  << " cool " << end << " " << batch << " " << security << "\n";

    if (security != 218391) {
        fclose(fin);
        return 0;
    }

    start = _start;
    startPrev = _start;
    length = _length;

    for (i = 0; i < 32; i++)
        bestHash[i] = 255;

    bestHashNonce = 0;
    hashesTotal = 0;

    g_id++;

    fclose(fin);

    if (g_debug)
        printf("DATA READ!!! %lu %lu %lu \n", length, start, end);


    return 1;

}


#endif




#ifndef GLOBAL_H
#define GLOBAL_H

// header contents


extern void d2base(int n, int b);

extern unsigned char  pwd[1024*1025], difficulty[32+1];

extern unsigned char  bestHash[34];
extern unsigned long bestHashNonce = 0;

extern std::string data[1000000];

extern void print_timediff(const struct timespec& start, const  struct timespec& end, int count);

extern struct timespec t_start, t_end;

extern unsigned char encode(char x);

extern void print(char * data );

int readData(char * filename);



#endif




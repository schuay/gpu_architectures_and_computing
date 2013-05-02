#ifndef __UTIL_H
#define __UTIL_H

#include "sigpt.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

float *random_array(int seed, int length);

int read_signal_file(const char* filename, sigpt_t** signal);
int write_signal_file(const char* filename, const sigpt_t* signal, int n);

#endif /* __UTIL_H */

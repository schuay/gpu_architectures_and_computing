#ifndef __UTIL_H
#define __UTIL_H

#include "sigpt.h"


float *random_array(int seed, int length);

sigpt_t *read_signal_file(const char* filename);
int write_signal_file(const char* filename, const sigpt_t* signal, int n);



#endif /* __UTIL_H */

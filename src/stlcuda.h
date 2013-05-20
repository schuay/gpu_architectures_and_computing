#ifndef __STLCUDA_H
#define __STLCUDA_H

#include "sigpt.h"


typedef struct {
    float *y;
    float *t;
    float *dy;
} sigarray_t;

//int stl_and(sigpt_t *sig1, int sig1_n, sigpt_t *sig2, int sig2_n, sigpt_t *out, int *out_n);
/*int stlcuda_and(float *sig1, float *sig1_time, int sig1_n,
                float *sig2, float *sig2_time, int sig2_n,
                float *out, float *out_time, int *out_n);
*/
int stlcuda_and(sigarray_t sig1, int sig1_n, 
                sigarray_t sig2, int sig2_n,
                sigarray_t *out, int *out_n);

int lib_test(int a, int b, sigarray_t sig);

#endif


extern "C" {
#include "stlcuda.h"
}

#include "operators/and.hpp"


//int stl_and(sigpt_t *sig1, int sig1_n, sigpt_t *sig2, int sig2_n, sigpt_t *out, int *out_n) {
/*int stlcuda_and(float *sig1, float *sig1_time, int sig1_n,
                float *sig2, float *sig2_time, int sig2_n,
                float *out, float *out_time, int *out_n) {
*/
int stlcuda_and(sigarray_t sig1, int sig1_n,
                sigarray_t sig2, int sig2_n,
                sigarray_t *out, int *out_n) {

    sigpt_t *a, *b;
    int i;

    a = (sigpt_t*) malloc(sig1_n * sizeof(sigpt_t));
    b = (sigpt_t*) malloc(sig2_n * sizeof(sigpt_t));

    for (i = 0; i < sig1_n; i++) {
        a[i].y = sig1.y[i];
        a[i].t = sig1.t[i];
        a[i].dy = 0;
    }

    for (i = 0; i < sig2_n; i++) {
        b[i].y = sig2.y[i];
        b[i].t = sig2.t[i];
        b[i].dy = 0;
    }

    thrust::device_vector<sigpt_t> lhs(a, a + sig1_n);
    thrust::device_vector<sigpt_t> rhs(b, b + sig2_n);

    thrust::device_ptr<sigpt_t> device_out;
    int nout;

    stl_and(&lhs[0], lhs.size(), &rhs[0], rhs.size(), &device_out, &nout);

    thrust::host_vector<sigpt_t> host_out(device_out, device_out + nout);

    out->y = (float*) malloc(nout * sizeof(float));
    out->t = (float*) malloc(nout * sizeof(float));

    for (i = 0; i < nout; i++) {
        out->y[i] = host_out[i].y;
        out->t[i] = host_out[i].t;
    }
    *out_n = nout;

    return nout; 
//    return 0;
/*
    out->y = sig1.y;
    out->t = sig2.t;
    *out_n = sig1_n + sig2_n;
    return sig1_n + sig2_n;
*/
}


extern int lib_test(int a, int b, sigarray_t sig) {
    return a + b;
}


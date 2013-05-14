#include "not.hpp"

#include "globals.h"

/**
 * sizeof(out) == sizeof(in).
 */
__global__ void
stl_not(const sigpt_t *in, sigpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sigpt_t s = in[i];
        s.y *= -1.f;
        s.dy *= -1.f;
        out[i] = s;
    }
}

void
stl_not(const thrust::device_ptr<sigpt_t> &in,
        const int nin,
        thrust::device_ptr<sigpt_t> *out,
        int *nout)
{
    thrust::device_ptr<sigpt_t> ptr_out = thrust::device_malloc<sigpt_t>(nin);

    stl_not<<<NBLOCKS, NTHREADS>>>(in.get(), ptr_out.get(), nin);

    *out = ptr_out;
    *nout = nin;
}

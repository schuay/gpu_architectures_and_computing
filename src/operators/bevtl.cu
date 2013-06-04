#include "bevtl.hpp"

#include <assert.h>

#include "globals.h"

/**
 * Computes the running maximum of the signal y
 * with the given window size.
 */
__global__ static void
bevtl_rmax(const sigpt_t *y,
           const int window,
           float *maxs,
           const int n);

#include <stdio.h> // TODO Remove me.

void
stl_bevtl(const thrust::device_ptr<sigpt_t> &in,
          const int nin,
          const float s,
          const float t,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    assert(s <= t);

    thrust::device_ptr<float> maxs = thrust::device_malloc<float>(nin);
    bevtl_rmax<<<NBLOCKS, NTHREADS>>>(in.get(), t - s + 1, maxs.get(), nin);

    thrust::host_vector<sigpt_t> h(in, in + nin);
    /* TODO: Remove me.
    sigpt_print("in", h.data(), nin);

    printf("maxs (s: %d, t: %d)\n", s, t);
    for (int i = 0; i < nin; i++) {
        const float f = maxs[i];
        printf("%d: %f\n", i, f);
    }
    */
}

__global__ static void
bevtl_rmax(const sigpt_t *y,
           const int window,
           float *maxs,
           const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* TODO: Optimization potential: Cache points in __shared__ memory,
     * and use a logarithmic MAX scheme:
     * this.max = MAX(this.y, this+1.y);
     * this.max = MAX(this.max, this+2.max
     * this.max = MAX(this.max, this+4.max
     *
     * A problem of this running max implementation is that it's dependent on
     * the size of the window w: O(w * n / p) whereas the algorithm used in
     * the paper is linear in the length of the signal.
     *
     * It also doesn't add new time points on intersections with 0 and
     * any other magic that might be required of evtl.
     */

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float max = y[i].y;

        for (int off = 1; off < window; off++) {
            if (i + off >= n) {
                continue;
            }

            const sigpt_t other = y[i + off];
            max = CUDA_MAX(max, other.y);
        }

        maxs[i] = max;
    }
}

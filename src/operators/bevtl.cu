#include "bevtl.hpp"

#include <assert.h>

#include "globals.h"

#include <thrust/merge.h>
#include <thrust/scan.h>

typedef struct {
    int i;
    float t;
    int count;
} idx_time;

__global__ static void
to_idx_time(const sigpt_t *y,
            const int n,
            idx_time *out,
            const int ofs,
            const int count);

__global__ static void
extract_count(const idx_time *in,
              const int n,
              int *out,
              int *outi);

__global__ static void
compact(const int *xs,
        const int n,
        const int *cs,
        const int *fs,
        int *out);

__global__ static void
calc_max_win_size(const int *sofs,
                  const int *tofs,
                  const int n,
                  int *wsizes);

struct idx_time_less : public thrust::binary_function<idx_time, idx_time, bool>
{
    __device__ bool
    operator()(const idx_time &lhs, const idx_time &rhs) const
    {
        return lhs.t < rhs.t;
    }
};

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

static void
time_to_idx_ofs(const thrust::device_ptr<sigpt_t> &in,
                const int nin,
                const float ofs,
                thrust::device_ptr<int> *out);

void
stl_bevtl(const thrust::device_ptr<sigpt_t> &in,
          const int nin,
          const float s,
          const float t,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    assert(s <= t);

    /* TODO: insert intersections at start and then consolidate with input. */

    /* Get start and end indices of the windows. */
    thrust::device_ptr<int> sofs = thrust::device_malloc<int>(nin);
    time_to_idx_ofs(in, nin, s, &sofs);
    thrust::device_ptr<int> tofs = thrust::device_malloc<int>(nin);
    time_to_idx_ofs(in, nin, t, &tofs);

    thrust::host_vector<sigpt_t> h(in, in + nin);
    sigpt_print("in", h.data(), nin);
    thrust::host_vector<int> hs(sofs, sofs + nin);
    thrust::host_vector<int> ht(tofs, tofs + nin);
    for (int i = 0; i < nin; i++) {
        printf("i: %d, t: %f, t+s: %f, i+s: %d\n", i, h[i].t, h[i].t + s, hs[i]);
        printf("i: %d, t: %f, t+t: %f, i+t: %d\n", i, h[i].t, h[i].t + t, ht[i]);
    }

    /* Calculate the window size for each element. */
    thrust::device_ptr<int> wsizes = thrust::device_malloc<int>(nin);
    calc_max_win_size<<<NBLOCKS, NTHREADS>>>(sofs.get(), tofs.get(), nin, wsizes.get());

    /* Calculate the position in the result array where each window will reside. */
    thrust::exclusive_scan(wsizes, wsizes + nin, wsizes, 0, thrust::plus<int>()); 

    /* TODO: calculate the evtl. */

    //thrust::device_ptr<float> maxs = thrust::device_malloc<float>(nin);
    //bevtl_rmax<<<NBLOCKS, NTHREADS>>>(in.get(), t - s + 1, maxs.get(), nin);

    /* TODO: Remove me.
    sigpt_print("in", h.data(), nin);

    printf("maxs (s: %d, t: %d)\n", s, t);
    for (int i = 0; i < nin; i++) {
        const float f = maxs[i];
        printf("%d: %f\n", i, f);
    }
    */

    thrust::device_free(sofs);
    thrust::device_free(tofs);
    thrust::device_free(wsizes);
}

static void
time_to_idx_ofs(const thrust::device_ptr<sigpt_t> &in,
                const int nin,
                const float ofs,
                thrust::device_ptr<int> *out)
{
    thrust::device_ptr<idx_time> nt = thrust::device_malloc<idx_time>(nin);
    to_idx_time<<<NBLOCKS, NTHREADS>>>(in.get(), nin, nt.get(), 0, 1);

    thrust::device_ptr<idx_time> st = thrust::device_malloc<idx_time>(nin);
    to_idx_time<<<NBLOCKS, NTHREADS>>>(in.get(), nin, st.get(), ofs, 0);

    int sntn = nin * 2;
    thrust::device_ptr<idx_time> snt = thrust::device_malloc<idx_time>(sntn);
    thrust::merge(nt, nt + nin, st, st + nin, snt, idx_time_less());

    thrust::device_free(nt);
    thrust::device_free(st);

    thrust::device_ptr<int> c = thrust::device_malloc<int>(sntn);
    thrust::device_ptr<int> ci = thrust::device_malloc<int>(sntn);

    extract_count<<<NBLOCKS, NTHREADS>>>(snt.get(), sntn, c.get(), ci.get());
    thrust::device_free(snt);

    thrust::exclusive_scan(c, c + sntn, c, 0, thrust::plus<int>()); 

    thrust::device_ptr<int> fs = thrust::device_malloc<int>(sntn);
    thrust::exclusive_scan(ci, ci + sntn, fs, 0, thrust::plus<int>()); 

    thrust::device_ptr<int> sofs = thrust::device_malloc<int>(nin);
    compact<<<NBLOCKS, NTHREADS>>>(c.get(), sntn, ci.get(), fs.get(), sofs.get());

    thrust::device_free(c);
    thrust::device_free(ci);
    thrust::device_free(fs);

    *out = sofs;
}

__global__ static void
calc_max_win_size(const int *sofs,
                  const int *tofs,
                  const int n,
                  int *wsizes)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        wsizes[i] = (tofs[i] - sofs[i]);
    }
}

__global__ static void
compact(const int *xs,
        const int n,
        const int *cs,
        const int *fs,
        int *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (cs[i] == 1) {
            out[fs[i]] = xs[i];
        }
    }
}

__global__ static void
to_idx_time(const sigpt_t *y,
            const int n,
            idx_time *out,
            const int ofs,
            const int count)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        idx_time it = { i, y[i].t + ofs, count };
        out[i] = it;
    }
}

__global__ static void
extract_count(const idx_time *in,
              const int n,
              int *out,
              int *outi)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i] = in[i].count;
        outi[i] = ! in[i].count;
    }
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

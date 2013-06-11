#include "bevtl.hpp"

#include <assert.h>

#include "globals.h"

#include "interpolate.hpp"

#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

typedef struct {
    int i;
    float t;
    int count;
} idx_time;

__global__ static void
to_idx_time(const sigpt_t *y,
            const int n,
            idx_time *out,
            const float ofs,
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

struct idx_time_less : public thrust::binary_function<idx_time, idx_time, bool>
{
    __device__ bool
    operator()(const idx_time &lhs, const idx_time &rhs) const
    {
        return lhs.t < rhs.t;
    }
};

struct sigpt_t_less : public thrust::binary_function<sigpt_t, sigpt_t, bool>
{
    __device__ bool
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        return lhs.t < rhs.t;
    }
};

struct sigpt_time_equals : public thrust::binary_function<sigpt_t, sigpt_t, bool>
{
    __device__ bool
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        return fabsf(lhs.t - rhs.t) < FLOAT_DELTA;
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
                const thrust::device_ptr<sigpt_t> &in_ofs,
                const int nin_ofs,
                const float ofs,
                thrust::device_ptr<int> *out);

__global__ static void
calc_ofs_points(const sigpt_t *in,
		const int *ofs_idx,
                const int n,
                const float ofs,
		sigpt_t *out);

__global__ static void
calc_win_max(const sigpt_t *in,
             const int *sofs,
             const int *tofs,
             const int n,
             const sigpt_t *all,
	     sigpt_t *out);

void
stl_bevtl(const thrust::device_ptr<sigpt_t> &in,
          const int nin,
          const float s,
          const float t,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    assert(s <= t);

    /* Get start and end indices of the windows. */
    thrust::device_ptr<int> sofs_no_endp;
    time_to_idx_ofs(in, nin, in, nin, s, &sofs_no_endp);
    thrust::device_ptr<int> tofs_no_endp;
    time_to_idx_ofs(in, nin, in, nin, t, &tofs_no_endp);

    /*thrust::host_vector<sigpt_t> h(in, in + nin);
    sigpt_print("in", h.data(), nin);
    thrust::host_vector<int> hs(sofs_no_endp, sofs_no_endp + nin);
    thrust::host_vector<int> ht(tofs_no_endp, tofs_no_endp + nin);
    for (int i = 0; i < nin; i++) {
        printf("i: %d, t: %f, t+s: %f, i+s: %d\n", i, h[i].t, h[i].t + s, hs[i]);
        printf("i: %d, t: %f, t+t: %f, i+t: %d\n", i, h[i].t, h[i].t + t, ht[i]);
    }*/

    /* Calculate start and end points. */
    thrust::device_ptr<sigpt_t> sofs_points = thrust::device_malloc<sigpt_t>(nin);
    calc_ofs_points<<<NBLOCKS, NTHREADS>>>(in.get(), sofs_no_endp.get(), nin, s, sofs_points.get());
    thrust::device_ptr<sigpt_t> tofs_points = thrust::device_malloc<sigpt_t>(nin);
    calc_ofs_points<<<NBLOCKS, NTHREADS>>>(in.get(), tofs_no_endp.get(), nin, t, tofs_points.get());

    /*thrust::host_vector<sigpt_t> hs(sofs_points, sofs_points + nin);
    thrust::host_vector<sigpt_t> ht(tofs_points, tofs_points + nin);
    sigpt_print("sofs", hs.data(), nin);
    sigpt_print("tofs", ht.data(), nin);*/

    thrust::device_free(sofs_no_endp);
    thrust::device_free(tofs_no_endp);

    /* Merge. */
    thrust::device_ptr<sigpt_t> stofs_points = thrust::device_malloc<sigpt_t>(2 * nin);
    thrust::merge(sofs_points, sofs_points + nin, tofs_points, tofs_points + nin, stofs_points, sigpt_t_less());

    thrust::device_free(sofs_points);
    thrust::device_free(tofs_points);

    int stofs_n = thrust::unique(stofs_points, stofs_points + 2 * nin, sigpt_time_equals()) - stofs_points;

    thrust::device_ptr<sigpt_t> all_points = thrust::device_malloc<sigpt_t>(3 * nin);
    thrust::merge(in, in + nin, stofs_points, stofs_points + stofs_n, all_points, sigpt_t_less());

    thrust::device_free(stofs_points);

    /* Remove duplicate points. */
    int all_n = thrust::unique(all_points, all_points + 3 * nin, sigpt_time_equals()) - all_points;

    thrust::host_vector<sigpt_t> a(all_points, all_points + all_n);
    sigpt_print("all", a.data(), all_n);

    /* Get start and end indices of the windows, this time including intersections. */
    thrust::device_ptr<int> sofs;
    time_to_idx_ofs(all_points, all_n, in, nin, s, &sofs);
    thrust::device_ptr<int> tofs;
    time_to_idx_ofs(all_points, all_n, in, nin, t, &tofs);

    /*thrust::host_vector<sigpt_t> h(in, in + nin);
    sigpt_print("in", h.data(), nin);
    thrust::host_vector<sigpt_t> h2(all_points, all_points + all_n);
    sigpt_print("all_points", h2.data(), all_n);
    thrust::host_vector<int> hs(sofs, sofs + nin);
    thrust::host_vector<int> ht(tofs, tofs + nin);
    for (int i = 0; i < nin; i++) {
        printf("i: %d, t: %f, t+s: %f, i+s: %d\n", i, h[i].t, h[i].t + s, hs[i]);
        printf("i: %d, t: %f, t+t: %f, i+t: %d\n", i, h[i].t, h[i].t + t, ht[i]);
    }*/

    thrust::device_ptr<sigpt_t> final = thrust::device_malloc<sigpt_t>(nin);

    calc_win_max<<<NBLOCKS, NTHREADS>>>(in.get(), sofs.get(), tofs.get(), nin, all_points.get(), final.get());

    thrust::host_vector<sigpt_t> hf(final, final + nin);
    sigpt_print("out", hf.data(), nin);

    *out = final;
    *nout = nin;

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
    thrust::device_free(all_points);
}

__global__ static void
calc_win_max(const sigpt_t *in,
             const int *sofs,
             const int *tofs,
             const int n,
             const sigpt_t *all,
	     sigpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float max = -INFINITY;
        for (int j = sofs[i] - 1; j < tofs[i]; j++) {
            if (all[j].y > max) {
                max = all[j].y;
            }
        }

        sigpt_t tmp = { in[i].t, max, 0 };
        out[i] = tmp;
    }
}

__global__ static void
calc_ofs_points(const sigpt_t *in,
		const int *ofs_idx,
                const int n,
                const float ofs,
		sigpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float t = in[i].t + ofs;
        if ((ofs_idx[i]) >= n || t > in[n - 1].t) {
            sigpt_t tmp = { t, -INFINITY, 0};
            out[i] = tmp;
        } else {
            const sigpt_t *l = &in[ofs_idx[i] - 1];
            const sigpt_t *r = &in[ofs_idx[i]];
            out[i] = interpolate(l, r, t);
        }
    }
}

static void
time_to_idx_ofs(const thrust::device_ptr<sigpt_t> &in,
                const int nin,
                const thrust::device_ptr<sigpt_t> &in_ofs,
                const int nin_ofs,
                const float ofs,
                thrust::device_ptr<int> *out)
{
    /* Create time to index mappings for regular and offset points and merge them. */

    thrust::device_ptr<idx_time> nt = thrust::device_malloc<idx_time>(nin);
    to_idx_time<<<NBLOCKS, NTHREADS>>>(in.get(), nin, nt.get(), 0, 1);

    thrust::device_ptr<idx_time> st = thrust::device_malloc<idx_time>(nin_ofs);
    to_idx_time<<<NBLOCKS, NTHREADS>>>(in_ofs.get(), nin_ofs, st.get(), ofs, 0);

    int sntn = nin + nin_ofs;
    thrust::device_ptr<idx_time> snt = thrust::device_malloc<idx_time>(sntn);
    thrust::merge(nt, nt + nin, st, st + nin_ofs, snt, idx_time_less());

    thrust::device_free(nt);
    thrust::device_free(st);

    /* Calculate the first real point before each offset point (in c). */

    thrust::device_ptr<int> c = thrust::device_malloc<int>(sntn);
    thrust::device_ptr<int> ci = thrust::device_malloc<int>(sntn);

    extract_count<<<NBLOCKS, NTHREADS>>>(snt.get(), sntn, c.get(), ci.get());
    thrust::device_free(snt);

    thrust::exclusive_scan(c, c + sntn, c, 0, thrust::plus<int>()); 

    /* Calculate final positions and compact index offset array. */

    thrust::device_ptr<int> fs = thrust::device_malloc<int>(sntn);
    thrust::exclusive_scan(ci, ci + sntn, fs, 0, thrust::plus<int>()); 

    thrust::device_ptr<int> sofs = thrust::device_malloc<int>(nin_ofs);
    compact<<<NBLOCKS, NTHREADS>>>(c.get(), sntn, ci.get(), fs.get(), sofs.get());

    thrust::device_free(c);
    thrust::device_free(ci);
    thrust::device_free(fs);

    *out = sofs;
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
            const float ofs,
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

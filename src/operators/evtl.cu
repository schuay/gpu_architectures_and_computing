#include "evtl.hpp"

#include <thrust/scan.h>

#include "globals.h"

#define CUDA_MAX(a, b) (((a) > (b)) * (a) + ((a) <= (b)) * (b))

struct sigpt_max : public thrust::binary_function<sigpt_t, sigpt_t, sigpt_t>
{
    __device__ sigpt_t
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        return (sigpt_t) {rhs.t, CUDA_MAX(lhs.y, rhs.y), 0};
    }
};

/* Stores all elements zs[i] at zs_intersect[i * 2] and uses zs and ys to
 * interpolate points at zs_intersect[i * 2 + 1] if needed.
 * For every position in zs_intersect that contains a valid point the
 * according position in cs is set to 1.
 * ys: the original signal.
 * zs: the signal after the initial scan.
 */
__global__ static void
evtl_intersect(const sigpt_t *ys,
               sigpt_t *zs,
               sigpt_t *zs_intersect,
               char *cs,
               const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        cs[i * 2] = 1;
        zs_intersect[i * 2] = zs[i];
        // FIXME: Branches are bad.
        if (i < n - 1 && zs[i].y > zs[i + 1].y) {
            cs[i * 2 + 1] = 1;
            zs_intersect[i * 2 + 1].t = zs[i + 1].t +
                (zs[i + 1].t - zs[i].t) *
                (zs[i + 1].y - ys[i + 1].y) / (ys[i + 1].y - ys[i].y);
            zs_intersect[i * 2 + 1].y = zs[i + 1].y;
        }
    }
}

/* Store each element zs[i] where cs[i] is 1 in zs_final[fs[i]]. */
__global__ static void
evtl_compact(const sigpt_t *zs,
             sigpt_t *zs_final,
             const char *cs,
             const size_t *fs,
             const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // FIXME: Branches are bad.
        if (cs[i] == 1) {
            zs_final[fs[i]] = zs[i];
        }
    }
}

void
stl_evtl(const thrust::device_ptr<sigpt_t> &in,
         const int nin,
         thrust::device_ptr<sigpt_t> *out,
         int *nout)
{
    const int dnout = 2 * nin;
    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(dnout);

    thrust::reverse_iterator<thrust::device_ptr<sigpt_t> > rin(in + nin);
    thrust::reverse_iterator<thrust::device_ptr<sigpt_t> > rout(dout + dnout);

    /* Propagate max rubustness backwards */
    thrust::inclusive_scan(rin, rin + nin, rout + nin, sigpt_max()); 

    const sigpt_t *ys = in.get();

    thrust::device_vector<sigpt_t> out_intersect(dnout);
    sigpt_t *zs = thrust::raw_pointer_cast(out_intersect.data());

    sigpt_t *zs_final = dout.get();

    /* Used positions in zs (out_intersect) are marked here.
     * This amounts to all positions where there was an original value plus
     * those where we had to interpolate a value. */
    thrust::device_vector<char> used(dnout, 0);
    char *cs = thrust::raw_pointer_cast(used.data());
    evtl_intersect<<<NBLOCKS, NTHREADS>>>(ys, zs_final, zs, cs, nin);

    thrust::device_vector<size_t> positions(dnout, 0);
    /* Scan over the used positions in cs and store the result in positions
     * For a point in zs[i], positions[i] will now contain the index at
     * which that point should end up in the result array. */
    thrust::exclusive_scan(used.cbegin(), used.cend(), positions.begin(), 0, thrust::plus<size_t>()); 

    size_t *fs = thrust::raw_pointer_cast(positions.data());
    evtl_compact<<<NBLOCKS, NTHREADS>>>(zs, zs_final, cs, fs, dnout);

    *out = dout;
    *nout = positions.back(); /* Note that we don't actually resize here. */
}

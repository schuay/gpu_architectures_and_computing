#include "until.hpp"

#include <assert.h>
#include <thrust/scan.h>

#include "consolidate.hpp"
#include "globals.h"

__global__ static void
until_segm_and_cnst_kernel(const sigpt_t *lhs,
                           const sigpt_t *rhs,
                           const int *is_isect,
                           const int *isect_ixs,
                           const int n,
                           sigpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n - 1; i += blockDim.x * gridDim.x) {
        const int isect_ix = isect_ixs[i];

        const sigpt_t l = lhs[i];
        const sigpt_t r = rhs[i];

        const int lhs_is_less = (l.y <= r.y);
        const int rhs_is_less = !lhs_is_less;

        out[i + isect_ix] = (sigpt_t){ l.t,
                                       lhs_is_less * l.y + rhs_is_less * r.y,
                                       lhs_is_less * l.dy + rhs_is_less * r.dy };

        /* Intersection. */
    }
}

__global__ static void
until_mark_isecs(const sigpt_t *lhs,
                 const sigpt_t *rhs,
                 const int n,
                 int *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n - 1; i += blockDim.x * gridDim.x) {
        const sigpt_t l = lhs[i];
        const sigpt_t r1 = rhs[i];
        const sigpt_t r2 = rhs[i + 1];
        const int is_isec = (l.y < r1.y && l.y > r2.y) ||
                            (l.y > r1.y && l.y < r2.y);
        out[i] = is_isec;
    }
}

/**
 * AND of a signal rhs and a segment-wise constant lhs.
 * lhs and rhs must be consolidated.
 */
static void
until_segm_and_cnst(const thrust::device_ptr<sigpt_t> &lhs,
                    const int nlhs,
                    const thrust::device_ptr<sigpt_t> &rhs,
                    const int nrhs,
                    thrust::device_ptr<sigpt_t> *out,
                    int *nout)
{
    /* In step one, we need to detect all points where the constant signal lhs[t]
     * intersects rhs[t, t+1] and use these to create new signals for both
     * lhs and rhs. All intersection points clone the previous point in *lhs*.
     */

    thrust::device_ptr<int> is_isect = thrust::device_malloc<int>(nrhs);
    until_mark_isecs<<<NBLOCKS, NTHREADS>>>(lhs.get(), rhs.get(), nrhs, is_isect.get());

    thrust::device_ptr<int> isect_ixs = thrust::device_malloc<int>(nrhs);
    thrust::exclusive_scan(is_isect, is_isect + nrhs, isect_ixs);

    /* For each intersection at point t, isect_ixs[t] now holds the index
     * of the intersection sequence. The length of the combined sequence is
     * nrhs + isect_ixs[nrhs - 1]. */

    const int ndout = nrhs + isect_ixs[nrhs - 1];
    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(*nout);

    until_segm_and_cnst_kernel<<<NBLOCKS, NTHREADS>>>(
            lhs.get(), rhs.get(), is_isect.get(),
            isect_ixs.get(), nrhs, dout.get());

    *nout = ndout;
    *out = dout;
}

__global__ static void
derivates(sigpt_t *sig,
          const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n - 1; i += blockDim.x * gridDim.x) {
        const sigpt_t p1 = sig[i];
        const sigpt_t p2 = sig[i + 1];

        /* Assumes p1.t != p2.t. */

        sig[i].dy = (p2.y - p1.y) / (p2.t - p1.t);
    }
}

/**
 * This implementation follows algorithm 2 presented in 
 * Efficient Robust Monitoring for STL.
 * We use iz1, iz2, iz2 when referring to z{1,2,3} in the if branch,
 * and ez1, ez2, ez3 in the else branch.
 */
void
stl_until(const thrust::device_ptr<sigpt_t> &lhs,
          const int nlhs,
          const thrust::device_ptr<sigpt_t> &rhs,
          const int nrhs,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    thrust::device_ptr<sigpt_t> clhs;
    thrust::device_ptr<sigpt_t> crhs;
    int nc;

    consolidate(lhs, nlhs, rhs, nrhs, &clhs, &crhs, &nc);

    /* TODO: Remove this once (if) all input signals have valid dy's. */

    derivates<<<NTHREADS, NBLOCKS>>>(clhs.get(), nc);
    derivates<<<NTHREADS, NBLOCKS>>>(crhs.get(), nc);

    /* Do smart stuff here. A couple of things to think about:
     * We can't just assume we're processing signals of segments with dy <= 0
     * and dy > 0 separately, so we need some way to handle a signal
     * made up of both rising and falling segments.
     * We also need to ensure dy is correct before proceeding. */

    thrust::device_free(clhs);
    thrust::device_free(crhs);
}

#include "until.hpp"

#include <assert.h>
#include <float.h>
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

#include "interpolate.hpp" /* TODO: Remove me. */

typedef sigpt_t (*sig_binary)(const sigpt_t l, const sigpt_t r);

static sigpt_t
seq_bin_and(const sigpt_t l,
            const sigpt_t r)
{
    return (l.y < r.y) ? l : r;
}

static sigpt_t
seq_bin_or(const sigpt_t l,
           const sigpt_t r)
{
    return (l.y > r.y) ? l : r;
}

static void
seq_bin(const sigpt_t *lhs,
        const int nlhs,
        const sigpt_t *rhs,
        const int nrhs,
        sigpt_t **out,
        int *nout,
        const sig_binary f)
{
    sigpt_t *dout = (sigpt_t *)malloc(2 * (nlhs + nrhs) * sizeof(sigpt_t));

    int i = 0;
    int j = 0;
    int k = 0;

    while (i < nlhs && j < nrhs) {
        sigpt_t l = lhs[i];
        sigpt_t r = rhs[j];

        if (FLOAT_EQ(l.t, r.t)) {
            i++;
            j++;
        } else if (l.t < r.t) {
            assert(j != 0);
            r = interpolate(&rhs[j - i], &r, l.t);
            i++;
        } else {
            assert(i != 0);
            l = interpolate(&lhs[i - 1], &l, r.t);
            j++;
        }

        dout[k] = f(l, r);
        k++;
    }

    assert(i == nlhs && j == nrhs);
    assert(k <= 2 * (nlhs + nrhs));

    *out = dout;
    *nout = k;
}

void
seq_evtl(const sigpt_t *in,
         const int nin,
         sigpt_t **out,
         int *nout)
{
    int i = nin - 1;
    sigpt_t *zs = (sigpt_t*) malloc(2 * nin * sizeof(sigpt_t));
    sigpt_t *zs_final;
    int *cs = (int*) calloc(2 * nin, sizeof(int));
    int *fs = (int*) malloc(2 * nin * sizeof(int));

    zs[i * 2] = in[i];
    cs[i * 2] = 1;
    i--;

    for (; i >= 0; i--) {
        sigpt_t tmp;
        tmp.t = in[i].t;
        tmp.y = CUDA_MAX(in[i].y, zs[(i + 1) * 2].y);

        zs[i * 2] = tmp;
        cs[i * 2] = 1;

        const sigpt_t prev = zs[(i + 1) * 2];
        if (in[i].y > prev.y && in[i + 1].y < prev.y) {
            cs[i * 2 + 1] = 1;
            zs[i * 2 + 1].t = prev.t +
                (prev.t - zs[i * 2].t) *
                (prev.y - in[i + 1].y) / (in[i + 1].y - in[i].y);
            zs[i * 2 + 1].y = prev.y;
        }
    }

    fs[0] = 0;
    for (i = 1; i < 2 * nin; i++) {
        fs[i] = fs[i - 1] + cs[i - 1];
    }

    *nout = fs[2 * nin - 1];

    zs_final = (sigpt_t*) malloc(2 * (*nout) * sizeof(sigpt_t));

    for (i = 0; i < 2 * nin; i++) {
        if(cs[i] == 1) {
            zs_final[fs[i]] = zs[i];
        }
    }

    free(zs);
    free(cs);
    free(fs);

    *out = zs_final;
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

    /* Host arrays for sequential impl. */
    sigpt_t *hlhs = (sigpt_t *)malloc(nc * sizeof(sigpt_t));
    sigpt_t *hrhs = (sigpt_t *)malloc(nc * sizeof(sigpt_t));

    for (int i = 0; i < nc; i++) {
        hlhs[i] = clhs[i];
        hrhs[i] = crhs[i];
    }

    /* The result host array. 10 is some arbitrary constant to ensure we have enough space. */
    const int nres = 10 * nc;
    sigpt_t *result = (sigpt_t *)malloc(nres * sizeof(sigpt_t));

    int i = nc - 2;
    int j = nres - 2;
    float z = FLT_MAX * -1;

    /* The final point is simply an AND. */
    result[nres - 1] = seq_bin_and(hlhs[nc - 1], hrhs[nc - 1]);

    while (i >= 0) {
        sigpt_t z0[2];
        z0[0].y = z;
        z0[0].t = hlhs[i].t;
        z0[1].y = z;
        z0[1].t = hlhs[i + 1].t;

        sigpt_t *z2;
        int nz2;
        sigpt_t *z3;
        int nz3;

        if (hlhs[i].dy <= 0) {
            sigpt_t *z1;
            int nz1;
            seq_evtl(hrhs + i, 2, &z1, &nz1);

            seq_bin(z1, nz1, hlhs + i, 2, &z2, &nz2, seq_bin_and);

            sigpt_t z3lhs[2];
            z3lhs[0] = hlhs[i + 1];
            z3lhs[0].t = hlhs[i].t;
            z3lhs[1] = hlhs[i + 1];
            z3lhs[1].t = hlhs[i + 1].t;

            seq_bin(z3lhs, 2, z0, 2, &z3, &nz3, seq_bin_and);

            free(z1);
        } else {
            sigpt_t z1rhs[2];
            z1rhs[0] = hlhs[i];
            z1rhs[0].t = hrhs[i].t;
            z1rhs[1] = hlhs[i];
            z1rhs[1].t = hrhs[i + 1].t;

            sigpt_t *z1;
            int nz1;
            seq_bin(hrhs + i, 2, z1rhs, 2, &z1, &nz1, seq_bin_and);

            seq_evtl(z1, nz1, &z2, &nz2);
            seq_bin(hlhs + i, 2, z0, 2, &z3, &nz3, seq_bin_and);

            free(z1);
        }

        sigpt_t *z4;
        int nz4;
        seq_bin(z2, nz2, z3, nz3, &z4, &nz4, seq_bin_or);

        /* Note: The last point in result is skipped since the interval
         * is half-open! */
        for (int k = nz4 - 1 - 1; k >= 0; k--) {
            result[j] = z4[k];
            j--;
        }
        
        z = z4[0].y;
        i--;

        free(z2);
        free(z3);
        free(z4);
    }

    /* Transfer sequential results back to device memory. */
    const int ndout = nres - j - 1;
    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(ndout);

    for (int i = 0; i < ndout; i++) {
        dout[ndout - i - 1] = result[nres - i - 1];
    }

    *out = dout;
    *nout = ndout;

    free(hlhs);
    free(hrhs);
    free(result);

    thrust::device_free(clhs);
    thrust::device_free(crhs);
}

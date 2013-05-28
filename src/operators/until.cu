#include "until.hpp"

#include <assert.h>
#include <float.h>
#include <thrust/scan.h>

#include "consolidate.hpp"
#include "intersect.hpp"
#include "globals.h"

typedef struct {
    int i;            /* The original index this interval came from. */
    sigpt_t pts[10];  /* TODO: Determine the actual max count of points in an interval. */
    int n;            /* The count of points in this interval. */
} ivalpt_t;

#include <stdio.h> /* TODO: Remove me. */

static void
ivalpt_print(const char *name,
             const thrust::device_ptr<ivalpt_t> &in,
             const int n)
{
    printf("%s (%d)\n", name, n);
    for (int i = 0; i < n; i++) {
        ivalpt_t pt = in[i];
        printf("ivalpt_t { i: %d, n: %d }\n", pt.i, pt.n);
        for (int j = 0; j < pt.n; j++) {
            sigpt_t sigpt = pt.pts[j];
            printf("%i: {t: %f, y: %f, dy: %f}\n", j, sigpt.t, sigpt.y, sigpt.dy);
        }
    }
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
    sigpt_t *lhsi = (sigpt_t *)malloc(2 * (nlhs + nrhs) * sizeof(sigpt_t));
    sigpt_t *rhsi = (sigpt_t *)malloc(2 * (nlhs + nrhs) * sizeof(sigpt_t));

    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;

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

        lhsi[m] = l;
        rhsi[m] = r;
        m++;
    }

    assert(i == nlhs && j == nrhs);
    assert(m <= 2 * (nlhs + nrhs));

    int l = 0;
    for (k = 0; k < m - 1; k++) {
        dout[l++] = f(lhsi[k], rhsi[k]);

        float t;
        if (intersect(&lhsi[k], &rhsi[k], &lhsi[k + 1], &rhsi[k + 1], &t)) {
            sigpt_t a, b;
            a = interpolate(&lhsi[k], &lhsi[k + 1], t);
            b = interpolate(&rhsi[k], &rhsi[k + 1], t);

            dout[l++] = f(a, b);
        }
    }

    dout[l++] = f(lhsi[k], rhsi[k]);

    assert(l <= 2 * (nlhs + nrhs));

    *out = dout;
    *nout = l;

    free(lhsi);
    free(rhsi);
}

void
seq_evtl(const sigpt_t *in,
         const int nin,
         sigpt_t **out,
         int *nout)
{
    const int nzs = 2 * nin;
    sigpt_t *zs = (sigpt_t*) malloc(nzs * sizeof(sigpt_t));

    int i = nin - 1;
    int j = nzs - 1;

    zs[j] = in[i];

    i--;
    j--;

    while (i >= 0) {
        assert(j > 0);

        const sigpt_t prev = zs[j + 1];
        sigpt_t curr       = in[i];
        curr.y = CUDA_MAX(in[i].y, prev.y);

        if (in[i].y > prev.y && in[i + 1].y < prev.y) {
            float t;
            const int is_isec = intersect(&prev, &in[i], &prev, &in[i + 1], &t); 
            assert(is_isec);

            zs[j] = (sigpt_t){ t, prev.y, 0 };

            j--;
        }

        zs[j] = curr;

        i--;
        j--;
    }

    const int ndout = nzs - j - 1;
    sigpt_t *dout = (sigpt_t*) malloc(ndout * sizeof(sigpt_t));

    memcpy(dout, zs + j + 1, ndout * sizeof(sigpt_t));

    *out = dout;
    *nout = ndout;
}

__global__ static void
mark_positive_dys(const sigpt_t *in,
                  const int n,
                  int *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i] = (in[i].dy > 0.f);
    }
}

__global__ static void
mark_negative_dys(const sigpt_t *in,
                  const int n,
                  int *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i] = (in[i].dy <= 0.f);
    }
}

/**
 * Well, this turned into a rather ugly function.
 * What it actually does is: it takes the raw input
 * signals lhs and rhs together with their element count n;
 * two integer arrays negative_indices / positive_indices
 * specifying the compacted target indices of elements with respectively
 * negative (or 0) or positive dy in lhs; and finally,
 * target arrays.
 *
 * We construct an interval from the current location by storing both
 * the current point and its predecessor together with the original index
 * in lhs / rhs.
 *
 * If the current lhs.dy is negative, we store the current lhs and rhs
 * interval into neg_{lhs,rhs}. Otherwise, it is stored into pos_{lhs, rhs}.
 */
__global__ static void
ivalpt_extract_by_dy(const sigpt_t *lhs,
                     const sigpt_t *rhs,
                     const int n,
                     const int *negative_indices,
                     const int *positive_indices,
                     ivalpt_t *neg_lhs,
                     ivalpt_t *neg_rhs,
                     ivalpt_t *pos_lhs,
                     ivalpt_t *pos_rhs)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        ivalpt_t l, r;
        l.i = i;
        r.i = i;

        const sigpt_t l1 = lhs[i];
        l.pts[0] = l1;
        l.n = 1;

        r.pts[0] = rhs[i];
        r.n = 1;

        if (i != n - 1) {
            l.pts[1] = lhs[i + 1];
            l.n = 2;

            r.pts[1] = rhs[i + 1];
            r.n = 2;
        }

        if (l1.dy <= 0.f) {
            neg_lhs[negative_indices[i]] = l;
            neg_rhs[negative_indices[i]] = r;
        } else {
            pos_lhs[positive_indices[i]] = l;
            pos_rhs[positive_indices[i]] = r;
        }
    }
}

/* TODO: Remove this once we have a segment EVTL implementation.
 * That one will need to take an array of ivalpt_t as input as
 * well as output. */
__global__ static void
naive_evtl(const ivalpt_t *in,
           const int n,
           ivalpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const ivalpt_t ival_in = in[i];
        
        ivalpt_t ival_out = ival_in;

        /* EVTL applied to a point results in the point. */

        if (ival_in.n == 1) {
            out[i] = ival_out;
            continue;
        }

        /* Otherwise, it's the maximum y of the current point and its successor. 
         * TODO: Don't assume an interval of two points. */

        const sigpt_t s = ival_in.pts[0];
        const sigpt_t t = ival_in.pts[1];
        sigpt_t result = (s.y > t.y) ? s : t;
        result.t = s.t;

        ival_out.pts[0] = result;
        ival_out.n = 2;

        out[i] = ival_out;
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
    /* First, ensure both incoming signals are well formed. */

    thrust::device_ptr<sigpt_t> clhs;
    thrust::device_ptr<sigpt_t> crhs;
    int nc;

    consolidate(lhs, nlhs, rhs, nrhs, &clhs, &crhs, &nc);

    /* Set dy for all points. */

    derivates<<<NTHREADS, NBLOCKS>>>(clhs.get(), nc);
    derivates<<<NTHREADS, NBLOCKS>>>(crhs.get(), nc);

    /* Split incoming signals into chunks we can use conveniently in parallel. */

    thrust::device_ptr<int> indices_of_negative_dys =
        thrust::device_malloc<int>(nc);
    mark_negative_dys<<<NBLOCKS, NTHREADS>>>(clhs.get(), nc,
            indices_of_negative_dys.get());
    thrust::exclusive_scan(indices_of_negative_dys,
            indices_of_negative_dys + nc,
            indices_of_negative_dys);

    thrust::device_ptr<int> indices_of_positive_dys =
        thrust::device_malloc<int>(nc);
    mark_positive_dys<<<NBLOCKS, NTHREADS>>>(clhs.get(), nc,
            indices_of_positive_dys.get());
    thrust::exclusive_scan(indices_of_positive_dys,
            indices_of_positive_dys + nc,
            indices_of_positive_dys);

    const int nnegative_dys = indices_of_negative_dys[nc - 1];
    thrust::device_ptr<ivalpt_t> neg_dys_lhs =
        thrust::device_malloc<ivalpt_t>(nnegative_dys);
    thrust::device_ptr<ivalpt_t> neg_dys_rhs =
        thrust::device_malloc<ivalpt_t>(nnegative_dys);

    const int npositive_dys = indices_of_positive_dys[nc - 1];
    thrust::device_ptr<ivalpt_t> pos_dys_lhs =
        thrust::device_malloc<ivalpt_t>(npositive_dys);
    thrust::device_ptr<ivalpt_t> pos_dys_rhs =
        thrust::device_malloc<ivalpt_t>(npositive_dys);

    ivalpt_extract_by_dy<<<NBLOCKS, NTHREADS>>>(
            clhs.get(), crhs.get(), nc,
            indices_of_negative_dys.get(),
            indices_of_positive_dys.get(),
            neg_dys_lhs.get(),
            neg_dys_rhs.get(),
            pos_dys_lhs.get(),
            pos_dys_rhs.get());

    /* neg_dys_lhs now holds all indices of negative dy's in clhs,
     * and pos_dys_lhs all indices of positive dy's in clhs. Likewise for
     * rhs (it still depends on dy's in clhs though!) */

    thrust::device_ptr<ivalpt_t> iz1 = thrust::device_malloc<ivalpt_t>(nnegative_dys);
    naive_evtl<<<NBLOCKS, NTHREADS>>>(neg_dys_rhs.get(), nnegative_dys, iz1.get());

    /* ================== The sequential implementation starts here. ================== */

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
    float z = -INFINITY;

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

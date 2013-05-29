#include "until.hpp"

#include <stdio.h>
#include <thrust/scan.h>

#include "consolidate.hpp"
#include "interpolate.hpp"
#include "intersect.hpp"
#include "globals.h"

/* TODO: Determine the actual max count of points in an interval. */
#define IVALPT_COUNT (10)

enum {
    OP_AND,
    OP_OR
};

typedef sigpt_t (*sig_binary)(const sigpt_t l, const sigpt_t r);

#include "until_seq.cu" /* TODO: Remove me. */

typedef struct {
    int i;                      /* The original index this interval came from. */
    sigpt_t pts[IVALPT_COUNT];  /* The points contained in this interval. */
    int n;                      /* The count of points in this interval. */
} ivalpt_t;


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

__host__ __device__ static void
seq_evtl_internal(const sigpt_t *in,
                  const int nin,
                  sigpt_t *out,
                  int *nout)
{
    const int nzs = *nout;
    sigpt_t *zs = out;

    int i = nin - 1;
    int j = nzs - 1;

    zs[j] = in[i];

    i--;
    j--;

    while (i >= 0) {
        const sigpt_t prev = zs[j + 1];
        sigpt_t curr       = in[i];
        curr.y = CUDA_MAX(in[i].y, prev.y);

        if (in[i].y > prev.y && in[i + 1].y < prev.y) {
            float t;
            const int is_isec = intersect(&prev, &in[i], &prev, &in[i + 1], &t); 

            zs[j] = (sigpt_t){ t, prev.y, 0 };

            j--;
        }

        zs[j] = curr;

        i--;
        j--;
    }

    const int ndout = nzs - j - 1;

    for (i = 0; i < ndout; i++) {
        out[i] = out[j + 1 + i];
    }

    *nout = ndout;
}

__host__ __device__ static sigpt_t
seq_bin_and(const sigpt_t l,
            const sigpt_t r)
{
    return (l.y < r.y) ? l : r;
}

__host__ __device__ static sigpt_t
seq_bin_or(const sigpt_t l,
           const sigpt_t r)
{
    return (l.y > r.y) ? l : r;
}

/* Note: The actual binary operator used to be a function pointer,
 * which makes much more sense than using this flag. However,
 * I couldn't figure out how to use them with CUDA. Additionally,
 * function pointers would require switching to at least sm_20. */
__host__ __device__ static void
seq_bin_internal(const sigpt_t *lhs,
                 const int nlhs,
                 const sigpt_t *rhs,
                 const int nrhs,
                 sigpt_t *out,
                 int *nout,
                 const int op)
{
    sigpt_t lhsi[IVALPT_COUNT];
    sigpt_t rhsi[IVALPT_COUNT];

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
            r = interpolate(&rhs[j - i], &r, l.t);
            i++;
        } else {
            l = interpolate(&lhs[i - 1], &l, r.t);
            j++;
        }

        lhsi[m] = l;
        rhsi[m] = r;
        m++;
    }

    int l = 0;
    for (k = 0; k < m - 1; k++) {
        out[l++] = (op == OP_AND) ? seq_bin_and(lhsi[k], rhsi[k])
                                  : seq_bin_or (lhsi[k], rhsi[k]);

        float t;
        if (intersect(&lhsi[k], &rhsi[k], &lhsi[k + 1], &rhsi[k + 1], &t)) {
            sigpt_t a, b;
            a = interpolate(&lhsi[k], &lhsi[k + 1], t);
            b = interpolate(&rhsi[k], &rhsi[k + 1], t);

            out[l++] = (op == OP_AND) ? seq_bin_and(a, b)
                                      : seq_bin_or (a, b);
        }
    }

    out[l++] = (op == OP_AND) ? seq_bin_and(lhsi[k], rhsi[k])
                              : seq_bin_or (lhsi[k], rhsi[k]);

    *nout = l;
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

__global__ static void
segment_evtl(const ivalpt_t *in,
             const int n,
             ivalpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const ivalpt_t ival_in = in[i];
        ivalpt_t ival_out = ival_in;
        ival_out.n = IVALPT_COUNT;

        seq_evtl_internal(ival_in.pts, ival_in.n, ival_out.pts, &ival_out.n);

        out[i] = ival_out;
    }
}

__global__ static void
segment_and(const ivalpt_t *lhs,
            const ivalpt_t *rhs,
            const int n,
            ivalpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const ivalpt_t ival_l = lhs[i];
        const ivalpt_t ival_r = rhs[i];

        ivalpt_t ival_out;
        ival_out.i = ival_l.i;
        ival_out.n = IVALPT_COUNT;

        seq_bin_internal(ival_l.pts, ival_l.n, ival_r.pts, ival_r.n,
                ival_out.pts, &ival_out.n, OP_AND);

        out[i] = ival_out;
    }
}

struct ivalpt_constantize : public thrust::unary_function<ivalpt_t, ivalpt_t>
{
    __device__ ivalpt_t
    operator()(const ivalpt_t &in) const
    {
        ivalpt_t out = in;

        for (int i = 1; i < out.n; i++) {
            out.pts[i].y = out.pts[0].y;
        }

        return out;
    }
};

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
     * rhs (it still depends on dy's in clhs though!)
     *
     * We still require the segment-wise constant signal of lhs. */

    thrust::device_ptr<ivalpt_t> neg_dys_lhs_c =
        thrust::device_malloc<ivalpt_t>(nnegative_dys);
    thrust::device_ptr<ivalpt_t> pos_dys_lhs_c =
        thrust::device_malloc<ivalpt_t>(npositive_dys);

    thrust::transform(neg_dys_lhs, neg_dys_lhs + nnegative_dys, neg_dys_lhs_c, ivalpt_constantize());
    thrust::transform(pos_dys_lhs, pos_dys_lhs + npositive_dys, pos_dys_lhs_c, ivalpt_constantize());

    /* We begin with the if branch: */

    thrust::device_ptr<ivalpt_t> iz1 = thrust::device_malloc<ivalpt_t>(nnegative_dys);
    segment_evtl<<<NBLOCKS, NTHREADS>>>(neg_dys_rhs.get(), nnegative_dys, iz1.get());

    thrust::device_ptr<ivalpt_t> iz2 = iz1;
    segment_and<<<NBLOCKS, NTHREADS>>>(iz1.get(), neg_dys_lhs.get(), nnegative_dys, iz2.get());

    /* On to the else branch: */

    thrust::device_ptr<ivalpt_t> ez1 = thrust::device_malloc<ivalpt_t>(npositive_dys);
    segment_and<<<NBLOCKS, NTHREADS>>>(pos_dys_rhs.get(), pos_dys_lhs_c.get(), npositive_dys, ez1.get());

    thrust::device_ptr<ivalpt_t> ez2 = ez1; /* If this is ever changed, don't forget to free. */
    segment_evtl<<<NBLOCKS, NTHREADS>>>(ez1.get(), npositive_dys, ez2.get());

    /* ================== The sequential implementation starts here. ================== */

    seq_until(clhs, crhs, nc, out, nout);

    thrust::device_free(clhs);
    thrust::device_free(crhs);
}

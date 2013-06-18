#include "until.hpp"

#include <stdio.h>
#include <thrust/merge.h>
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

typedef struct {
    int i;                      /* The original index this interval came from. */
    sigpt_t pts[IVALPT_COUNT];  /* The points contained in this interval. */
    int n;                      /* The count of points in this interval. */
} ivalpt_t;

#include "until_seq.cu" /* TODO: Remove me. */


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

struct ivalpt_less : public thrust::binary_function<ivalpt_t, ivalpt_t, bool>
{
    __device__ bool
    operator()(const ivalpt_t &lhs, const ivalpt_t &rhs) const
    {
        return lhs.i < rhs.i;
    }
};

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

/**
 * Sequentially calculate z0. The first version actually ran on the host,
 * but that turned out to be much slower than just using a single GPU thread.
 */

__global__ static void
seq_z0(const sigpt_t *clhs,
       const sigpt_t *crhs,
       const ivalpt_t *z2,
       const int n,
       ivalpt_t *out)
{
    /* The final point is simply FALSE. */
    sigpt_t z0 = (sigpt_t){ clhs[n - 1].t, -INFINITY, 0.f };

    ivalpt_t ival;
    ival.i = n - 1;
    ival.n = 1;
    ival.pts[0] = (sigpt_t){ clhs[n - 1].t, z0.y, 0.f };
    out[n - 1] = ival;

    for (int i = n - 2; i >= 0; i--) {
        sigpt_t z3;

        if (clhs[i].dy <= 0) {
            const sigpt_t z3lhs = (sigpt_t){ clhs[i].t, clhs[i + 1].y, 0.f };
            z3 = seq_bin_and(z3lhs, z0);
        } else {
            z3 = seq_bin_and(clhs[i], z0);
        }

        /* Transfer sequential results back to device memory. */
        ival.i = i;
        ival.n = 2;
        ival.pts[0] = (sigpt_t){ clhs[i].t, z0.y, 0.f };
        ival.pts[1] = (sigpt_t){ clhs[i + 1].t, z0.y, 0.f };
        out[i] = ival;

        z0 = seq_bin_or(z2[i].pts[0], z3);
    }
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
segment_bin(const ivalpt_t *lhs,
            const ivalpt_t *rhs,
            const int n,
            ivalpt_t *out,
            const int op)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const ivalpt_t ival_l = lhs[i];
        const ivalpt_t ival_r = rhs[i];

        ivalpt_t ival_out;
        ival_out.i = ival_l.i;
        ival_out.n = IVALPT_COUNT;

        seq_bin_internal(ival_l.pts, ival_l.n, ival_r.pts, ival_r.n,
                ival_out.pts, &ival_out.n, op);

        out[i] = ival_out;
    }
}

__global__ static void
extract_z4(const sigpt_t *y,
           const int n,
           ivalpt_t *z4)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const sigpt_t y0 = y[i];

        ivalpt_t ival;
        ival.i = i;
        
        if (i == n - 1) {
            ival.n = 1;
            ival.pts[0] = y0;
            z4[i] = ival;
            continue;
        }

        const sigpt_t y1 = y[i + 1];

        ival.n = 2;
        ival.pts[1] = y1;
        ival.pts[0].t = y0.t;
        ival.pts[0].y = (y0.dy <= 0) ? y1.y : y0.y;

        z4[i] = ival;
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

struct ivalpt_n_minus_one : public thrust::unary_function<ivalpt_t, int>
{
    __device__ int
    operator()(const ivalpt_t &in) const
    {
        const int n = in.n - 1;
        return (n > 0) ? n : 1;
    }
};

/**
 * If you know Haskell's concat, this does pretty much the same thing:
 * It takes a list of lists, and concatenates them into one single list.
 * The only difference is that the last point for each interval is omitted
 * (except in the very last interval).
 */
__global__ static void
ivalpt_concat(const ivalpt_t *ivals,
              const int *ixs,
              const int n,
              sigpt_t *sigs)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const ivalpt_t ival = ivals[i];
        const int ix = ixs[i];

        const int the_end = CUDA_MAX(1, ival.n - 1);
        for (int j = 0; j < the_end; j++) {
            sigs[ix + j] = ival.pts[j];
        }
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

    const sigpt_t last = clhs[nc - 1];
    const int last_is_falling = (last.dy <= 0);
    const int last_is_rising = !last_is_falling;

    const int nnegative_dys = indices_of_negative_dys[nc - 1] + last_is_falling;
    thrust::device_ptr<ivalpt_t> neg_dys_lhs =
        thrust::device_malloc<ivalpt_t>(nnegative_dys);
    thrust::device_ptr<ivalpt_t> neg_dys_rhs =
        thrust::device_malloc<ivalpt_t>(nnegative_dys);

    const int npositive_dys = indices_of_positive_dys[nc - 1] + last_is_rising;
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

    thrust::device_free(indices_of_negative_dys);
    thrust::device_free(indices_of_positive_dys);

    /* neg_dys_lhs now holds all indices of negative dy's in clhs,
     * and pos_dys_lhs all indices of positive dy's in clhs. Likewise for
     * rhs (it still depends on dy's in clhs though!)
     *
     * We still require the segment-wise constant signal of lhs. */

    thrust::device_ptr<ivalpt_t> pos_dys_lhs_c =
        thrust::device_malloc<ivalpt_t>(npositive_dys);

    thrust::transform(pos_dys_lhs, pos_dys_lhs + npositive_dys, pos_dys_lhs_c, ivalpt_constantize());

    thrust::device_free(pos_dys_lhs);

    /* We begin with the if branch: */

    thrust::device_ptr<ivalpt_t> iz1 = thrust::device_malloc<ivalpt_t>(nnegative_dys);
    segment_evtl<<<NBLOCKS, NTHREADS>>>(neg_dys_rhs.get(), nnegative_dys, iz1.get());

    thrust::device_free(neg_dys_rhs);

    thrust::device_ptr<ivalpt_t> iz2 = iz1; /* If this is ever changed, don't forget to free. */
    segment_bin<<<NBLOCKS, NTHREADS>>>(iz1.get(), neg_dys_lhs.get(), nnegative_dys, iz2.get(), OP_AND);

    thrust::device_free(neg_dys_lhs);

    /* On to the else branch: */

    thrust::device_ptr<ivalpt_t> ez1 = thrust::device_malloc<ivalpt_t>(npositive_dys);
    segment_bin<<<NBLOCKS, NTHREADS>>>(pos_dys_rhs.get(), pos_dys_lhs_c.get(), npositive_dys, ez1.get(), OP_AND);

    thrust::device_free(pos_dys_lhs_c);
    thrust::device_free(pos_dys_rhs);

    thrust::device_ptr<ivalpt_t> ez2 = ez1; /* If this is ever changed, don't forget to free. */
    segment_evtl<<<NBLOCKS, NTHREADS>>>(ez1.get(), npositive_dys, ez2.get());

    /* Merge z2 back into one sequence for sequential computation of z0. */

    thrust::device_ptr<ivalpt_t> z2 = thrust::device_malloc<ivalpt_t>(nc);
    thrust::merge(iz2, iz2 + nnegative_dys, ez2, ez2 + npositive_dys, z2, ivalpt_less());

    thrust::device_free(ez2);
    thrust::device_free(iz2);

    /* *Sequentially* compute z0. */

    thrust::device_ptr<ivalpt_t> z0 = thrust::device_malloc<ivalpt_t>(nc);
    seq_z0<<<1, 1>>>(clhs.get(), crhs.get(), z2.get(), nc, z0.get());

    thrust::device_free(crhs);

    /* Extract z4 (the LHS input to the z3 steps). */

    thrust::device_ptr<ivalpt_t> z4 = thrust::device_malloc<ivalpt_t>(nc);
    extract_z4<<<NBLOCKS, NTHREADS>>>(clhs.get(), nc, z4.get());

    thrust::device_free(clhs);

    /* z3 and the interval-wise result are common over both branches. */

    thrust::device_ptr<ivalpt_t> z3 = thrust::device_malloc<ivalpt_t>(nc);
    segment_bin<<<NBLOCKS, NTHREADS>>>(z4.get(), z0.get(), nc, z3.get(), OP_AND);

    thrust::device_free(z0);
    thrust::device_free(z4);

    thrust::device_ptr<ivalpt_t> iout = thrust::device_malloc<ivalpt_t>(nc);
    segment_bin<<<NBLOCKS, NTHREADS>>>(z2.get(), z3.get(), nc, iout.get(), OP_OR);

    thrust::device_free(z2);
    thrust::device_free(z3);

    /* Finally, merge our interval points back into an sigpt_t sequence. */

    thrust::device_ptr<int> concat_indices = thrust::device_malloc<int>(nc);
    thrust::transform(iout, iout + nc, concat_indices, ivalpt_n_minus_one());
    thrust::exclusive_scan(concat_indices, concat_indices + nc, concat_indices);

    const ivalpt_t the_end = iout[nc - 1];
    const int ndout = concat_indices[nc - 1] + the_end.n;
    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(ndout);

    ivalpt_concat<<<NBLOCKS, NTHREADS>>>(iout.get(), concat_indices.get(), nc, dout.get());

    thrust::device_free(iout);
    thrust::device_free(concat_indices);

    *out = dout;
    *nout = ndout;
}

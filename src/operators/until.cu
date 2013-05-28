#include "until.hpp"

#include <assert.h>
#include <float.h>
#include <thrust/scan.h>

#include "consolidate.hpp"
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

__global__ static void
extract_indices_by_dy(const sigpt_t *sig,
                      const int n,
                      const int *negative_indices,
                      const int *positive_indices,
                      int *negatives,
                      int *positives)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (sig[i].dy <= 0.f) {
            negatives[negative_indices[i]] = i;
        } else {
            positives[positive_indices[i]] = i;
        }
    }
}

/* TODO: Remove this once we have a segment EVTL implementation.
 * That one will need to take an array of ivalpt_t as input as
 * well as output. */
__global__ static void
naive_evtl(const sigpt_t *sig,
           const int nsig,
           const int *indices,
           const int nind,
           ivalpt_t *out)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < nind; i += blockDim.x * gridDim.x) {
        const int ind = indices[i];
        const sigpt_t s = sig[ind];
        
        ivalpt_t ival;
        ival.i = ind;

        /* The last point always results in itself. */

        if (ind == nsig - 1) {
            ival.pts[0] = s;
            ival.n = 1;
            out[i] = ival;
            continue;
        }

        /* Otherwise, it's the maximum y of the current point and its successor. */

        const sigpt_t t = sig[ind + 1];
        sigpt_t result = (s.y > t.y) ? s : t;
        result.t = s.t;

        ival.pts[0] = result;
        ival.pts[1] = t;
        ival.n = 2;

        out[i] = ival;
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
    thrust::device_ptr<int> negative_dys =
        thrust::device_malloc<int>(nnegative_dys);

    const int npositive_dys = indices_of_positive_dys[nc - 1];
    thrust::device_ptr<int> positive_dys =
        thrust::device_malloc<int>(npositive_dys);

    extract_indices_by_dy<<<NBLOCKS, NTHREADS>>>(
            clhs.get(), nc,
            indices_of_negative_dys.get(),
            indices_of_positive_dys.get(),
            negative_dys.get(),
            positive_dys.get());

    /* negative_dys now holds all indices of negative dy's in clhs,
     * and positive_dys all indices of positive dy's in clhs. */

    thrust::device_ptr<ivalpt_t> iz1 = thrust::device_malloc<ivalpt_t>(nnegative_dys);
    naive_evtl<<<NBLOCKS, NTHREADS>>>(
            clhs.get(), nc,
            negative_dys.get(), nnegative_dys,
            iz1.get());

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

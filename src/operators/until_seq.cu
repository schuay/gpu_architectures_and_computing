#include "interpolate.hpp" /* TODO: Remove me. */

__host__ __device__ static void
seq_evtl_internal(const sigpt_t *in,
                  const int nin,
                  sigpt_t *out,
                  int *nout);

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

static void
seq_evtl(const sigpt_t *in,
         const int nin,
         sigpt_t **out,
         int *nout)
{
    int nzs = 2 * nin;
    sigpt_t *zs = (sigpt_t*) malloc(nzs * sizeof(sigpt_t));

    seq_evtl_internal(in, nin, zs, &nzs);

    *out = zs;
    *nout = nzs;
}

static void
seq_until(const thrust::device_ptr<sigpt_t> &clhs,
          const thrust::device_ptr<sigpt_t> &crhs,
          const int n,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    /* Host arrays for sequential impl. */
    sigpt_t *hlhs = (sigpt_t *)malloc(n * sizeof(sigpt_t));
    sigpt_t *hrhs = (sigpt_t *)malloc(n * sizeof(sigpt_t));

    for (int i = 0; i < n; i++) {
        hlhs[i] = clhs[i];
        hrhs[i] = crhs[i];
    }

    /* The result host array. 10 is some arbitrary constant to ensure we have enough space. */
    const int nres = 10 * n;
    sigpt_t *result = (sigpt_t *)malloc(nres * sizeof(sigpt_t));

    int i = n - 2;
    int j = nres - 2;
    float z = -INFINITY;

    /* The final point is simply an AND. */
    result[nres - 1] = seq_bin_and(hlhs[n - 1], hrhs[n - 1]);

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

        /* Note: The last point in result is skipped sine the interval
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
}

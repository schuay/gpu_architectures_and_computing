#include <assert.h>
#include <float.h>

__host__ __device__ static void
seq_evtl_internal(const sigpt_t *in,
                  const int nin,
                  sigpt_t *out,
                  int *nout);

__host__ __device__ static void
seq_bin_internal(const sigpt_t *lhs,
                 const int nlhs,
                 const sigpt_t *rhs,
                 const int nrhs,
                 sigpt_t *out,
                 int *nout,
                 const int op);

__host__ __device__ static sigpt_t
seq_bin_and(const sigpt_t l,
            const sigpt_t r);

__host__ __device__ static sigpt_t
seq_bin_or(const sigpt_t l,
           const sigpt_t r);

static void
seq_bin(const sigpt_t *lhs,
        const int nlhs,
        const sigpt_t *rhs,
        const int nrhs,
        sigpt_t **out,
        int *nout,
        const int op)
{
    int ndout = 2 * (nlhs + nrhs);
    sigpt_t *dout = (sigpt_t *)malloc(2 * (nlhs + nrhs) * sizeof(sigpt_t));

    seq_bin_internal(lhs, nlhs, rhs, nrhs, dout, &ndout, op);

    *out = dout;
    *nout = ndout;
}

static void
seq_until(const thrust::device_ptr<sigpt_t> &clhs,
          const thrust::device_ptr<sigpt_t> &crhs,
          const thrust::device_ptr<ivalpt_t> &z2,
          const int n,
          thrust::device_ptr<sigpt_t> *out,
          int *nout)
{
    /* Host arrays for sequential impl. */
    thrust::host_vector<sigpt_t> hlhs(clhs, clhs + n);
    thrust::host_vector<sigpt_t> hrhs(crhs, crhs + n);
    thrust::host_vector<ivalpt_t> hz2(z2, z2 + n);

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

        sigpt_t *z3;
        int nz3;

        if (hlhs[i].dy <= 0) {
            sigpt_t z3lhs[2];
            z3lhs[0] = hlhs[i + 1];
            z3lhs[0].t = hlhs[i].t;
            z3lhs[1] = hlhs[i + 1];
            z3lhs[1].t = hlhs[i + 1].t;

            seq_bin(z3lhs, 2, z0, 2, &z3, &nz3, OP_AND);
        } else {
            seq_bin(&hlhs[i], 2, z0, 2, &z3, &nz3, OP_AND);
        }

        sigpt_t *z4;
        int nz4;
        seq_bin(hz2[i].pts, hz2[i].n, z3, nz3, &z4, &nz4, OP_OR);

        /* Note: The last point in result is skipped since the interval
         * is half-open! */
        for (int k = nz4 - 1 - 1; k >= 0; k--) {
            result[j] = z4[k];
            j--;
        }
        
        z = z4[0].y;
        i--;

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

    free(result);
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/scan.h>

extern "C" {
#include "sigpt.h"
}

#define NBLOCKS (256)
#define NTHREADS (256)

#define NITEMS (256 * 257)

#define FLOAT_DELTA (0.000000001f)

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CUDA_MAX(a, b) (((a) > (b)) * (a) + ((a) <= (b)) * (b))

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)

bool
_checkCudaError(cudaError_t result, const char *func, const char *file, int line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        return true;
    } else {
        return false;
    }
}

/**
 * sizeof(out) == sizeof(in).
 */
__global__ void
stl_not(const sigpt_t *in, sigpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sigpt_t s = in[i];
        s.y *= -1.f;
        s.dy *= -1.f;
        out[i] = s;
    }
}

#define FLAG_LHS (1 << 0)
#define FLAG_RHS (1 << 1)
#define FLAG_ISC (1 << 2)
#define FLAG_DEL (1 << 3)

typedef struct {
    float t;
    int flags;
} seqpt_t;

struct seqpt_less : public thrust::binary_function<seqpt_t, seqpt_t, bool>
{
    __device__ bool
    operator()(const seqpt_t &lhs, const seqpt_t &rhs) const
    {
        return lhs.t < rhs.t;
    }
};

__global__ void
time_seq_with_proto_intersections(const sigpt_t *in, seqpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float t = in[i].t;
        out[i * 2].t = t;
        out[i * 2 + 1].t = t;
        out[i * 2 + 1].flags |= FLAG_ISC;
    }
}

/**
 * sizeof(out) = 4 * max(sizeof(lhs), sizeof(rhs)).
 */
void
stl_and(const thrust::device_vector<sigpt_t> &lhs,
        const thrust::device_vector<sigpt_t> &rhs,
        thrust::device_vector<sigpt_t> out)
{
    /* Construct a sorted time sequence ts which contains all t <- lhs, all t <- rhs, and all
     * intersection points between lhs and rhs. The sequence contains only unique points.
     * 
     * Using interpolation, construct lhs' and rhs' such that they contain all t <- ts.
     *
     * Finally, do a simple min() over these arrays.
     *
     * We need:
     * * efficient interpolation
     * * a parallel merge. bitonic sort? a binary sort merge? preserve information of
     *   source array and source pointer so we can do intersections later. how to get memory
     *   for all of these additional fields?
     * * a parallel method to find all intersection points. we can use the merged section,
     *   but this is also not as simple as it seems at first.
     */

    /* What we could now do is create two vectors of seqpt_t lhst, rhst such that
     * lhst[0] ~ lhs[0], flagged LHS; lhst[1] ~ lhs[0], flagged ISC (intersection),
     * lhst[2] ~ lhs[1], ...
     */

    const sigpt_t *ptr_lhs = thrust::raw_pointer_cast(lhs.data());

    seqpt_t seqinit = { 0.f, FLAG_LHS };
    thrust::device_vector<seqpt_t> lhst(lhs.size() * 2, seqinit);
    seqpt_t *ptr_lhst = thrust::raw_pointer_cast(lhst.data());

    time_seq_with_proto_intersections<<<NBLOCKS, NTHREADS>>>(ptr_lhs, ptr_lhst, lhs.size());


    const sigpt_t *ptr_rhs = thrust::raw_pointer_cast(rhs.data());

    seqinit.flags = FLAG_RHS;
    thrust::device_vector<seqpt_t> rhst(rhs.size() * 2, seqinit);
    seqpt_t *ptr_rhst = thrust::raw_pointer_cast(rhst.data());

    time_seq_with_proto_intersections<<<NBLOCKS, NTHREADS>>>(ptr_rhs, ptr_rhst, rhs.size());

    /* We then merge both of these vectors. */

    const int nts = lhst.size() + rhst.size();
    thrust::device_vector<seqpt_t> ts(nts, seqinit);

    thrust::merge(lhst.begin(), lhst.end(),
            rhst.begin(), rhst.end(),
            ts.begin(),
            seqpt_less());

    for (thrust::device_vector<seqpt_t>::iterator iter = ts.begin();
         iter != ts.begin() + 10; iter++) {
    	seqpt_t s = *iter;
    	printf("{ %f, %x }\n", s.t, s.flags);
    }

    /* Next, we go through and fill in ISC elements; if there's an intersection
     * we set the time accordingly, and if there isn't, we mark it as DEL.
     * An intersection must always be between the latest (lhs, rhs) and the next such
     * pair.
     * Then, mark duplicate time values DEL.
     * Finally we compact the array, removing all DEL elements.
     */
}

int
main(int argc, char **argv)
{
    sigpt_t *a = sigpt_random(42, NITEMS);
    sigpt_t *b = sigpt_random(43, NITEMS);
    sigpt_t *c = (sigpt_t *)calloc(4 * NITEMS,sizeof(sigpt_t));

    thrust::device_vector<sigpt_t> lhs(a, a + NITEMS);
    thrust::device_vector<sigpt_t> rhs(b, b + NITEMS);
    thrust::device_vector<sigpt_t> out(c, c + 4 * NITEMS);

    stl_and(lhs, rhs, out);

    free(a);
    free(b);
    free(c);

    /* And a Thrust scan operation, let's see how we can integrate that with the rest of
     * the code...
     */

    /*

    thrust::plus<float> binary_op;

    thrust::host_vector<float> hostVector(NITEMS);
    thrust::generate(hostVector.begin(), hostVector.end(), rand);

    thrust::device_vector<float> deviceVector = hostVector;
    thrust::exclusive_scan(deviceVector.begin(), deviceVector.end(), deviceVector.begin(), 0.f, binary_op);

    thrust::copy(deviceVector.begin(), deviceVector.end(), hostVector.begin());

    for (thrust::device_vector<float>::iterator iter = deviceVector.begin();
         iter != deviceVector.begin() + 10;
         iter++) {
    	float val = *iter;
    	printf("%f ", val);
    }
    printf("\n");

    */

    return 0;
}

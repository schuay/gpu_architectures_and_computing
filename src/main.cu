#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

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
    float t;    /**< The time value. */
    int i;      /**< The original index. */
    int i_lhs, i_rhs;
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

struct sigpt_min : public thrust::binary_function<sigpt_t, sigpt_t, sigpt_t>
{
    __device__ sigpt_t
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        sigpt_t r = rhs;
        sigpt_t s = lhs;

        const int is_rhs_min = (r.y <= s.y) != 0;

        s.y += is_rhs_min * (r.y - s.y);
        s.dy += is_rhs_min * (r.dy - s.dy);

        return s;
    }
};

__global__ void
sigpt_extrapolate(const sigpt_t *lhs,
                  const sigpt_t *rhs,
                  const int *lhs_max,
                  const int *rhs_max,
                  const seqpt_t *ts,
                  sigpt_t *lhs_extrapolated,
                  sigpt_t *rhs_extrapolated)
{
    /* TODO: Use the information provided by lhs, rhs, lhs_max, rhs_max, and ts
     * to extrapolated a signal point sequence for both lhs and rhs for each
     * time point in ts. */
}

__global__ void
extract_i(const seqpt_t *in, int *out, int n, int flag)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t s = in[i];
        const int has_flag = (s.flags & flag) != 0;
        out[i] = has_flag * s.i;
    }
}

__global__ void
time_seq_with_proto_intersections(const sigpt_t *in, seqpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        float t = in[i].t;
        out[i * 2].t = t;
        out[i * 2].i = i;
        out[i * 2 + 1].t = t;
        out[i * 2 + 1].i = i;
        out[i * 2 + 1].flags |= FLAG_ISC;
    }
}

__global__ void
calculate_intersection_seqs(const sigpt_t *lhs, const sigpt_t *rhs,
                            const int *lhs_max, const int *rhs_max,
                            seqpt_t *ts, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* At this point, we are only interested in intersection elements in ts.
     * These are located at every index 2 * i + 1, i <- N.
     *
     * Assuming ts.flags ~ FLAG_LHS: ts[i].i is the index of the current
     * point in lhs (the next point is obviously lhs[ts[i].i + 1]).
     * The closest point to the left in in rhs is located at index
     * rhs_max[i], and the closest to the right is rhs_max[i] + 1.
     *
     * This is enough information to determine the time of the signal 
     * intersection.
     */

    for (int i = tid; 2 * i + 1 < n; i += blockDim.x * gridDim.x) {
        const int ii = 2 * i + 1;
        seqpt_t s = ts[ii];

        const int is_lhs = (s.flags & FLAG_LHS) != 0;

        /* TODO: Optimize. */
        const sigpt_t *this_sig = is_lhs ? lhs : rhs;
        const sigpt_t *other_sig = is_lhs ? rhs : lhs;
        const int *other_sig_max = is_lhs ? lhs_max : rhs_max;

        /* We now have four points corresponding to the end points of the
         * two line segments. (x1, y1) and (x2, y2) for one line segment,
         * (x3, y3) and (x4, y4) for the other line segment.
         * We are interested in the x coordinate of the intersection:
         * x = ((x1y2 - y1x2)(x3 - x4) - (x1 - x2)(x3y4 - y3x4)) /
         *     ((x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)).
         * If the denominator is 0, the lines are parallel. We only
         * care about intersections in a specific interval - if 
         * there is none, we mark the element with FLAG_DEL.
         */

        const sigpt_t p1 = this_sig[s.i];
        const sigpt_t p2 = this_sig[s.i + 1]; /* TODO: Range checks! */
        const sigpt_t p3 = other_sig[other_sig_max[i]];
        const sigpt_t p4 = other_sig[other_sig_max[i] + 1]; /* TODO: Range checks! */

        const float denom = (p1.t - p2.t) * (p3.y - p4.y) -
                            (p1.y - p2.y) * (p3.t - p4.t);
        const float numer = (p1.t * p2.y - p1.y * p2.t) * (p3.t - p4.t) -
                            (p1.t - p2.t) * (p3.t * p4.y - p3.y * p4.t);

        /* Lines parallel? */
        if (denom == 0.f) {
            ts[ii].flags |= FLAG_DEL;
            continue; /* TODO: Optimize */
        }

        const float t = numer / denom;

        /* Intersection outside of line segment range? */
        if (t <= p1.t || t >= p2.t || t <= p3.t || t >= p4.t) {
            ts[ii].flags |= FLAG_DEL;
            continue; /* TODO: Optimize */
        }

        ts[ii].t = t;
    }
}

struct seqpt_same_time : public thrust::binary_function<seqpt_t, seqpt_t, bool>
{
    __device__ bool
    operator()(const seqpt_t &lhs, const seqpt_t &rhs) const
    {
        return abs(lhs.t - rhs.t) < FLOAT_DELTA;
    }
};


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

    seqpt_t seqinit = { 0.f, 0, 0, 0, FLAG_LHS };
    thrust::device_vector<seqpt_t> lhst(lhs.size() * 2, seqinit);
    seqpt_t *ptr_lhst = thrust::raw_pointer_cast(lhst.data());

    time_seq_with_proto_intersections<<<NBLOCKS, NTHREADS>>>(ptr_lhs, ptr_lhst, lhs.size());


    const sigpt_t *ptr_rhs = thrust::raw_pointer_cast(rhs.data());

    seqinit.flags = FLAG_RHS;
    thrust::device_vector<seqpt_t> rhst(rhs.size() * 2, seqinit);
    seqpt_t *ptr_rhst = thrust::raw_pointer_cast(rhst.data());

    time_seq_with_proto_intersections<<<NBLOCKS, NTHREADS>>>(ptr_rhs, ptr_rhst, rhs.size());

    /* TODO: To avoid having out-of-order time points later on, we need to remove
     * duplicate elements before inserting proto intersections. */

    /* We then merge both of these vectors. */

    const int nts = lhst.size() + rhst.size();
    thrust::device_vector<seqpt_t> ts(nts, seqinit);

    thrust::merge(lhst.begin(), lhst.end(),
            rhst.begin(), rhst.end(),
            ts.begin(),
            seqpt_less());

    /* Now, for every proto intersection of side s <- { lhs, rhs }, we need to
     * find the index of the element to its immediate left of the opposing side.
     */ 

    seqpt_t *ptr_ts = thrust::raw_pointer_cast(ts.data());

    thrust::device_vector<int> lhs_max(nts, 0);
    int *ptr_lhs_max = thrust::raw_pointer_cast(lhs_max.data());
    extract_i<<<NBLOCKS, NTHREADS>>>(ptr_ts, ptr_lhs_max, nts, FLAG_LHS);

    thrust::inclusive_scan(lhs_max.begin(), lhs_max.end(), lhs_max.begin(),
                           thrust::maximum<int>());

    thrust::device_vector<int> rhs_max(nts, 0);
    int *ptr_rhs_max = thrust::raw_pointer_cast(rhs_max.data());
    extract_i<<<NBLOCKS, NTHREADS>>>(ptr_ts, ptr_rhs_max, nts, FLAG_RHS);

    thrust::inclusive_scan(rhs_max.begin(), rhs_max.end(), rhs_max.begin(),
                           thrust::maximum<int>());

    /* Next, we go through and fill in ISC elements; if there's an intersection
     * we set the time accordingly, and if there isn't, we mark it as DEL.
     * An intersection must always be between the latest (lhs, rhs) and the next such
     * pair.
     */

    calculate_intersection_seqs<<<NBLOCKS, NTHREADS>>>(
            ptr_lhs, ptr_rhs, 
            ptr_lhs_max, ptr_rhs_max,
            ptr_ts, nts);

    /* Then, mark duplicate time values DEL.
     * Finally we compact the array, removing all DEL elements.
     */

    thrust::unique(ts.begin(), ts.end(), seqpt_same_time());

    int i = 0;
    for (thrust::device_vector<seqpt_t>::iterator iter = ts.begin();
         iter != ts.begin() + 40; iter++) {
    	seqpt_t s = *iter;
        int j = lhs_max[i];
        int k = rhs_max[i++];
    	printf("{ %f, %d, %d, %d, %x }\n", s.t, s.i, j, k, s.flags);
    }

    /* We now have the complete time sequence stored in ts, including
     * all points in lhs, rhs, and intersections of the two (what a bitch).
     * Extrapolate the sigpt_t sequence of both signals for each point <- ts.
     */

    sigpt_t sigpt_init = { 0.f, 0.f, 0.f };
    thrust::device_vector<sigpt_t> lhs_extrapolated(ts.size(), sigpt_init);
    thrust::device_vector<sigpt_t> rhs_extrapolated(ts.size(), sigpt_init);

    sigpt_extrapolate<<<NBLOCKS, NTHREADS>>>(
            ptr_lhs,
            ptr_rhs,
            ptr_lhs_max,
            ptr_rhs_max,
            ptr_ts,
            thrust::raw_pointer_cast(lhs_extrapolated.data()),
            thrust::raw_pointer_cast(rhs_extrapolated.data()));

    /* And *finally* run the actual and operator. */

    thrust::transform(lhs_extrapolated.begin(), lhs_extrapolated.end(),
            rhs_extrapolated.begin(),
            out.begin(),
            sigpt_min());
}

struct sigpt_max : public thrust::binary_function<sigpt_t, sigpt_t, sigpt_t>
{
    __device__ sigpt_t
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        /*sigpt_t ret;
        if (lhs.y > rhs.y) {
            ret.y = lhs.y;
        } else {
            ret.y = rhs.y;
        }

        // Keep the time.
        ret.t = rhs.t;

        return ret;*/

        return (sigpt_t) {rhs.t, CUDA_MAX(lhs.y, rhs.y), 0};
    }
};

__global__ void
eventually_intersect(const sigpt_t *ys, sigpt_t *zs, sigpt_t *zs_intersect, char *cs, int n)
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

__global__ void
eventually_compact(const sigpt_t *zs, sigpt_t *zs_final, const char *cs, const size_t *fs, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // FIXME: Branches are bad.
        if (cs[i] == 1) {
            zs_final[fs[i]] = zs[i];
        }
    }
}

/**
 * sizeof(out) = 2 * sizeof(in).
 */
void
stl_eventually(const thrust::device_vector<sigpt_t> &in,
        thrust::device_vector<sigpt_t> &out)
{
    thrust::inclusive_scan(in.crbegin(), in.crend(), out.rbegin() + in.size(), sigpt_max()); 

    const sigpt_t *ys = thrust::raw_pointer_cast(in.data());

    thrust::device_vector<sigpt_t> out_intersect(in.size() * 2);
    sigpt_t *zs = thrust::raw_pointer_cast(out_intersect.data());

    sigpt_t *zs_final = thrust::raw_pointer_cast(out.data());

    thrust::device_vector<char> used(in.size() * 2, 0);
    char *cs = thrust::raw_pointer_cast(used.data());
    eventually_intersect<<<NBLOCKS, NTHREADS>>>(ys, zs_final, zs, cs, in.size());

    thrust::device_vector<size_t> positions(in.size() * 2, 0);
    thrust::exclusive_scan(used.cbegin(), used.cend(), positions.begin(), 0, thrust::plus<size_t>()); 

    size_t *fs = thrust::raw_pointer_cast(positions.data());
    eventually_compact<<<NBLOCKS, NTHREADS>>>(zs, zs_final, cs, fs, in.size() * 2);

    out.resize(positions.back());
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

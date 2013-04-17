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

/* TODO: Handle multiple GPUs. */

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

typedef struct {
    float t;        /**< The time value. */
    int i;          /**< The original index. */
    int assoc_i;    /**< The associated index of the other signal. */
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

/**
 * Given a sequence point t with l.t <= t.t <= r.t,
 * returns an interpolated signal point at time t.t.
 */
__device__ sigpt_t
interpolate(const sigpt_t *l,
            const sigpt_t *r,
            const seqpt_t *t)
{
    const sigpt_t l_reg = *l;
    const sigpt_t r_reg = *r;

    const float dt = r_reg.t - l_reg.t;
    const float dy = r_reg.y - l_reg.y;
    const float dy_normed = dy / dt; /* TODO: Assumes dt != 0.f. */

    sigpt_t sigpt = { t->t, l_reg.y + dy_normed * (t->t - l_reg.t), dy_normed };
    return sigpt;
}

__global__ void
sigpt_extrapolate(const sigpt_t *lhs,
                  const sigpt_t *rhs,
                  const seqpt_t *ts,
                  sigpt_t *lhs_extrapolated,
                  sigpt_t *rhs_extrapolated,
                  int n_lhs,
                  int n_rhs,
                  int n_ts)
{
    /* Use the information provided by lhs, rhs, and ts
     * to extrapolated a signal point sequence for both lhs and rhs for each
     * time point in ts. */

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n_ts; i += blockDim.x * gridDim.x) {
        const seqpt_t seqpt = ts[i];

        const int is_lhs = (seqpt.flags & FLAG_LHS) != 0;
        const int is_rhs = !is_lhs;

        const int assoc_lhs = is_lhs * seqpt.i + is_rhs * seqpt.assoc_i;
        const int assoc_rhs = is_rhs * seqpt.i + is_lhs * seqpt.assoc_i;

        /* TODO: Optimize. */

        if (assoc_lhs >= n_lhs - 1) {
            lhs_extrapolated[i] = (sigpt_t){ seqpt.t, lhs[i].y, lhs[i].dy };
        } else {
            lhs_extrapolated[i] = interpolate(lhs + assoc_lhs,
                                              lhs + assoc_lhs + 1,
                                              &seqpt);
        }

        if (assoc_rhs >= n_rhs - 1) {
            rhs_extrapolated[i] = (sigpt_t){ seqpt.t, rhs[i].y, rhs[i].dy };
        } else {
            rhs_extrapolated[i] = interpolate(rhs + assoc_rhs,
                                              rhs + assoc_rhs + 1,
                                              &seqpt);
        }
    }
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
merge_i(const int *lhs, const int *rhs, seqpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        const int is_lhs = (out[i].flags & FLAG_LHS) != 0;
        const int is_rhs = !is_lhs;

        out[i].assoc_i = is_lhs * rhs[i] + is_rhs * lhs[i];
    }
}

__global__ void
sigpt_to_seqpt(const sigpt_t *in, seqpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i].t = in[i].t;
        out[i].i = i;
    }
}

__global__ void
insert_proto_intersections(const seqpt_t *in, seqpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t seqpt = in[i];
        out[i * 2] = seqpt;

        seqpt.flags |= FLAG_ISC;
        out[i * 2 + 1] = seqpt;
    }
}

__global__ void
calc_intersections(const sigpt_t *lhs,
                   const sigpt_t *rhs,
                   seqpt_t *ts,
                   int n_lhs,
                   int n_rhs,
                   int n_ts)
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

    for (int i = tid; 2 * i + 1 < n_ts; i += blockDim.x * gridDim.x) {
        const int ii = 2 * i + 1;
        seqpt_t s = ts[ii];

        const int is_lhs = (s.flags & FLAG_LHS) != 0;
        const int is_rhs = !is_lhs;

        /* TODO: Optimize. */
        const sigpt_t *this_sig = is_lhs ? lhs : rhs;
        const sigpt_t *other_sig = is_lhs ? rhs : lhs;

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

        if ((is_lhs && (s.i >= n_lhs - 2 || s.assoc_i >= n_rhs - 2)) ||
            (is_rhs && (s.i >= n_rhs - 2 || s.assoc_i >= n_lhs - 2))) {
            continue; /* TODO: Optimize */
        }

        const sigpt_t p1 = this_sig[s.i];
        const sigpt_t p2 = this_sig[s.i + 1];
        const sigpt_t p3 = other_sig[s.assoc_i];
        const sigpt_t p4 = other_sig[s.assoc_i + 1];

        const float denom = (p1.t - p2.t) * (p3.y - p4.y) -
                            (p1.y - p2.y) * (p3.t - p4.t);
        const float numer = (p1.t * p2.y - p1.y * p2.t) * (p3.t - p4.t) -
                            (p1.t - p2.t) * (p3.t * p4.y - p3.y * p4.t);

        /* Lines parallel? */
        if (denom == 0.f) {
            continue; /* TODO: Optimize */
        }

        const float t = numer / denom;

        /* Intersection outside of line segment range? */
        if (t <= p1.t || t >= p2.t || t <= p3.t || t >= p4.t) {
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
    /* A rough outline of the function:
     *
     * Construct a sorted time sequence ts
     * which contains all t <- lhs, all t <- rhs, and all intersection points
     * between lhs and rhs. The sequence contains only unique points.
     * 
     * Using interpolation, construct lhs' and rhs' such that they contain all
     * t <- ts.
     *
     * Finally, do a simple min() over these arrays.
     */

    /* First, extract the time sequences, merge them, and remove duplicates. */

    const sigpt_t *ptr_lhs = thrust::raw_pointer_cast(lhs.data());
    seqpt_t seqinit = { 0.f, 0, 0, FLAG_LHS };
    thrust::device_vector<seqpt_t> lhs_ts(lhs.size(), seqinit);
    seqpt_t *ptr_lhs_ts = thrust::raw_pointer_cast(lhs_ts.data());
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(ptr_lhs, ptr_lhs_ts, lhs.size());

    const sigpt_t *ptr_rhs = thrust::raw_pointer_cast(rhs.data());
    seqinit.flags = FLAG_RHS;
    thrust::device_vector<seqpt_t> rhs_ts(rhs.size(), seqinit);
    seqpt_t *ptr_rhs_ts = thrust::raw_pointer_cast(rhs_ts.data());
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(ptr_rhs, ptr_rhs_ts, rhs.size());

    thrust::device_vector<seqpt_t> ts(lhs_ts.size() + rhs_ts.size(), seqinit);
    thrust::merge(lhs_ts.begin(), lhs_ts.end(), rhs_ts.begin(), rhs_ts.end(),
                  ts.begin(), seqpt_less());

    thrust::device_vector<seqpt_t>::iterator ts_end =
        thrust::unique(ts.begin(), ts.end(), seqpt_same_time());
    int ts_size = ts_end - ts.begin();

    /* Add a proto-intersection after each point in the resulting sequence. */

    seqinit.flags = 0;
    const seqpt_t *ptr_ts = thrust::raw_pointer_cast(ts.data());
    thrust::device_vector<seqpt_t> tsi(ts_size * 2, seqinit);
    seqpt_t *ptr_tsi = thrust::raw_pointer_cast(tsi.data());
    insert_proto_intersections<<<NBLOCKS, NTHREADS>>>(ptr_ts, ptr_tsi, ts_size);

    /* Now, for every proto-intersection of side s <- { lhs, rhs }, we need to
     * find the index of the element to its immediate left of the opposing side.
     * We do this by first extracting the indices of each side to an array,
     * running a max() scan over it, and finally merging these arrays back into
     * seqpt_t.assoc_i.
     */ 

    thrust::device_vector<int> lhs_i_max(tsi.size(), 0);
    int *ptr_lhs_i_max = thrust::raw_pointer_cast(lhs_i_max.data());
    extract_i<<<NBLOCKS, NTHREADS>>>(ptr_tsi, ptr_lhs_i_max, tsi.size(), FLAG_LHS);

    thrust::inclusive_scan(lhs_i_max.begin(), lhs_i_max.end(), lhs_i_max.begin(),
                           thrust::maximum<int>());

    thrust::device_vector<int> rhs_i_max(tsi.size(), 0);
    int *ptr_rhs_i_max = thrust::raw_pointer_cast(rhs_i_max.data());
    extract_i<<<NBLOCKS, NTHREADS>>>(ptr_tsi, ptr_rhs_i_max, tsi.size(), FLAG_RHS);

    thrust::inclusive_scan(rhs_i_max.begin(), rhs_i_max.end(), rhs_i_max.begin(),
                           thrust::maximum<int>());

    merge_i<<<NBLOCKS, NTHREADS>>>(ptr_lhs_i_max, ptr_rhs_i_max, ptr_tsi, tsi.size());

    /* Next, we go through and fill in ISC elements; if there's an intersection
     * we set the time accordingly, and if there isn't, we mark it as DEL.
     * An intersection must always be between the latest (lhs, rhs) and the next such
     * pair.
     */

    calc_intersections<<<NBLOCKS, NTHREADS>>>(ptr_lhs, ptr_rhs, ptr_tsi,
                                              lhs.size(), rhs.size(), tsi.size());

    /* Finally we again remove all duplicate elements (= all proto-intersections
     * which did not turn out to actually be real intersections).
     */

    thrust::device_vector<seqpt_t>::iterator tsi_end =
        thrust::unique(tsi.begin(), tsi.end(), seqpt_same_time());
    int tsi_size = ts_end - ts.begin();

    /* We now have the complete time sequence stored in ts, including
     * all points in lhs, rhs, and intersections of the two (what a bitch).
     * Extrapolate the sigpt_t sequence of both signals for each point <- ts.
     */

    sigpt_t sigpt_init = { 0.f, 0.f, 0.f };
    thrust::device_vector<sigpt_t> lhs_extrapolated(tsi_size, sigpt_init);
    thrust::device_vector<sigpt_t> rhs_extrapolated(tsi_size, sigpt_init);
    sigpt_extrapolate<<<NBLOCKS, NTHREADS>>>(ptr_lhs, ptr_rhs, ptr_tsi,
            thrust::raw_pointer_cast(lhs_extrapolated.data()),
            thrust::raw_pointer_cast(rhs_extrapolated.data()),
            lhs.size(), rhs.size(), tsi_size);

    /* And *finally* run the actual and operator. */

    thrust::transform(lhs_extrapolated.begin(), lhs_extrapolated.end(),
            rhs_extrapolated.begin(),
            out.begin(),
            sigpt_min());

    /* TODO: Instead of allocating all of these device vectors between 
     * kernel calls, try to be a bit smarter about it. For example,
     * we could queue the allocations on a separate stream. */

    printf("lhs (%d):\n", lhs.size());
    for (int i = 0; i < 10; i++) {
        sigpt_t sigpt = lhs[i];
    	printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\nrhs (%d):\n", rhs.size());
    for (int i = 0; i < 10; i++) {
        sigpt_t sigpt = rhs[i];
    	printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\ntsi (%d):\n", tsi_size);
    for (int i = 0; i < 10; i++) {
    	seqpt_t s = tsi[i];
    	printf("{ %f, %d, %d, %x }\n", s.t, s.i, s.assoc_i, s.flags);
    }

    printf("\nlhs_extrapolated (%d):\n", lhs_extrapolated.size());
    for (int i = 0; i < 10; i++) {
        sigpt_t sigpt = lhs_extrapolated[i];
    	printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\nrhs_extrapolated (%d):\n", rhs_extrapolated.size());
    for (int i = 0; i < 10; i++) {
        sigpt_t sigpt = rhs_extrapolated[i];
    	printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }
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

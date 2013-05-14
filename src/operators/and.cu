#include "and.hpp"

#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "globals.h"

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
sigpt_to_seqpt(const sigpt_t *in, seqpt_t *out, int n, int flags)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t seqpt = { in[i].t, i, 0, flags };
        out[i] = seqpt;
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

        if ((is_lhs && (s.i > n_lhs - 2 || s.assoc_i > n_rhs - 2)) ||
            (is_rhs && (s.i > n_rhs - 2 || s.assoc_i > n_lhs - 2))) {
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
 * sizeof(out) = 4 * max(sizeof(lhs), sizeof(rhs)).
 */
void
stl_and(const thrust::device_ptr<sigpt_t> &lhs,
        const int nlhs,
        const thrust::device_ptr<sigpt_t> &rhs,
        const int nrhs,
        thrust::device_ptr<sigpt_t> *out,
        int *nout)
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

    /* First, extract the time sequences, merge them. */

    thrust::device_ptr<seqpt_t> lhs_ts = thrust::device_malloc<seqpt_t>(nlhs);
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(lhs.get(), lhs_ts.get(), nlhs, FLAG_LHS);

    thrust::device_ptr<seqpt_t> rhs_ts = thrust::device_malloc<seqpt_t>(nrhs);
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(rhs.get(), rhs_ts.get(), nrhs, FLAG_RHS);

    int n_ts = nrhs + nlhs;
    thrust::device_ptr<seqpt_t> ts = thrust::device_malloc<seqpt_t>(n_ts);
    thrust::merge(lhs_ts, lhs_ts + nrhs, rhs_ts, rhs_ts + nrhs,
                  ts, seqpt_less());

    /* Now, for every proto-intersection of side s <- { lhs, rhs }, we need to
     * find the index of the element to its immediate left of the opposing side.
     * We do this by first extracting the indices of each side to an array,
     * running a max() scan over it, and finally merging these arrays back into
     * seqpt_t.assoc_i.
     */ 

    thrust::device_ptr<int> lhs_i_max = thrust::device_malloc<int>(n_ts);
    extract_i<<<NBLOCKS, NTHREADS>>>(ts.get(), lhs_i_max.get(), n_ts, FLAG_LHS);

    thrust::inclusive_scan(lhs_i_max, lhs_i_max + n_ts, lhs_i_max,
                           thrust::maximum<int>());

    thrust::device_ptr<int> rhs_i_max = thrust::device_malloc<int>(n_ts);
    extract_i<<<NBLOCKS, NTHREADS>>>(ts.get(), rhs_i_max.get(), n_ts, FLAG_RHS);

    thrust::inclusive_scan(rhs_i_max, rhs_i_max + n_ts, rhs_i_max,
                           thrust::maximum<int>());

    merge_i<<<NBLOCKS, NTHREADS>>>(lhs_i_max.get(), rhs_i_max.get(), ts.get(), n_ts);

    /* Remove duplicates. */

    thrust::device_ptr<seqpt_t> ts_end =
        thrust::unique(ts, ts + n_ts, seqpt_same_time());
    n_ts = ts_end - ts;

    /* Add a proto-intersection after each point in the resulting sequence. */

    int n_tsi = n_ts * 2;
    thrust::device_ptr<seqpt_t> tsi = thrust::device_malloc<seqpt_t>(n_tsi);
    insert_proto_intersections<<<NBLOCKS, NTHREADS>>>(ts.get(), tsi.get(), n_ts);

    /* Next, we go through and fill in ISC elements; if there's an intersection
     * we set the time accordingly, and if there isn't, we mark it as DEL.
     * An intersection must always be between the latest (lhs, rhs) and the next such
     * pair.
     */

    calc_intersections<<<NBLOCKS, NTHREADS>>>(lhs.get(), rhs.get(), tsi.get(),
                                              nlhs, nrhs, n_tsi);

    /* Finally we again remove all duplicate elements (= all proto-intersections
     * which did not turn out to actually be real intersections).
     */

    thrust::device_ptr<seqpt_t> tsi_end =
        thrust::unique(tsi, tsi + n_tsi, seqpt_same_time());
    n_tsi = tsi_end - tsi;

    /* We now have the complete time sequence stored in ts, including
     * all points in lhs, rhs, and intersections of the two (what a bitch).
     * Extrapolate the sigpt_t sequence of both signals for each point <- ts.
     */

    thrust::device_ptr<sigpt_t> lhs_extrapolated = thrust::device_malloc<sigpt_t>(n_tsi);
    thrust::device_ptr<sigpt_t> rhs_extrapolated = thrust::device_malloc<sigpt_t>(n_tsi);
    sigpt_extrapolate<<<NBLOCKS, NTHREADS>>>(lhs.get(), rhs.get(), tsi.get(),
                                             lhs_extrapolated.get(), rhs_extrapolated.get(),
                                             nlhs, nrhs, n_tsi);

    /* And *finally* run the actual and operator. */

    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(n_tsi);
    thrust::transform(lhs_extrapolated, lhs_extrapolated + n_tsi,
                      rhs_extrapolated, dout, sigpt_min());

    *out = dout;
    *nout = n_tsi;

    /* TODO: Instead of allocating all of these device vectors between 
     * kernel calls, try to be a bit smarter about it. For example,
     * we could queue the allocations on a separate stream. */

/*    printf("lhs (%d):\n", nlhs);
    for (int i = 0; i < nlhs && i < 10; i++) {
        sigpt_t sigpt = lhs[i];
        printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\nrhs (%d):\n", nrhs);
    for (int i = 0; i < nrhs && i < 10; i++) {
        sigpt_t sigpt = rhs[i];
        printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\ntsi (%d):\n", n_tsi);
    for (int i = 0; i < n_tsi && i < 10; i++) {
        seqpt_t s = tsi[i];
        printf("{ %f, %d, %d, %x }\n", s.t, s.i, s.assoc_i, s.flags);
    }

    printf("\nlhs_extrapolated (%d):\n", n_tsi);
    for (int i = 0; i < n_tsi && i < 10; i++) {
        sigpt_t sigpt = lhs_extrapolated[i];
        printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\nrhs_extrapolated (%d):\n", n_tsi);
    for (int i = 0; i < n_tsi && i < 10; i++) {
        sigpt_t sigpt = rhs_extrapolated[i];
        printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }

    printf("\nresult (%d):\n", out.size());
    for (int i = 0; i < out.size() && i < 10; i++) {
        sigpt_t sigpt = out[i];
        printf("%i: {%f, %f, %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }
*/
    thrust::device_free(lhs_ts);
    thrust::device_free(rhs_ts);
    thrust::device_free(ts);
    thrust::device_free(tsi);
    thrust::device_free(lhs_i_max);
    thrust::device_free(rhs_i_max);
    thrust::device_free(lhs_extrapolated);
    thrust::device_free(rhs_extrapolated);
}

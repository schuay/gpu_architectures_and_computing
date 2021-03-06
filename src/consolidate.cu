#include "consolidate.hpp"

#include <math.h>
#include <thrust/functional.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include "globals.h"
#include "interpolate.hpp"

#define FLAG_LHS (1 << 0)
#define FLAG_RHS (1 << 1)
#define FLAG_ISC (1 << 2)

typedef struct {
    float t;        /**< The time value. */
    int i;          /**< The original index. */
    int ilhs, irhs; /**< The indices of the last elements in lhs, rhs with t' <= t. */
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

__global__ static void
sigpt_extrapolate(const sigpt_t *lhs,
                  const sigpt_t *rhs,
                  const seqpt_t *ts,
                  sigpt_t *clhs,
                  sigpt_t *crhs,
                  const int n_lhs,
                  const int n_rhs,
                  const int n_ts)
{
    /* Use the information provided by lhs, rhs, and ts
     * to extrapolated a signal point sequence for both lhs and rhs for each
     * time point in ts. */

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n_ts; i += blockDim.x * gridDim.x) {
        const seqpt_t seqpt = ts[i];

        const int ilhs = seqpt.ilhs;
        const int irhs = seqpt.irhs;

        /* TODO: Optimize. */

        if (ilhs >= n_lhs - 1) {
            clhs[i] = (sigpt_t){ seqpt.t, lhs[ilhs].y, lhs[ilhs].dy };
        } else {
            clhs[i] = interpolate(lhs + ilhs, lhs + ilhs + 1, seqpt.t);
        }

        if (irhs >= n_rhs - 1) {
            crhs[i] = (sigpt_t){ seqpt.t, rhs[irhs].y, rhs[irhs].dy };
        } else {
            crhs[i] = interpolate(rhs + irhs, rhs + irhs + 1, seqpt.t);
        }
    }
}

__global__ static void
extract_i(const seqpt_t *in,
          int *out,
          const int n,
          const int flag)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t s = in[i];
        const int has_flag = (s.flags & flag) != 0;
        out[i] = has_flag * s.i;
    }
}

__global__ static void
merge_i(const int *lhs,
        const int *rhs,
        seqpt_t *out,
        const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        out[i].ilhs = lhs[i];
        out[i].irhs = rhs[i];
    }
}

__global__ static void
sigpt_to_seqpt(const sigpt_t *in,
               seqpt_t *out,
               const int n,
               const int flags)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t seqpt = { in[i].t, i, 0, 0, flags };
        out[i] = seqpt;
    }
}

__global__ static void
insert_proto_intersections(const seqpt_t *in,
                           seqpt_t *out,
                           const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        seqpt_t seqpt = in[i];
        out[i * 2] = seqpt;

        seqpt.flags |= FLAG_ISC;
        out[i * 2 + 1] = seqpt;
    }
}

__global__ static void
calc_intersections(const sigpt_t *lhs,
                   const sigpt_t *rhs,
                   seqpt_t *ts,
                   const int n_lhs,
                   const int n_rhs,
                   const int n_ts)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* At this point, we are only interested in intersection elements in ts.
     * These are located at every index 2 * i + 1, i <- n_ts / 2.
     *
     * ts[i].ilhs is the index of the last point in lhs with t <= ts[i].t,
     * and ts[i].irhs is the corresponding point in rhs.
     *
     * This is enough information to determine the time of the signal
     * intersection.
     */

    for (int i = tid; 2 * i + 1 < n_ts; i += blockDim.x * gridDim.x) {
        const int ii = 2 * i + 1;
        const seqpt_t s = ts[ii];

        /* We now have four points corresponding to the end points of the
         * two line segments. (x1, y1) and (x2, y2) for one line segment,
         * (x3, y3) and (x4, y4) for the other line segment.
         * We are interested in the x coordinate of the intersection:
         * x = ((x1y2 - y1x2)(x3 - x4) - (x1 - x2)(x3y4 - y3x4)) /
         *     ((x1 - x2)(y3 - y4) - (y1 - y2)(x3 - x4)).
         * If the denominator is 0, the lines are parallel. We only
         * care about intersections in a specific interval.
         */

        if (s.ilhs > n_lhs - 2 || s.irhs > n_rhs - 2) {
            continue; /* TODO: Optimize */
        }

        const sigpt_t p1 = lhs[s.ilhs];
        const sigpt_t p2 = lhs[s.ilhs + 1];
        const sigpt_t p3 = rhs[s.irhs];
        const sigpt_t p4 = rhs[s.irhs + 1];

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
        return fabsf(lhs.t - rhs.t) < FLOAT_DELTA;
    }
};

void
consolidate(const thrust::device_ptr<sigpt_t> &lhs,
            const int nlhs,
            const thrust::device_ptr<sigpt_t> &rhs,
            const int nrhs,
            thrust::device_ptr<sigpt_t> *olhs,
            thrust::device_ptr<sigpt_t> *orhs,
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
     */

    /* First, extract the time sequences and merge them. */

    thrust::device_ptr<seqpt_t> lhs_ts = thrust::device_malloc<seqpt_t>(nlhs);
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(lhs.get(), lhs_ts.get(), nlhs, FLAG_LHS);

    thrust::device_ptr<seqpt_t> rhs_ts = thrust::device_malloc<seqpt_t>(nrhs);
    sigpt_to_seqpt<<<NBLOCKS, NTHREADS>>>(rhs.get(), rhs_ts.get(), nrhs, FLAG_RHS);

    int n_ts = nrhs + nlhs;
    thrust::device_ptr<seqpt_t> ts = thrust::device_malloc<seqpt_t>(n_ts);
    thrust::merge(lhs_ts, lhs_ts + nrhs, rhs_ts, rhs_ts + nrhs, ts, seqpt_less());

    thrust::device_free(lhs_ts);
    thrust::device_free(rhs_ts);

    /* Associate every sequence point t <- ts with the latest element of the other signal
     * that satisfies t' <= t. For example, if the current t has FLAG_LHS and t = 3.5,
     * we associate it with the latest point p <- rhs such that p.t <= 3.5.
     *
     * We do this by first extracting the indices of each side to an array,
     * running a max() scan over it, and finally merging these arrays back into
     * seqpt_t.ilhs and seqpt_t.irhs.
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

    thrust::device_free(lhs_i_max);
    thrust::device_free(rhs_i_max);

    /* Remove duplicates. Again, this is less trivial than it looks at first. If
     * we need to keep the *last* element of each run of equal elements in order
     * to pick up the correct associated index at points in time where both LHS
     * and RHS are defined. Therefore, run unique() with a reverse iterator
     * and use some pointer arithmetic to address the compacted range afterwards.
     */

    thrust::reverse_iterator<thrust::device_ptr<seqpt_t> > rts(ts + n_ts);
    thrust::reverse_iterator<thrust::device_ptr<seqpt_t> > rts_end =
        thrust::unique(rts, rts + n_ts, seqpt_same_time());
    const int n_rts = rts_end - rts;

    /* Add a proto-intersection after each point in the resulting sequence. */

    int n_tsi = n_rts * 2;
    thrust::device_ptr<seqpt_t> tsi = thrust::device_malloc<seqpt_t>(n_tsi);
    insert_proto_intersections<<<NBLOCKS, NTHREADS>>>(ts.get() + n_ts - n_rts, tsi.get(), n_rts);

    thrust::device_free(ts);

    /* Next, we go through and fill in ISC elements; if there's an intersection
     * we set the time accordingly.
     */

    calc_intersections<<<NBLOCKS, NTHREADS>>>(lhs.get(), rhs.get(), tsi.get(),
                                              nlhs, nrhs, n_tsi);

    /* Finally we again remove all duplicate elements (= all proto-intersections
     * which did not turn out to actually be real intersections).
     */

    thrust::device_ptr<seqpt_t> tsi_end =
        thrust::unique(tsi, tsi + n_tsi, seqpt_same_time());
    n_tsi = tsi_end - tsi;

    /* We now have the complete time sequence stored in tsi, including
     * all points in lhs, rhs, and intersections of the two (what a bitch).
     * Extrapolate the sigpt_t sequence of both signals for each point <- tsi.
     */

    thrust::device_ptr<sigpt_t> lhs_extrapolated = thrust::device_malloc<sigpt_t>(n_tsi);
    thrust::device_ptr<sigpt_t> rhs_extrapolated = thrust::device_malloc<sigpt_t>(n_tsi);
    sigpt_extrapolate<<<NBLOCKS, NTHREADS>>>(lhs.get(), rhs.get(), tsi.get(),
                                             lhs_extrapolated.get(), rhs_extrapolated.get(),
                                             nlhs, nrhs, n_tsi);

    *olhs = lhs_extrapolated;
    *orhs = rhs_extrapolated;
    *nout = n_tsi;

    /* TODO: Instead of allocating all of these device vectors between
     * kernel calls, try to be a bit smarter about it. For example,
     * we could queue the allocations on a separate stream. */

    thrust::device_free(tsi);
}

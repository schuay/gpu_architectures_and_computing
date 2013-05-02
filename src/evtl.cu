#include "evtl.hpp"

#include <thrust/scan.h>

#include "globals.h"

#define CUDA_MAX(a, b) (((a) > (b)) * (a) + ((a) <= (b)) * (b))

struct sigpt_max : public thrust::binary_function<sigpt_t, sigpt_t, sigpt_t>
{
    __device__ sigpt_t
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
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

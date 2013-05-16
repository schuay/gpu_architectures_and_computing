#include "and.hpp"

#include "consolidate.hpp"

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

void
stl_and(const thrust::device_ptr<sigpt_t> &lhs,
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

    thrust::device_ptr<sigpt_t> dout = thrust::device_malloc<sigpt_t>(nc);
    thrust::transform(clhs, clhs + nc, crhs, dout, sigpt_min());

    thrust::device_free(clhs);
    thrust::device_free(crhs);

    *out = dout;
    *nout = nc;
}

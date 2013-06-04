#include "buntil.hpp"

#include "and.hpp"
#include "balw.hpp"
#include "bevtl.hpp"
#include "until.hpp"

void
stl_buntil(const thrust::device_ptr<sigpt_t> &lhs,
           const int nlhs,
           const thrust::device_ptr<sigpt_t> &rhs,
           const int nrhs,
           const float s,
           const float t,
           thrust::device_ptr<sigpt_t> *out,
           int *nout)
{
    /* l UNTIL[s, t] r = ALW[0, s] l AND (EVTL[s, t] r AND (EVTL[s, s] (l UNTIL r))).
     *   x6              x5          x6   x3           x4   x2            x1
     */

    thrust::device_ptr<sigpt_t> x1;
    int nx1;
    stl_until(lhs, nlhs, rhs, nrhs, &x1, &nx1);

    thrust::device_ptr<sigpt_t> x2;
    int nx2;
    stl_bevtl(x1, nx1, s, s, &x2, &nx2);
    thrust::device_free(x1);

    thrust::device_ptr<sigpt_t> x3;
    int nx3;
    stl_bevtl(rhs, nrhs, s, t, &x3, &nx3);

    thrust::device_ptr<sigpt_t> x4;
    int nx4;
    stl_and(x3, nx3, x2, nx2, &x4, &nx4);
    thrust::device_free(x2);
    thrust::device_free(x3);

    thrust::device_ptr<sigpt_t> x5;
    int nx5;
    stl_balw(lhs, nlhs, 0, s, &x5, &nx5);

    thrust::device_ptr<sigpt_t> x6;
    int nx6;
    stl_and(x5, nx5, x4, nx4, &x6, &nx6);
    thrust::device_free(x4);
    thrust::device_free(x5);

    *out = x6;
    *nout = nx6;
}

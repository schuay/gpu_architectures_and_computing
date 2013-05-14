#include "or.hpp"

#include "and.hpp"
#include "not.hpp"

void
stl_or(const thrust::device_ptr<sigpt_t> &lhs,
       const int nlhs,
       const thrust::device_ptr<sigpt_t> &rhs,
       const int nrhs,
       thrust::device_ptr<sigpt_t> *out,
       int *nout)
{
    /* l OR r = NOT (NOT l AND NOT r). */

    thrust::device_ptr<sigpt_t> not_lhs;
    int nnot_lhs;

    stl_not(lhs, nlhs, &not_lhs, &nnot_lhs);

    thrust::device_ptr<sigpt_t> not_rhs;
    int nnot_rhs;

    stl_not(rhs, nrhs, &not_rhs, &nnot_rhs);

    thrust::device_ptr<sigpt_t> pand;
    int npand;

    stl_and(not_lhs, nnot_lhs,
            not_rhs, nnot_rhs,
            &pand, &npand);

    thrust::device_ptr<sigpt_t> not_and;
    int nnot_and;

    stl_not(pand, npand, &not_and, &nnot_and);

    thrust::device_free(not_lhs);
    thrust::device_free(not_rhs);
    thrust::device_free(pand);

    *out = not_and;
    *nout = nnot_and;
}


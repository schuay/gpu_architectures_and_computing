#include "balw.hpp"

#include "bevtl.hpp"
#include "not.hpp"

void
stl_balw(const thrust::device_ptr<sigpt_t> &in,
         const int nin,
         const int s,
         const int t,
         thrust::device_ptr<sigpt_t> *out,
         int *nout)
{
    /* ALW s = NOT EVTL (NOT s). */

    thrust::device_ptr<sigpt_t> not_in;
    int nnot_in;

    stl_not(in, nin, &not_in, &nnot_in);

    thrust::device_ptr<sigpt_t> pevtl;
    int npevtl;

    stl_bevtl(not_in, nnot_in, s, t, &pevtl, &npevtl);

    thrust::device_ptr<sigpt_t> not_evtl;
    int nnot_evtl;

    stl_not(pevtl, npevtl, &not_evtl, &nnot_evtl);

    thrust::device_free(not_in);
    thrust::device_free(pevtl);

    *out = not_evtl;
    *nout = nnot_evtl;
}

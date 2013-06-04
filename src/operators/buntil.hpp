#ifndef __BUNTIL_H
#define __BUNTIL_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_buntil(const thrust::device_ptr<sigpt_t> &lhs,
           const int nlhs,
           const thrust::device_ptr<sigpt_t> &rhs,
           const int nrhs,
           const float s,
           const float t,
           thrust::device_ptr<sigpt_t> *out,
           int *nout);

#endif /* __BUNTIL_H */

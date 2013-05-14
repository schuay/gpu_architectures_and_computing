#ifndef __AND_H
#define __AND_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_and(const thrust::device_ptr<sigpt_t> &lhs,
        const int nlhs,
        const thrust::device_ptr<sigpt_t> &rhs,
        const int nrhs,
        thrust::device_ptr<sigpt_t> *out,
        int *nout);

#endif /* __AND_H */

#ifndef __AND_H
#define __AND_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_and(const thrust::device_vector<sigpt_t> &lhs,
        const thrust::device_vector<sigpt_t> &rhs,
        thrust::device_vector<sigpt_t> &out);

#endif /* __AND_H */

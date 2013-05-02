#ifndef __STL_H
#define __STL_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_and(const thrust::device_vector<sigpt_t> &lhs,
        const thrust::device_vector<sigpt_t> &rhs,
        thrust::device_vector<sigpt_t> &out);

#endif /* __STL_H */

#ifndef __NOT_H
#define __NOT_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_not(const thrust::device_ptr<sigpt_t> &in,
        const int nin,
        thrust::device_ptr<sigpt_t> *out,
        int *nout);

#endif /* __NOT_H */

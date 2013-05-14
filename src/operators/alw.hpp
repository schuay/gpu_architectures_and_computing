#ifndef __ALW_H
#define __ALW_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_alw(const thrust::device_ptr<sigpt_t> &in,
        const int nin,
        thrust::device_ptr<sigpt_t> *out,
        int *nout);

#endif /* __ALW_H */

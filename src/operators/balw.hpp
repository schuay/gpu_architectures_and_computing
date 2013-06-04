#ifndef __BALW_H
#define __BALW_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_balw(const thrust::device_ptr<sigpt_t> &in,
         const int nin,
         const float s,
         const float t,
         thrust::device_ptr<sigpt_t> *out,
         int *nout);

#endif /* __BALW_H */

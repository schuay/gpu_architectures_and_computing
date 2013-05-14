#ifndef __EVTL_H
#define __EVTL_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_evtl(const thrust::device_ptr<sigpt_t> &in,
         const int nin,
         thrust::device_ptr<sigpt_t> *out,
         int *nout);

#endif /* __EVTL_H */

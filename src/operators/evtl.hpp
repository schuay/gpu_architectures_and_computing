#ifndef __EVTL_H
#define __EVTL_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_eventually(const thrust::device_vector<sigpt_t> &in,
               thrust::device_vector<sigpt_t> &out);

#endif /* __EVTL_H */

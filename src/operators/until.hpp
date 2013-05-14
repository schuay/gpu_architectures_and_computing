#ifndef __UNTIL_H
#define __UNTIL_H

#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
}

void
stl_until(const thrust::device_ptr<sigpt_t> &lhs,
          const int nlhs,
          const thrust::device_ptr<sigpt_t> &rhs,
          const int nrhs,
          thrust::device_ptr<sigpt_t> *out,
          int *nout);

#endif /* __UNTIL_H */

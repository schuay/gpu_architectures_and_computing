#ifndef __CONSOLIDATE_H
#define __CONSOLIDATE_H

#include <thrust/device_ptr.h>

extern "C" {
#include "sigpt.h"
}

/**
 * Given two incoming signals lhs and rhs, consolidate
 * builds a merged time sequence containing all points
 * of both signals plus all intersection points between both signals.
 * The output signals are constructed using interpolation and
 * contain all of these time points.
 */
void
consolidate(const thrust::device_ptr<sigpt_t> &lhs,
            const int nlhs,
            const thrust::device_ptr<sigpt_t> &rhs,
            const int nrhs,
            thrust::device_ptr<sigpt_t> *olhs,
            thrust::device_ptr<sigpt_t> *orhs,
            int *nout);

#endif /* __CONSOLIDATE_H */

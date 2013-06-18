#ifndef __SIGCMP_H
#define __SIGCMP_H

#include "sigpt.h"

int
sigcmp(const sigpt_t *lhs,
       const int nlhs,
       const sigpt_t *rhs,
       const int nrhs);

/* We need these also in the bevtl test. */
#define FLOAT_ROUGHLY_EQUALS(x, y) (fabs((x) - (y)) < 5e-5f)
#define FLOAT_EQUALS(x, y) (fabs((x) - (y)) < 5e-5)

#endif /* __SIGCMP_H */

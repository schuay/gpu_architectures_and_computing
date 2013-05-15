#ifndef __SIGCMP_H
#define __SIGCMP_H

#include "sigpt.h"

int
sigcmp(const sigpt_t *lhs,
       const int nlhs,
       const sigpt_t *rhs,
       const int nrhs);

#endif /* __SIGCMP_H */

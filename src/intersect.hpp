#ifndef __INTERSECT_H
#define __INTERSECT_H

extern "C" {
#include "sigpt.h"

#include <math.h>
}

#define TIME_FLOAT_DELTA (1e-6f)
#define TIME_EQUALS(x, y) (fabs((x) - (y)) < TIME_FLOAT_DELTA)

static
__host__ __device__ int
intersect(const sigpt_t *l1,
          const sigpt_t *r1,
          const sigpt_t *l2,
          const sigpt_t *r2,
          float *out)
{
    const float denom = (l1->t - l2->t) * (r1->y - r2->y) -
                        (l1->y - l2->y) * (r1->t - r2->t);
    const float numer = (l1->t * l2->y - l1->y * l2->t) * (r1->t - r2->t) -
                        (l1->t - l2->t) * (r1->t * r2->y - r1->y * r2->t);

    /* Lines parallel? */
    if (denom == 0.f) {
        return 0; /* TODO: Optimize */
    }

    const float t = numer / denom;

    if (isnan(t)) {
        return 0;
    }

    if (TIME_EQUALS(l1->t, t) || TIME_EQUALS(l2->t, t)) {
        return 0;
    }

    /* Intersection outside of line segment range? */
    if (t <= l1->t || t >= l2->t || t <= r1->t || t >= r2->t) {
        return 0; /* TODO: Optimize */
    }

    *out = t;

    return 1;
}

#endif /* __INTERSECT_H */

#include "sigcmp.hpp"

#include <stdio.h>
#include <math.h>

#include "interpolate.hpp"

#define FLOAT_ROUGHLY_EQUALS(x, y) (fabs((x) - (y)) < 5e-5f)
#define FLOAT_EQUALS(x, y) (fabs((x) - (y)) < 5e-5)

int
sigcmp(const sigpt_t *lhs,
       const int nlhs,
       const sigpt_t *rhs,
       const int nrhs)
{
    int i = 0, j = 0;
    const sigpt_t *l, *r;

    if (!FLOAT_EQUALS(lhs[0].t, rhs[0].t)) {
        fprintf(stderr, "Different signal start points. lhs: %f, rhs: %f\n",
                lhs[0].t, rhs[0].t);
        return -1;
    }

    if (!FLOAT_EQUALS(lhs[nlhs - 1].t, rhs[nrhs - 1].t)) {
        fprintf(stderr, "Different signal end points. lhs: %f, rhs: %f\n",
                lhs[nlhs - 1].t, rhs[nrhs - 1].t);
        return -1;
    }

    while (i < nlhs && j < nrhs) {
        sigpt_t interpolated;

        if (FLOAT_EQUALS(lhs[i].t, rhs[j].t)) {
            l = lhs + i;
            r = rhs + j;
        } else if (lhs[i].t < rhs[j].t) {
            /* Interpolate rhs. */
            interpolated = interpolate(rhs + j - 1, rhs + j, lhs[i].t);
            l = lhs + i;
            r = &interpolated;
        } else {
            /* Interpolate lhs. */
            interpolated = interpolate(lhs + i - 1, lhs + i, rhs[j].t);
            l = &interpolated;
            r = rhs + j;
        }

        /* Note: dy is ignored for now because it's missing in the breach signal traces. */
        if (!FLOAT_ROUGHLY_EQUALS(l->t, r->t) ||
                !FLOAT_ROUGHLY_EQUALS(l->y, r->y)) {
            fprintf(stderr, "lhs[%d]%s: { t: %f, y: %f, dy: %f } != "
                            "rhs[%d]%s: { t: %f, y: %f, dy: %f }\n",
                    i, (l == &interpolated) ? " interpolated" : "" , l->t, l->y, l->dy,
                    j, (r == &interpolated) ? " interpolated" : "" , r->t, r->y, r->dy);
            return -1;
        }

        if (FLOAT_EQUALS(lhs[i].t, rhs[j].t)) {
            i++;
            j++;
        } else if (lhs[i].t < rhs[j].t) {
            i++;
        } else {
            j++;
        }
    }

    return 0;
}

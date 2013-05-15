#include "sigcmp.hpp"

#include <stdio.h>
#include <math.h>

#include "interpolate.hpp"

#define FLOAT_EQUALS(x, y) (fabs((x) - (y)) < 0.05)

int
sigcmp(const sigpt_t *lhs,
       const sigpt_t *rhs,
       const int n)
{
    int i = 0, j = 0;
    const sigpt_t *l, *r;

    /* TODO: One signal shorter than other. */

    while (i < n && j < n) {
        l = lhs + i;
        r = rhs + j;

        /* Note: dy is ignored for now because it's missing in the breach signal traces. */
        if (!FLOAT_EQUALS(l->t, r->t) ||
                !FLOAT_EQUALS(l->y, r->y)) {
            fprintf(stderr, "lhs[%d]: { t: %f, y: %f, dy: %f } != "
                            "rhs[%d]: { t: %f, y: %f, dy: %f }\n",
                    i, l->t, l->y, l->dy,
                    j, r->t, r->y, r->dy);
            return -1;
        }

        do {
            i++;
        } while (i < n && FLOAT_EQUALS(l->y, lhs[i].y));

        do {
            j++;
        } while (j < n && FLOAT_EQUALS(r->y, rhs[j].y));

    }

    return 0;
}

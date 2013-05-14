#ifndef __INTERPOLATE_H
#define __INTERPOLATE_H

extern "C" {
#include "sigpt.h"
}

static
__host__ __device__ sigpt_t
interpolate(const sigpt_t *l,
            const sigpt_t *r,
            const float t)
{
    const sigpt_t l_reg = *l;
    const sigpt_t r_reg = *r;

    const float dt = r_reg.t - l_reg.t;
    const float dy = r_reg.y - l_reg.y;
    const float dy_normed = dy / dt; /* TODO: Assumes dt != 0.f. */

    sigpt_t sigpt = { t, l_reg.y + dy_normed * (t - l_reg.t), dy_normed };
    return sigpt;
}

#endif /* __INTERPOLATE_H */

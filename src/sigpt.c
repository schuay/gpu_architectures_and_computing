#include "sigpt.h"

#include <stdio.h>
#include <stdlib.h>

sigpt_t *
sigpt_random(const int seed,
             const int n)
{
    sigpt_t *buffer = (sigpt_t *)malloc(sizeof(sigpt_t) * n);
    if (buffer == NULL) {
        return NULL;
    }

    float t = 0;

    srand(seed);
    for (int i = 0; i < n; i++) {
        buffer[i].t = t;
        buffer[i].y = 5.f * (float)rand() / RAND_MAX;
        buffer[i].dy = 0.f;
        
        t += (float)rand() / (float)RAND_MAX;
    }

    return buffer;
}

void
sigpt_print(const char *name,
            const sigpt_t *in,
            const int n)
{
    printf("%s (%d)\n", name, n);
    for (int i = 0; i < n; i++) {
        sigpt_t sigpt = in[i];
        printf("%i: {t: %f, y: %f, dy: %f}\n", i, sigpt.t, sigpt.y, sigpt.dy);
    }
}

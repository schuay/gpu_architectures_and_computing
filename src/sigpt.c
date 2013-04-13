#include "sigpt.h"

#include <stdlib.h>

sigpt_t *sigpt_random(int seed, int n)
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

#include "util.h"

#include <stdlib.h>

float *random_array(int seed, int length)
{
    float *fs = (float *)malloc(length * sizeof(float));
    if (fs == NULL) {
        return fs;
    }

    srand(seed);

    for (int i = 0; i < length; i++) {
        fs[i] = (float)rand() / (float)RAND_MAX;
    }

    return fs;
}

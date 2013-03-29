#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#include "util.h"

int
main(int argc, char **argv)
{
    printf("Hello World...\n");

    float *fs = random_array(42, 64);
    free(fs);

    return 0;
}

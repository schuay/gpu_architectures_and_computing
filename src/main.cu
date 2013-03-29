#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#include "util.h"

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)

bool
_checkCudaError(cudaError_t result, const char *func, const char *file, int line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        return true;
    } else {
        return false;
    }
}

int
main(int argc, char **argv)
{
    printf("Hello World...\n");

    float *fs = random_array(42, 64);
    float *d_fs;
    checkCudaError(cudaMalloc((void **)&d_fs, 64 * sizeof(float)));

    checkCudaError(cudaMemcpy(d_fs, fs, 64 * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaError(cudaFree(d_fs));
    free(fs);

    return 0;
}

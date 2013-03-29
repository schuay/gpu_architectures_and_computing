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

__global__ void simpleAnd(const float *lhs, const float *rhs, float *result, int nitems)
{
	result[0] = 1.f;
}

int
main(int argc, char **argv)
{
    printf("Hello World...\n");

    float *fs = random_array(42, 64);
    float *d_fs;
    checkCudaError(cudaMalloc((void **)&d_fs, 64 * sizeof(float)));

    checkCudaError(cudaMemcpy(d_fs, fs, 64 * sizeof(float), cudaMemcpyHostToDevice));

    printf("fs[0]: %f\n", fs[0]);

    printf("Launching kernel...\n");
    simpleAnd<<<1, 1>>>(d_fs, d_fs, d_fs, 64);
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaMemcpy(fs, d_fs, 64 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("fs[0]: %f\n", fs[0]);

    checkCudaError(cudaFree(d_fs));
    free(fs);

    return 0;
}

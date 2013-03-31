#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

extern "C" {
#include "util.h"
}

#define NBLOCKS (256)
#define NTHREADS (256)

#define NITEMS (256 * 257)

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
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = tid; i < nitems; i += blockDim.x * gridDim.x) {
		result[i] = ((lhs[i] + rhs[i]) >= 1.f) ? 1.f : 0.f;
	}
}

int
main(int argc, char **argv)
{
    printf("Hello World...\n");

    float *a = random_array(42, NITEMS);
    float *b = random_array(43, NITEMS);
    float *c = (float *)calloc(NITEMS, sizeof(float));
    float *devA, *devB, *devC;

    checkCudaError(cudaMalloc((void **)&devA, NITEMS * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&devB, NITEMS * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&devC, NITEMS * sizeof(float)));

    checkCudaError(cudaMemcpy(devA, a, NITEMS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(devB, b, NITEMS * sizeof(float), cudaMemcpyHostToDevice));

    printf("Launching kernel...\n");
    simpleAnd<<<NBLOCKS, NTHREADS>>>(devA, devB, devC, NITEMS);

    checkCudaError(cudaMemcpy(a, devA, NITEMS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(b, devB, NITEMS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(c, devC, NITEMS * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 20; i++) {
    	printf("%f & %f = %f\n", a[i], b[i], c[i]);
    }

    checkCudaError(cudaFree(devA));
    checkCudaError(cudaFree(devB));
    checkCudaError(cudaFree(devC));

    free(a);
    free(b);
    free(c);

    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

extern "C" {
#include "util.h"
}

#define NBLOCKS (256)
#define NTHREADS (256)

#define NITEMS (256 * 257)

#define FLOAT_DELTA (0.000000001f)

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

__global__ void
simpleAnd(const float *lhs, const float *rhs, float *result, int nitems)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = tid; i < nitems; i += blockDim.x * gridDim.x) {
		result[i] = ((lhs[i] + rhs[i]) >= 2.f - FLOAT_DELTA) ? 1.f : 0.f;
	}
}

/* Note that this implementation does not produce the same results for
 * "not (a and b)" and "(not a or not b)" since "and" requires both
 * operands to equal 1.f while "or" only checks the operand sum.
 */

__global__ void
simpleOr(const float *lhs, const float *rhs, float *result, int nitems)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = tid; i < nitems; i += blockDim.x * gridDim.x) {
		result[i] = ((lhs[i] + rhs[i]) >= 1.f - FLOAT_DELTA) ? 1.f : 0.f;
	}
}

__global__ void
simpleNot(const float *in, float *result, int nitems)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = tid; i < nitems; i += blockDim.x * gridDim.x) {
		result[i] = 1.f - in[i];
	}
}

__global__ void
simpleScan(const float *in, float *result, int nitems)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = tid; i < nitems; i += blockDim.x * gridDim.x) {
		result[i] = 1.f - in[i];
	}
}

int
main(int argc, char **argv)
{
    float *a = random_array(42, NITEMS);
    float *b = random_array(43, NITEMS);
    float *c = (float *)calloc(NITEMS, sizeof(float));
    float *devA, *devB, *devC;

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    checkCudaError(cudaMalloc((void **)&devA, NITEMS * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&devB, NITEMS * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&devC, NITEMS * sizeof(float)));

    checkCudaError(cudaMemcpy(devA, a, NITEMS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(devB, b, NITEMS * sizeof(float), cudaMemcpyHostToDevice));

    printf("Launching kernel...\n");

    /**
     * These calls are asynchronous and just queue the pending operations up on the stream.
     * We can queue our entire operator tree here and run it without interruption from the host.
     *
     * The following sequence computes ((not a or b) and b).
     */
    simpleNot<<<NBLOCKS, NTHREADS, 0, stream>>>(devA, devA, NITEMS);
    simpleOr<<<NBLOCKS, NTHREADS, 0, stream>>>(devA, devB, devC, NITEMS);
    simpleAnd<<<NBLOCKS, NTHREADS, 0, stream>>>(devC, devB, devC, NITEMS);

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

    /* And a Thrust scan operation, let's see how we can integrate that with the rest of
     * the code...
     */

    thrust::plus<float> binary_op;

    thrust::host_vector<float> hostVector(NITEMS);
    thrust::generate(hostVector.begin(), hostVector.end(), rand);

    thrust::device_vector<float> deviceVector = hostVector;
    thrust::exclusive_scan(deviceVector.begin(), deviceVector.end(), deviceVector.begin(), 0.f, binary_op);

    thrust::copy(deviceVector.begin(), deviceVector.end(), hostVector.begin());

    for (thrust::device_vector<float>::iterator iter = deviceVector.begin();
         iter != deviceVector.begin() + 10;
         iter++) {
    	float val = *iter;
    	printf("%f ", val);
    }
    printf("\n");

    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

extern "C" {
#include "sigpt.h"
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
stl_not(const sigpt_t *in, sigpt_t *out, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sigpt_t s = in[i];
        s.y *= -1.f;
        s.dy *= -1.f;
        out[i] = s;
    }
}

int
main(int argc, char **argv)
{
    sigpt_t *a = sigpt_random(42, NITEMS);
    sigpt_t *b = sigpt_random(43, NITEMS);
    sigpt_t *c = (sigpt_t *)calloc(NITEMS, sizeof(sigpt_t));
    sigpt_t *devA, *devB, *devC;

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    checkCudaError(cudaMalloc((void **)&devA, NITEMS * sizeof(sigpt_t)));
    checkCudaError(cudaMalloc((void **)&devB, NITEMS * sizeof(sigpt_t)));
    checkCudaError(cudaMalloc((void **)&devC, NITEMS * sizeof(sigpt_t)));

    checkCudaError(cudaMemcpy(devA, a, NITEMS * sizeof(sigpt_t), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(devB, b, NITEMS * sizeof(sigpt_t), cudaMemcpyHostToDevice));

    printf("Launching kernel...\n");

    /**
     * These calls are asynchronous and just queue the pending operations up on the stream.
     * We can queue our entire operator tree here and run it without interruption from the host.
     *
     * The following sequence computes ((not a or b) and b).
     */
    stl_not<<<NBLOCKS, NTHREADS, 0, stream>>>(devA, devB, NITEMS);

    checkCudaError(cudaMemcpy(a, devA, NITEMS * sizeof(sigpt_t), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(b, devB, NITEMS * sizeof(sigpt_t), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(c, devC, NITEMS * sizeof(sigpt_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 20; i++) {
        printf("not { %f, %f, %f } = { %f, %f, %f }\n",
                a[i].t, a[i].y, a[i].dy,
                b[i].t, b[i].y, b[i].dy);
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

    /*

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

    */

    return 0;
}

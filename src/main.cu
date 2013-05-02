#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

extern "C" {
#include "sigpt.h"
#include "util.h"
}

#include "and.hpp"
#include "globals.h"

#define NITEMS (256 * 257)

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CUDA_MAX(a, b) (((a) > (b)) * (a) + ((a) <= (b)) * (b))

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)

#define TESTFILES_PATH "../matlab/cuda_impl/"


/* TODO: Handle multiple GPUs. */

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

/**
 * sizeof(out) == sizeof(in).
 */
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

struct sigpt_max : public thrust::binary_function<sigpt_t, sigpt_t, sigpt_t>
{
    __device__ sigpt_t
    operator()(const sigpt_t &lhs, const sigpt_t &rhs) const
    {
        return (sigpt_t) {rhs.t, CUDA_MAX(lhs.y, rhs.y), 0};
    }
};

__global__ void
eventually_intersect(const sigpt_t *ys, sigpt_t *zs, sigpt_t *zs_intersect, char *cs, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        cs[i * 2] = 1;
	zs_intersect[i * 2] = zs[i];
        // FIXME: Branches are bad.
        if (i < n - 1 && zs[i].y > zs[i + 1].y) {
            cs[i * 2 + 1] = 1;
            zs_intersect[i * 2 + 1].t = zs[i + 1].t +
		    (zs[i + 1].t - zs[i].t) *
		    (zs[i + 1].y - ys[i + 1].y) / (ys[i + 1].y - ys[i].y);
            zs_intersect[i * 2 + 1].y = zs[i + 1].y;
        }
    }
}

__global__ void
eventually_compact(const sigpt_t *zs, sigpt_t *zs_final, const char *cs, const size_t *fs, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // FIXME: Branches are bad.
        if (cs[i] == 1) {
            zs_final[fs[i]] = zs[i];
        }
    }
}

/**
 * sizeof(out) = 2 * sizeof(in).
 */
void
stl_eventually(const thrust::device_vector<sigpt_t> &in,
        thrust::device_vector<sigpt_t> &out)
{
    thrust::inclusive_scan(in.crbegin(), in.crend(), out.rbegin() + in.size(), sigpt_max()); 

    const sigpt_t *ys = thrust::raw_pointer_cast(in.data());

    thrust::device_vector<sigpt_t> out_intersect(in.size() * 2);
    sigpt_t *zs = thrust::raw_pointer_cast(out_intersect.data());

    sigpt_t *zs_final = thrust::raw_pointer_cast(out.data());

    thrust::device_vector<char> used(in.size() * 2, 0);
    char *cs = thrust::raw_pointer_cast(used.data());
    eventually_intersect<<<NBLOCKS, NTHREADS>>>(ys, zs_final, zs, cs, in.size());

    thrust::device_vector<size_t> positions(in.size() * 2, 0);
    thrust::exclusive_scan(used.cbegin(), used.cend(), positions.begin(), 0, thrust::plus<size_t>()); 

    size_t *fs = thrust::raw_pointer_cast(positions.data());
    eventually_compact<<<NBLOCKS, NTHREADS>>>(zs, zs_final, cs, fs, in.size() * 2);

    out.resize(positions.back());
}


/**
 * test functions
 */
void and_test(const char* sig1_filename, const char* sig2_filename,
		const char* result_filename) {
    sigpt_t *a;
    sigpt_t *b;
    sigpt_t *c;

    int a_n = 0;
    int b_n = 0;
    a_n = read_signal_file(sig1_filename, &a);
    b_n = read_signal_file(sig2_filename, &b);
    if (a_n > 0 && b_n > 0) {
    	c = (sigpt_t *)calloc(4 * NITEMS,sizeof(sigpt_t));
    	if (c == NULL)
    		return;

    	thrust::device_vector<sigpt_t> sig1(a, a + a_n);
    	thrust::device_vector<sigpt_t> sig2(b, b + b_n);
    	thrust::device_vector<sigpt_t> d_result(c, c + 4 * max(a_n, b_n));

        cudaEvent_t start, stop;
	    float elapsedTime;


	    checkCudaError(cudaEventCreate(&start));
	    checkCudaError(cudaEventCreate(&stop));
	    checkCudaError(cudaEventRecord(start, 0));

    	stl_and(sig1, sig2, d_result);

	    checkCudaError(cudaEventRecord(stop, 0));
	    checkCudaError(cudaEventSynchronize(stop));
	    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
	    checkCudaError(cudaEventDestroy(start));
	    checkCudaError(cudaEventDestroy(stop));

	    printf("\tElapsed time: %f ms\n", elapsedTime);



    	thrust::host_vector<sigpt_t> result(d_result);

    	/* there must be a much better way to fetch data back to host */
    	//for (int i = 0; i < result.size(); i++)
    	//	c[i] = result[i];

    	write_signal_file(result_filename,
    			result.data(), result.size());

    	free(a);
    	free(b);
    	free(c);
    } else {
    	fprintf(stderr, "couldn't open one of the test files\n");
    }
}


void eventually_test(const char* sig_filename, const char* result_filename) {
	sigpt_t *a;
	sigpt_t *b;

	int a_n = read_signal_file(sig_filename, &a);
	if (a_n > 0) {
		thrust::device_vector<sigpt_t> in(a, a + a_n);
		thrust::device_vector<sigpt_t> out(a_n * 2);

	    cudaEvent_t start, stop;
	    float elapsedTime;


	    checkCudaError(cudaEventCreate(&start));
	    checkCudaError(cudaEventCreate(&stop));
	    checkCudaError(cudaEventRecord(start, 0));

		stl_eventually(in, out);

	    checkCudaError(cudaEventRecord(stop, 0));
	    checkCudaError(cudaEventSynchronize(stop));
	    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
	    checkCudaError(cudaEventDestroy(start));
	    checkCudaError(cudaEventDestroy(stop));

	    printf("\tElapsed time: %f ms\n", elapsedTime);


		b = (sigpt_t *)calloc(a_n * 2, sizeof(sigpt_t));
		if (b == NULL)
			return;

		for (int i = 0; i < out.size(); i++)
			b[i] = out[i];

		write_signal_file(result_filename, b, out.size());

		free(a);
		free(b);
	} else {
		fprintf(stderr, "couldn't open test file\n");
	}
}

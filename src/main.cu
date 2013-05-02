#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
#include "util.h"
}

#include "and.hpp"
#include "evtl.hpp"
#include "globals.h"

#define NITEMS (256 * 257)

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)

#define TESTFILES_PATH "../matlab/cuda_impl/"


/* TODO: Handle multiple GPUs. */

bool
_checkCudaError(cudaError_t result,
                const char *func,
                const char *file,
                int line)
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
 * test functions
 */
void
and_test(const char* sig1_filename,
         const char* sig2_filename,
		 const char* result_filename)
{
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

void
eventually_test(const char* sig_filename,
                const char* result_filename)
{
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

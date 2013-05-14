#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
#include "util.h"
}

#include "operators/and.hpp"
#include "operators/evtl.hpp"
#include "globals.h"

#define NITEMS (256 * 257)
#define TESTFILES_PATH "matlab/cuda_impl/"

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)


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

    	thrust::device_ptr<sigpt_t> d_result;
        int nout;

        cudaEvent_t start, stop;
	    float elapsedTime;


	    checkCudaError(cudaEventCreate(&start));
	    checkCudaError(cudaEventCreate(&stop));
	    checkCudaError(cudaEventRecord(start, 0));

    	stl_and(&sig1[0], sig1.size(), &sig2[0], sig2.size(), &d_result, &nout);

	    checkCudaError(cudaEventRecord(stop, 0));
	    checkCudaError(cudaEventSynchronize(stop));
	    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
	    checkCudaError(cudaEventDestroy(start));
	    checkCudaError(cudaEventDestroy(stop));

	    printf("\tElapsed time: %f ms\n", elapsedTime);

    	thrust::host_vector<sigpt_t> result(d_result, d_result + nout);

    	/* there must be a much better way to fetch data back to host */
    	//for (int i = 0; i < result.size(); i++)
    	//	c[i] = result[i];

    	write_signal_file(result_filename,
    			result.data(), result.size());

        thrust::device_free(d_result);

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

		stl_evtl(in, out);

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

int
main(int argc, char **argv)
{
    cudaEvent_t start, stop;
    float elapsedTime;

    sigpt_t *a = sigpt_random(42, NITEMS);
    sigpt_t *b = sigpt_random(43, NITEMS);
    sigpt_t *c = (sigpt_t *)calloc(4 * NITEMS,sizeof(sigpt_t));

    thrust::device_vector<sigpt_t> lhs(a, a + NITEMS);
    thrust::device_vector<sigpt_t> rhs(b, b + NITEMS);

    thrust::device_ptr<sigpt_t> out;
    int nout;

    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    checkCudaError(cudaEventRecord(start, 0));

    stl_and(&lhs[0], lhs.size(), &rhs[0], rhs.size(), &out, &nout);

    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    printf("\n\nElapsed time: %f ms\n", elapsedTime);

    thrust::device_free(out);

    free(a);
    free(b);
    free(c);

    printf("run AND test 1\n");
    and_test(TESTFILES_PATH "and-test-sig1.txt",
            TESTFILES_PATH "and-test-sig2.txt",
            TESTFILES_PATH "and-test-gpu-result.txt");

    printf("run AND test 2\n");
    and_test(TESTFILES_PATH "and-test2-sig1.txt",
            TESTFILES_PATH "and-test2-sig2.txt",
            TESTFILES_PATH "and-test2-gpu-result.txt");

    printf("run AND test 3 (100) \n");
    and_test(TESTFILES_PATH "and-test3-100-sig1.txt",
            TESTFILES_PATH "and-test3-100-sig2.txt",
            TESTFILES_PATH "and-test3-100-gpu-result.txt");

    printf("run AND test 3 (1000) \n");
    and_test(TESTFILES_PATH "and-test3-1000-sig1.txt",
            TESTFILES_PATH "and-test3-1000-sig2.txt",
            TESTFILES_PATH "and-test3-1000-gpu-result.txt");

    printf("run AND test 3 (10000) \n");
    and_test(TESTFILES_PATH "and-test3-10000-sig1.txt",
            TESTFILES_PATH "and-test3-10000-sig2.txt",
            TESTFILES_PATH "and-test3-10000-gpu-result.txt");

    printf("run EVENTUALLY test 1\n");
    eventually_test(TESTFILES_PATH "eventually-test-sig.txt",
                   TESTFILES_PATH "eventually-test-gpu-result.txt");

    printf("run EVENTUALLY test 2\n");
    eventually_test(TESTFILES_PATH "eventually-test2-sig.txt",
                   TESTFILES_PATH "eventually-test2-gpu-result.txt");

    printf("run EVENTUALLY test 3 (100) \n");
    eventually_test(TESTFILES_PATH "eventually-test3-100-sig.txt",
                   TESTFILES_PATH "eventually-test3-100-gpu-result.txt");

    printf("run EVENTUALLY test 3 (1000) \n");
    eventually_test(TESTFILES_PATH "eventually-test3-1000-sig.txt",
                   TESTFILES_PATH "eventually-test3-1000-gpu-result.txt");

    printf("run EVENTUALLY test 3 (10000) \n");
    eventually_test(TESTFILES_PATH "eventually-test3-10000-sig.txt",
                   TESTFILES_PATH "eventually-test3-10000-gpu-result.txt");

    printf("run EVENTUALLY test 3 (100000) \n");
    eventually_test(TESTFILES_PATH "eventually-test3-100000-sig.txt",
                   TESTFILES_PATH "eventually-test3-100000-gpu-result.txt");

    printf("run EVENTUALLY test 3 (1000000) \n");
    eventually_test(TESTFILES_PATH "eventually-test3-1000000-sig.txt",
                   TESTFILES_PATH "eventually-test3-1000000-gpu-result.txt");

    return 0;
}

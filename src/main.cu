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
#include "operators/or.hpp"
#include "globals.h"

#define NITEMS (256 * 257)
#define TESTFILES_PATH "matlab/traces/"

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)


/* TODO: Handle multiple GPUs. */

static bool
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

static void
or_test(const char* sig1_filename,
        const char* sig2_filename,
        const char* result_filename)
{
    sigpt_t *a;
    sigpt_t *b;

    int a_n = read_signal_file(sig1_filename, &a);
    int b_n = read_signal_file(sig2_filename, &b);

    if (a_n == 0 || b_n == 0) {
        fprintf(stderr, "couldn't open one of the test files\n");
        return;
    }

    thrust::device_vector<sigpt_t> sig1(a, a + a_n);
    thrust::device_vector<sigpt_t> sig2(b, b + b_n);

    thrust::device_ptr<sigpt_t> d_result;
    int nout;

    cudaEvent_t start, stop;
    float elapsedTime;


    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    checkCudaError(cudaEventRecord(start, 0));

    stl_or(&sig1[0], sig1.size(), &sig2[0], sig2.size(), &d_result, &nout);

    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    printf("\tElapsed time: %f ms\n", elapsedTime);

    thrust::host_vector<sigpt_t> result(d_result, d_result + nout);

    write_signal_file(result_filename,
            result.data(), result.size());

    thrust::device_free(d_result);

    free(a);
    free(b);
}

static void
and_test(const char* sig1_filename,
         const char* sig2_filename,
         const char* result_filename)
{
    sigpt_t *a;
    sigpt_t *b;

    int a_n = read_signal_file(sig1_filename, &a);
    int b_n = read_signal_file(sig2_filename, &b);

    if (a_n == 0 || b_n == 0) {
        fprintf(stderr, "couldn't open one of the test files\n");
        return;
    }

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

    write_signal_file(result_filename,
            result.data(), result.size());

    thrust::device_free(d_result);

    free(a);
    free(b);
}

static void
eventually_test(const char* sig_filename,
                const char* result_filename)
{
    sigpt_t *a;
    sigpt_t *b;

    int a_n = read_signal_file(sig_filename, &a);
    if (a_n > 0) {
        thrust::device_vector<sigpt_t> in(a, a + a_n);
        thrust::device_ptr<sigpt_t> out;
        int nout;

        cudaEvent_t start, stop;
        float elapsedTime;


        checkCudaError(cudaEventCreate(&start));
        checkCudaError(cudaEventCreate(&stop));
        checkCudaError(cudaEventRecord(start, 0));

        stl_evtl(&in[0], in.size(), &out, &nout);

        checkCudaError(cudaEventRecord(stop, 0));
        checkCudaError(cudaEventSynchronize(stop));
        checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
        checkCudaError(cudaEventDestroy(start));
        checkCudaError(cudaEventDestroy(stop));

        printf("\tElapsed time: %f ms\n", elapsedTime);


        b = (sigpt_t *)calloc(a_n * 2, sizeof(sigpt_t));
        if (b == NULL)
            return;

        for (int i = 0; i < nout; i++)
            b[i] = out[i];

        write_signal_file(result_filename, b, nout);

        thrust::device_free(out);

        free(a);
        free(b);
    } else {
        fprintf(stderr, "couldn't open test file\n");
    }
}

int
main(int argc, char **argv)
{
    and_test(TESTFILES_PATH "sig01.trace",
             TESTFILES_PATH "sig02.trace",
             TESTFILES_PATH "and_sig01_sig02.gpu.trace");
    and_test(TESTFILES_PATH "sig03.trace",
             TESTFILES_PATH "sig04.trace",
             TESTFILES_PATH "and_sig03_sig04.gpu.trace");

    or_test(TESTFILES_PATH "sig01.trace",
            TESTFILES_PATH "sig02.trace",
            TESTFILES_PATH "or_sig01_sig02.gpu.trace");
    or_test(TESTFILES_PATH "sig03.trace",
            TESTFILES_PATH "sig04.trace",
            TESTFILES_PATH "or_sig03_sig04.gpu.trace");

    eventually_test(TESTFILES_PATH "sig05.trace",
                    TESTFILES_PATH "ev_sig05.gpu.trace");
    eventually_test(TESTFILES_PATH "sig06.trace",
                    TESTFILES_PATH "ev_sig06.gpu.trace");

    return 0;
}

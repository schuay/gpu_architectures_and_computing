#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>   /* getopt */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

extern "C" {
#include "sigpt.h"
#include "util.h"
}

#include "operators/and.hpp"
#include "operators/evtl.hpp"
#include "operators/or.hpp"
#include "operators/not.hpp"
#include "operators/alw.hpp"
#include "operators/until.hpp"
#include "globals.h"

#define NITEMS (256 * 257)
#define TESTFILES_PATH "matlab/traces/"

#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)


typedef void (*binary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                   const thrust::device_ptr<sigpt_t> &, const int,
                                   thrust::device_ptr<sigpt_t>*, int*);

typedef void (*unary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                  thrust::device_ptr<sigpt_t>*, int*);

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

static float
binary_operator_test(binary_operator_t op_function,
                     const sigpt_t* a, const int a_n,
                     const sigpt_t* b, const int b_n,
                     const char *result_filename)
{

    thrust::device_vector<sigpt_t> sig1(a, a + a_n);
    thrust::device_vector<sigpt_t> sig2(b, b + b_n);

    thrust::device_ptr<sigpt_t> d_result;
    int nout;

    cudaEvent_t start, stop;
    float elapsedTime;

    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    checkCudaError(cudaEventRecord(start, 0));

    op_function(&sig1[0], sig1.size(), &sig2[0], sig2.size(), &d_result, &nout);

    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    if (result_filename) {
        thrust::host_vector<sigpt_t> result_h(d_result, d_result + nout);
        write_signal_file(result_filename,
                result_h.data(), result_h.size());
    }

    thrust::device_free(d_result);

    return elapsedTime;
}

static float
unary_operator_test(unary_operator_t op_function,
                    const sigpt_t *a, const int a_n,
                    const char *result_filename)
{

    thrust::device_vector<sigpt_t> in(a, a + a_n);
    thrust::device_ptr<sigpt_t> out;
    int nout;

    cudaEvent_t start, stop;
    float elapsedTime;

    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    checkCudaError(cudaEventRecord(start, 0));

    op_function(&in[0], in.size(), &out, &nout);

    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    if (result_filename) {
        thrust::host_vector<sigpt_t> result_h(out, out + nout);
        write_signal_file(result_filename, 
                result_h.data(), result_h.size());
    }

    thrust::device_free(out);

    return elapsedTime;
}

void
usage(char* prog_name)
{
    printf("Usage: %s [-o resultfile] <formular> <signal1> [<signal2>]\n", prog_name);
    printf("   calculate robustness on the given formular and signal traces\n");
    printf("\n");
    printf("   <formular>   defines the stl formular, currently only the operator names\n");
    printf("                can be given: valid operator names:\n");
    printf("                  AND, OR, NOT, UNTIL, ALW, EVTL\n");
    printf("   <signal1>    input signal 1\n");
    printf("   <signal2>    input signal 2\n");
    printf("\n");
    printf("   -o file      filename for storing the resulting signal\n");
    printf("   -h           show help (this page)\n");
    printf("\n");
}

void
print_elapsed_time(const char *formular, 
                   const char *sig1_file, 
                   const char *sig2_file,
                   float time) {
    printf("finished test %s for %s", formular, sig1_file);
    if (sig2_file)
        printf(", %s", sig2_file);

    printf(", elapsed time: %.6f s\n", time / 1000);
}



int
main(int argc, char **argv)
{
    int opt;
    char *result_filename = NULL;
    char *sig1_filename = NULL;
    char *sig2_filename = NULL;
    char *formular;

    sigpt_t *sig1;
    sigpt_t *sig2;
    sigpt_t *result;
    int sig1_n, sig2_n, result_n;


    while((opt = getopt(argc, argv, "ho:")) != -1) {
        switch(opt) {
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
            break;

        case 'o':
            result_filename = optarg;
            break;

        default:
            fprintf(stderr, "Invalid option -%c", opt);
            usage(argv[0]);
            exit(EXIT_FAILURE);
       }
    }

    if ( (optind + 1) >= argc ) {
        fprintf(stderr, "Expected formular and input signal filename\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    formular = argv[optind];
    sig1_filename = argv[optind + 1];
    if ( (optind + 2) <= argc )
        sig2_filename = argv[optind + 2];

/*    
    printf("f: %s, s1: %s, s2: %s\n", formular, sig1_filename, sig2_filename ? sig2_filename : "(null)");
    if (result_filename)
        printf("result: %s\n", result_filename);
*/
    sig1_n = read_signal_file(sig1_filename, &sig1);
    if (sig1_n <= 0) {
        fprintf(stderr, "could not open signal file '%s' or to less memory available.\n", 
                sig1_filename);
        exit(EXIT_FAILURE);
    }
    if (sig2_filename) {
        sig2_n = read_signal_file(sig2_filename, &sig2);
        if (sig2_n <= 0) {
            fprintf(stderr, "could not open signal file '%s' or to less memory available.\n", 
                    sig2_filename);
            exit(EXIT_FAILURE);
        }
    }

    float time;

    
    if (strncmp(formular, "AND", 3) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "AND operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_and, sig1, sig1_n, sig2, sig2_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "OR", 2) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "OR operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_or, sig1, sig1_n, sig2, sig2_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formular, "NOT", 3) == 0) {

        time = unary_operator_test(stl_not, sig1, sig1_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formular, "UNTIL", 5) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "UNTIL operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_until, sig1, sig1_n, sig2, sig2_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "ALW", 3) == 0) {
    
        time = unary_operator_test(stl_alw, sig1, sig1_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "EVTL", 4) == 0) {
    
        time = unary_operator_test(stl_evtl, sig1, sig1_n, result_filename);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);
    }



    exit(EXIT_SUCCESS);
}

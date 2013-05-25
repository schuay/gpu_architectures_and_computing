#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>   /* getopt */
#include <libgen.h>   /* basename */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

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

#include "sigcmp.hpp"


#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)


typedef void (*binary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                   const thrust::device_ptr<sigpt_t> &, const int,
                                   thrust::device_ptr<sigpt_t>*, int*);

typedef void (*unary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                  thrust::device_ptr<sigpt_t>*, int*);

static char *prog_name = NULL;

static void
usage()
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
    printf("   -c file      compare calculated sig with sig from file\n");
    printf("   -h           show help (this page)\n");
    printf("\n");
}


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


static float
binary_operator_test(binary_operator_t op_function,
                     const thrust::host_vector<sigpt_t> &a,
                     const thrust::host_vector<sigpt_t> &b,
                     thrust::host_vector<sigpt_t> &result)
{
    thrust::device_vector<sigpt_t> sig1(a);
    thrust::device_vector<sigpt_t> sig2(b);
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

    result.assign(d_result, d_result + nout);
    thrust::device_free(d_result);

    return elapsedTime;
}


static float
unary_operator_test(unary_operator_t op_function,
                    const thrust::host_vector<sigpt_t> &a,
                    thrust::host_vector<sigpt_t> &result)
{

    thrust::device_vector<sigpt_t> in(a);
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


    result.assign(out, out + nout);
    thrust::device_free(out);

    return elapsedTime;
}


static void
print_elapsed_time(const char *formular, 
                   const char *sig1_file, 
                   const char *sig2_file,
                   float time) 
{
    printf("%s: finished test %s ", prog_name, formular);
    printf(" elapsed time: %.6f s\n", time / 1000); // print in sec to be inline with matlab
}


static bool
read_signal(const char *filename, 
            thrust::host_vector<sigpt_t> &v) 
{
    sigpt_t *sig;
    int n = read_signal_file(filename, &sig);
    if (n <= 0) {
        return false;
    }
    v.assign(sig, sig + n);
    free(sig);

    return true;
}


int
main(int argc, char **argv)
{
    int opt;
    char *result_filename = NULL;
    char *sig1_filename = NULL;
    char *sig2_filename = NULL;
    char *cmp_filename = NULL;
    char *formular;

    prog_name = basename(argv[0]);

    thrust::host_vector<sigpt_t> sig1;
    thrust::host_vector<sigpt_t> sig2;
    thrust::host_vector<sigpt_t> result;
    thrust::host_vector<sigpt_t> cmp;

    while((opt = getopt(argc, argv, "ho:c:")) != -1) {
        switch(opt) {
        case 'h':
            usage();
            exit(EXIT_SUCCESS);
            break;

        case 'o':
            result_filename = optarg;
            break;

        case 'c':
            cmp_filename = optarg;
            break;

        default:
            fprintf(stderr, "Invalid option -%c", opt);
            usage();
            exit(EXIT_FAILURE);
       }
    }

    if ( (optind + 1) >= argc ) {
        fprintf(stderr, "Expected formular and input signal filename\n");
        usage();
        exit(EXIT_FAILURE);
    }

    formular = argv[optind];
    sig1_filename = argv[optind + 1];
    if ( (optind + 2) <= argc )
        sig2_filename = argv[optind + 2];


    /*
     * read signal files
     */
    if (!read_signal(sig1_filename, sig1)) {
        fprintf(stderr, "could not open signal file '%s' or to less memory available.\n", 
                sig1_filename);
        exit(EXIT_FAILURE);
    }
    if (sig2_filename) {
        if (!read_signal(sig2_filename, sig2)) {
            fprintf(stderr, "could not open signal file '%s' or to less memory available.\n", 
                    sig2_filename);
            exit(EXIT_FAILURE);
        }
    }



    float time;
    /*
     * check for operator and execute it 
     * TODO: can we do this with the parser???
     */
    if (strncmp(formular, "AND", 3) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "AND operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_and, sig1, sig2, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "OR", 2) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "OR operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_or, sig1, sig2, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formular, "NOT", 3) == 0) {

        time = unary_operator_test(stl_not, sig1, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formular, "UNTIL", 5) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, "UNTIL operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_until, sig1, sig2, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "ALW", 3) == 0) {
    
        time = unary_operator_test(stl_alw, sig1, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else if (strncmp(formular, "EVTL", 4) == 0) {
    
        time = unary_operator_test(stl_evtl, sig1, result);
        print_elapsed_time(formular, sig1_filename, sig2_filename, time);

    } else {
        fprintf(stderr, "unknown operator '%s'\n", formular);
        exit(EXIT_FAILURE);
    }

    /*
     * write result file (if needed)
     */
    if (result_filename) {
        write_signal_file(result_filename, result.data(), result.size());
    }

    /*
     * check resulting signal with given compare signal (if any)
     */
    if (cmp_filename) {
        if (!read_signal(cmp_filename, cmp)) {
            fprintf(stderr, "could not read signal from file %s or not enough memory available\n",
                    cmp_filename);
            exit(EXIT_FAILURE);
        }
        if (sigcmp(result.data(), result.size(), cmp.data(), cmp.size()) != 0) {
            fprintf(stderr, "calculated signal dosn't match with given compare signal\n");
            exit(EXIT_FAILURE);
        }
    }

    exit(EXIT_SUCCESS);
}


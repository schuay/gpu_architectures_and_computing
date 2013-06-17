#include <libgen.h>   /* basename */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unistd.h>   /* getopt */

extern "C" {
#include "sigpt.h"
#include "util.h"
}

#include "globals.h"
#include "operators/alw.hpp"
#include "operators/and.hpp"
#include "operators/evtl.hpp"
#include "operators/not.hpp"
#include "operators/or.hpp"
#include "operators/until.hpp"
#include "operators/buntil.hpp"
#include "operators/bevtl.hpp"
#include "operators/balw.hpp"
#include "sigcmp.hpp"


// define operator names (for commanline parsing)
#define OPNAME_NOT      "not"
#define OPNAME_AND      "and"
#define OPNAME_OR       "or"
#define OPNAME_ALW      "alw"
#define OPNAME_EVTL     "ev"
#define OPNAME_UNTIL    "until"


#define checkCudaError(val) do { _checkCudaError((val), #val, __FILE__, __LINE__); } while (0)

#define PRE_MEASURE_TIME \
    cudaEvent_t start, stop; \
    float elapsedTime;       \
                             \
    checkCudaError(cudaEventCreate(&start));    \
    checkCudaError(cudaEventCreate(&stop));     \
    checkCudaError(cudaEventRecord(start, 0))

#define POST_MEASURE_TIME \
    checkCudaError(cudaEventRecord(stop, 0));   \
    checkCudaError(cudaEventSynchronize(stop)); \
    checkCudaError(cudaEventElapsedTime(&elapsedTime, start, stop));    \
    checkCudaError(cudaEventDestroy(start));    \
    checkCudaError(cudaEventDestroy(stop))


typedef void (*binary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                   const thrust::device_ptr<sigpt_t> &, const int,
                                   thrust::device_ptr<sigpt_t>*, int*);

typedef void (*unary_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                  thrust::device_ptr<sigpt_t>*, int*);

typedef void (*binary_bound_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                         const thrust::device_ptr<sigpt_t> &, const int,
                                         const float s, const float t,
                                         thrust::device_ptr<sigpt_t>*, int*);

typedef void (*unary_bound_operator_t) (const thrust::device_ptr<sigpt_t> &, const int,
                                        const float s, const float t,
                                        thrust::device_ptr<sigpt_t>*, int*);

static char *prog_name = NULL;

static void
usage()
{
    printf("Usage: %s [-o resultfile] <formula> <signal1> [<signal2>]\n", prog_name);
    printf("   calculate robustness on the given formula and signal traces\n");
    printf("\n");
    printf("   <formula>    defines the stl formula, currently only the operator names\n");
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

    PRE_MEASURE_TIME;

    op_function(&sig1[0], sig1.size(), &sig2[0], sig2.size(), &d_result, &nout);

    POST_MEASURE_TIME;

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
    
    PRE_MEASURE_TIME;
    
    op_function(&in[0], in.size(), &out, &nout);

    POST_MEASURE_TIME;

    result.assign(out, out + nout);
    thrust::device_free(out);

    return elapsedTime;
}


static float
binary_bound_operator_test(binary_bound_operator_t op_function,
                           const thrust::host_vector<sigpt_t> &a,
                           const thrust::host_vector<sigpt_t> &b,
                           const float s, const float t,
                           thrust::host_vector<sigpt_t> &result)
{
    thrust::device_vector<sigpt_t> sig1(a);
    thrust::device_vector<sigpt_t> sig2(b);
    thrust::device_ptr<sigpt_t> d_result;
    int nout;

    PRE_MEASURE_TIME;

    op_function(&sig1[0], sig1.size(), &sig2[0], sig2.size(), s, t, &d_result, &nout);

    POST_MEASURE_TIME;

    result.assign(d_result, d_result + nout);
    thrust::device_free(d_result);

    return elapsedTime;
}


static float
unary_bound_operator_test(unary_bound_operator_t op_function,
                          const thrust::host_vector<sigpt_t> &a,
                          const float s, const float t, 
                          thrust::host_vector<sigpt_t> &result)
{

    thrust::device_vector<sigpt_t> in(a);
    thrust::device_ptr<sigpt_t> out;
    int nout;
    
    PRE_MEASURE_TIME;
    
    op_function(&in[0], in.size(), s, t, &out, &nout);

    POST_MEASURE_TIME;

    result.assign(out, out + nout);
    thrust::device_free(out);

    return elapsedTime;
}


static void
print_elapsed_time(const char *formula, 
                   const char *sig1_file, 
                   const char *sig2_file,
                   float time) 
{
    printf("%s: finished test %s elapsed time: %.6f s\n", 
           prog_name, formula, time / 1000); // print in sec to be inline with matlab
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
    char *formula;

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
        fprintf(stderr, "Expected formula and input signal filename\n");
        usage();
        exit(EXIT_FAILURE);
    }

    formula = argv[optind];
    sig1_filename = argv[optind + 1];
    if ( (optind + 2) <= argc )
        sig2_filename = argv[optind + 2];


    /*
     * read signal files
     */
    if (!read_signal(sig1_filename, sig1)) {
        fprintf(stderr, "could not open signal file '%s' or too less memory available.\n", 
                sig1_filename);
        exit(EXIT_FAILURE);
    }
    if (sig2_filename) {
        if (!read_signal(sig2_filename, sig2)) {
            fprintf(stderr, "could not open signal file '%s' or too less memory available.\n", 
                    sig2_filename);
            exit(EXIT_FAILURE);
        }
    }



    float time;
    float lower_bound;
    float upper_bound;
    /*
     * check for operator and execute it 
     * TODO: can we do this with the parser???
     */
    if (strncmp(formula, OPNAME_AND, strlen(OPNAME_AND)) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, OPNAME_AND " operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_and, sig1, sig2, result);
        print_elapsed_time(formula, sig1_filename, sig2_filename, time);

    } else if (strncmp(formula, OPNAME_OR, strlen(OPNAME_OR)) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, OPNAME_OR " operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        time = binary_operator_test(stl_or, sig1, sig2, result);
        print_elapsed_time(formula, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formula, OPNAME_NOT, strlen(OPNAME_NOT)) == 0) {

        time = unary_operator_test(stl_not, sig1, result);
        print_elapsed_time(formula, sig1_filename, sig2_filename, time);
    
    } else if (strncmp(formula, OPNAME_UNTIL, strlen(OPNAME_UNTIL)) == 0) {
        if (!sig2_filename) {
            fprintf(stderr, OPNAME_UNTIL " operator requires two input signals\n");
            exit(EXIT_FAILURE);
        }

        if (strlen(formula) > strlen(OPNAME_UNTIL)) {
            if (sscanf(formula, OPNAME_UNTIL "_[%f,%f]", &lower_bound, &upper_bound) != 2) {
                fprintf(stderr, "could not parse bound values for operator " 
                                OPNAME_UNTIL "\n");
                exit(EXIT_FAILURE);
            } else {
                time = binary_bound_operator_test(stl_buntil, sig1, sig2, 
                                                  lower_bound, upper_bound, result);
            }
        } else {
            time = binary_operator_test(stl_until, sig1, sig2, result);
        }

        print_elapsed_time(formula, sig1_filename, sig2_filename, time);
        
    } else if (strncmp(formula, OPNAME_ALW, strlen(OPNAME_ALW)) == 0) {
    
        if (strlen(formula) > strlen(OPNAME_ALW)) {
            if (sscanf(formula, OPNAME_ALW "_[%f,%f]", &lower_bound, &upper_bound) != 2) {
                fprintf(stderr, "cound not parse bound values for operator "
                                OPNAME_ALW "\n");
                exit(EXIT_FAILURE);
            } else {
                time = unary_bound_operator_test(stl_balw, sig1, lower_bound, upper_bound, result);
            }
        } else {
            time = unary_operator_test(stl_alw, sig1, result);
        }
        print_elapsed_time(formula, sig1_filename, sig2_filename, time);

    } else if (strncmp(formula, OPNAME_EVTL, strlen(OPNAME_EVTL)) == 0) {

        if (strlen(formula) > strlen(OPNAME_EVTL)) {
            if (sscanf(formula, OPNAME_EVTL "_[%f,%f]", &lower_bound, &upper_bound) != 2) {
                fprintf(stderr, "cound not parse bound values for operator "
                                OPNAME_EVTL "\n");
                exit(EXIT_FAILURE);
            } else {
                time = unary_bound_operator_test(stl_bevtl, sig1, lower_bound, upper_bound, result);
            }
        } else {
            time = unary_operator_test(stl_evtl, sig1, result);
        }
        print_elapsed_time(formula, sig1_filename, sig2_filename, time);

    } else {
        fprintf(stderr, "unknown operator '%s'\n", formula);
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


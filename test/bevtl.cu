#include "operators/bevtl.hpp"

#include "interpolate.hpp"

extern "C" {
#include <check.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "sigpt.h"
#include "util.h"
}

#include "sigcmp.hpp"

#define NITEMS (42)

#define BEVTL_TEST(name, s, t, fin, fexpected) \
START_TEST(name) \
{ \
    sigpt_t *in, *out; \
 \
    int in_n = read_signal_file(SIG_PATH "/" fin, &in); \
    fail_unless(in_n != -1); \
 \
    thrust::device_vector<sigpt_t> vin(in, in + in_n); \
    thrust::device_ptr<sigpt_t> vout; \
    int nout = 0; \
 \
    stl_bevtl(&vin[0], vin.size(), s, t, &vout, &nout); \
 \
    thrust::host_vector<sigpt_t> host_out(vout, vout + nout); \
 \
    int out_n = read_signal_file(SIG_PATH "/" fexpected, &out); \
    fail_unless(out_n != -1); \
 \
    fail_unless(sigcmp_bevtl(out, out_n, host_out.data(), host_out.size()) == 0); \
    thrust::device_free(vout); \
 \
    free(in); \
    free(out); \
} \
END_TEST

START_TEST(test_sanity)
{
    sigpt_t *a = sigpt_random(42, NITEMS);

    thrust::device_vector<sigpt_t> in(a, a + NITEMS);
    thrust::device_ptr<sigpt_t> out;
    int nout;

    stl_bevtl(&in[0], NITEMS, 5, 10, &out, &nout);

    thrust::device_free(out);

    free(a);
}
END_TEST

int
sigcmp_bevtl(const sigpt_t *lhs,
             const int nlhs,
             const sigpt_t *rhs,
             const int nrhs)
{
    sigpt_t *lhs_new = (sigpt_t *) lhs;
    int nlhs_new = nlhs;
    sigpt_t *rhs_end = (sigpt_t *) (rhs + (nrhs - 1));
    int nrhs_new = nrhs;

    if (nlhs == 0 || nrhs == 0) {
        return (nlhs != 0 || nrhs != 0);
    }

    /* Discard early breach results. */
    for (; lhs_new->t < rhs->t && lhs_new < lhs + nlhs; lhs_new++) {
        nlhs_new--;
    }

    if (nlhs_new == 0) {
        return -1;
    }

    if (lhs_new != lhs && !FLOAT_EQUALS(lhs_new->t, rhs->t)) {
        sigpt_t interpolated = interpolate(lhs_new - 1, lhs_new, rhs->t);
        lhs_new--;
        nlhs_new++;
        *lhs_new = interpolated;
    }

    /* Discard our own late results. */
    for (; rhs_end->t > lhs[nlhs - 1].t && rhs_end >= rhs; rhs_end--) {
        nrhs_new--;
    }

    if (nrhs_new == 0) {
        return -1;
    }

    if (rhs_end != (rhs + (nrhs - 1)) && !FLOAT_EQUALS(lhs[nlhs - 1].t, rhs_end->t)) {
        sigpt_t interpolated = interpolate(rhs_end, rhs_end + 1, lhs[nlhs - 1].t);
        nrhs_new++;
        *(rhs_end + 1) = interpolated;
    }

    return sigcmp(lhs_new, nlhs_new, rhs, nrhs_new);
}

BEVTL_TEST(test_sig1, 0.5, 1.5, "sig05.trace", "bevtl_0.5_1.5_sig05.breach.trace")
BEVTL_TEST(test_sig2, 0.5, 1.5, "sig06.trace", "bevtl_0.5_1.5_sig06.breach.trace")

static Suite *
create_suite(void)
{
    Suite *s = suite_create(__FILE__);
    TCase *tc_core = tcase_create("core");

    tcase_add_test(tc_core, test_sanity);
    tcase_add_test(tc_core, test_sig1);
    tcase_add_test(tc_core, test_sig2);

    suite_add_tcase(s, tc_core);

    return s;
}

int
main(int argc __attribute__ ((unused)),
     char **argv __attribute__ ((unused)))
{
    int number_failed;
    Suite *s = create_suite();
    SRunner *sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

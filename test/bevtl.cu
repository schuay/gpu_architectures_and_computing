#include "operators/bevtl.hpp"

extern "C" {
#include <check.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "sigcmp.h"
#include "sigpt.h"
#include "util.h"
}

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
    int nout; \
 \
    stl_bevtl(&vin[0], vin.size(), s, t, &vout, &nout); \
 \
    thrust::host_vector<sigpt_t> host_out(vout, vout + nout); \
 \
    int out_n = read_signal_file(SIG_PATH "/" fexpected, &out); \
    fail_unless(out_n != -1); \
 \
    fail_unless(sigcmp(out, host_out.data(), MAX(out_n, host_out.size())) == 0); \
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

BEVTL_TEST(test_sig1, 5, 10, "sig05.trace", "bevtl_5_10_sig05.breach.trace")
BEVTL_TEST(test_sig2, 5, 10, "sig06.trace", "bevtl_5_10_sig06.breach.trace")

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

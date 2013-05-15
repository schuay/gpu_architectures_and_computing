#include "operators/buntil.hpp"

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

#define BUNTIL_TEST(name, s, t, flhs, frhs, fexpected) \
START_TEST(name) \
{ \
    sigpt_t *a, *b, *c; \
 \
    int a_n = read_signal_file(SIG_PATH "/" flhs, &a); \
    int b_n = read_signal_file(SIG_PATH "/" frhs, &b); \
    fail_unless(a_n != -1 && b_n != -1); \
 \
    thrust::device_vector<sigpt_t> lhs(a, a + a_n); \
    thrust::device_vector<sigpt_t> rhs(b, b + b_n); \
 \
    thrust::device_ptr<sigpt_t> out; \
    int nout; \
 \
    stl_buntil(&lhs[0], lhs.size(), &rhs[0], rhs.size(), s, t, &out, &nout); \
 \
    thrust::host_vector<sigpt_t> host_out(out, out + nout); \
 \
    int c_n = read_signal_file(SIG_PATH "/" fexpected, &c); \
    fail_unless(c_n != -1); \
 \
    fail_unless(sigcmp(c, host_out.data(), MAX(c_n, host_out.size())) == 0); \
 \
    thrust::device_free(out); \
 \
    free(a); \
    free(b); \
    free(c); \
} \
END_TEST

START_TEST(test_sanity)
{
    sigpt_t *a = sigpt_random(42, NITEMS);
    sigpt_t *b = sigpt_random(43, NITEMS);

    thrust::device_vector<sigpt_t> lhs(a, a + NITEMS);
    thrust::device_vector<sigpt_t> rhs(b, b + NITEMS);

    thrust::device_ptr<sigpt_t> out;
    int nout;

    stl_buntil(&lhs[0], lhs.size(), &rhs[0], rhs.size(), 5, 10, &out, &nout);

    thrust::device_free(out);

    free(a);
    free(b);
}
END_TEST

BUNTIL_TEST(test_sig1, 5, 10, "sig01.trace", "sig02.trace", "buntil_5_10_sig01_sig02.breach.trace")
BUNTIL_TEST(test_sig2, 5, 10, "sig03.trace", "sig04.trace", "buntil_5_10_sig03_sig04.breach.trace")

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

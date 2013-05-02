#include "and.hpp"

extern "C" {
#include <check.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "sigpt.h"
#include "util.h"
}

#define NITEMS (42)

int
sigcmp(const sigpt_t *lhs, const sigpt_t *rhs, int n)
{
    /* TODO: Won't work, fix me once we have decent test files. */
    return memcmp(lhs, rhs, n);
}

START_TEST(test_sanity)
{
    sigpt_t *a = sigpt_random(42, NITEMS);
    sigpt_t *b = sigpt_random(43, NITEMS);
    sigpt_t *c = (sigpt_t *)calloc(4 * NITEMS,sizeof(sigpt_t));

    thrust::device_vector<sigpt_t> lhs(a, a + NITEMS);
    thrust::device_vector<sigpt_t> rhs(b, b + NITEMS);
    thrust::device_vector<sigpt_t> out(c, c + 4 * NITEMS);

    stl_and(lhs, rhs, out);

    free(a);
    free(b);
    free(c);
}
END_TEST

START_TEST(test_sig1)
{
    sigpt_t *a, *b, *c;

    int a_n = read_signal_file(SIG_PATH "/and-test-sig1.txt", &a);
    int b_n = read_signal_file(SIG_PATH "/and-test-sig2.txt", &b);
    fail_unless(a_n != -1 && b_n != -1);

    thrust::device_vector<sigpt_t> lhs(a, a + a_n);
    thrust::device_vector<sigpt_t> rhs(b, b + b_n);
    thrust::device_vector<sigpt_t> out(4 * MAX(a_n, b_n));

    stl_and(lhs, rhs, out);

    thrust::host_vector<sigpt_t> host_out(out);

    int c_n = read_signal_file(SIG_PATH "/and-test-breach-result.txt", &c);
    fail_unless(c_n != -1);

    fail_unless(sigcmp(c, host_out.data(), MAX(c_n, host_out.size())) == 0);

    free(a);
    free(b);
    free(c);
}
END_TEST

static Suite *
create_suite(void)
{
    Suite *s = suite_create(__FILE__);
    TCase *tc_core = tcase_create("core");

    tcase_add_test(tc_core, test_sanity);
    tcase_add_test(tc_core, test_sig1);

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

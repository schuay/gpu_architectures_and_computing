if ~ exist('init_test_cases', 'file')
    addpath('util')
end

init_test_cases

alw_tests
and_tests
balw_tests
bevtl_tests
buntil_tests
evtl_tests
not_tests
until_tests

writeResultFiles
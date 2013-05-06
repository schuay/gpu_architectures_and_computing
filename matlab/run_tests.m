if ~ exist('init_test_cases', 'file')
    addpath('util')
end
init_test_cases


fprintf('\nrunning AND tests...\n');
and_tests

fprintf('\nrunning EVENTUALLY tests...\n');
eventually_tests
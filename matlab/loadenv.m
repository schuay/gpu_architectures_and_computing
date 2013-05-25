% get path definitions from environment add do addpath(...)

% breach path
n = getenv('BREACH_PATH');
if ~isempty(n)
    addpath(n);
end

% load test environment
addpath('util');
addpath('benchmarks');

% basic testing of cuda implementation of robustness operators
% (most things taken from Music example)

% clear workspace
clear all

%addpath('../..');   % path to breach
InitBreach;
%define_breach_system;
% this part comes from define_breach_system
vars = {'s1', 's2'};
params = {'p0', 'p1','p2', 'p3'};  % generic parameters  
p0 = [ 0 0, 0 0 0 0 ];
Sys = CreateExternSystem('myTest', vars, params, p0);

% define path to trace files
TRACE_PATH = 'traces/';

% place resulting (breach generated) traces in 
% this array. optionally write these values to the result files
% format: each element must have this format
%    val.X     ...  trace values
%    val.time  ... time domain for the trace
%    val.name  ... name for the trace file
% 
resultArray = [];
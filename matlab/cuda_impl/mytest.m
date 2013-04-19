% basic testing of cuda implementation of robustness operators
% (most things taken from Music example)

% clear workspace
clear all

%addpath('../..');   % path to breach
InitBreach;

%define_breach_system;
% this part comes from define_breach_system
vars = {'s1', 's2'};
params = {'p0', 'p1','p2', 'p3'};                                                            % generic parameters  
p0 = [ 0 0, 0 0 0 0 ];
Sys = CreateExternSystem('myTest', vars, params, p0);

% define our signal
%traj.time = [ 0 1 2 3 4 5 6 ];
%traj.X = [-2.0 0.0 2.0 1.0 0.5 0.2 -0.5];
traj.time = [ 0.2 0.6 1.0 1.5 1.9 2.3 2.5 3.0 ];
traj.X = [ -0.5 -0.7 1.3 2.5 3.0 0.5 -0.3 2.0,
            1.5 0.75 -1.3 1.3 4 1.5 0.5 0.1 ];
traj.param = Sys.p;

% taken from init_examples.m
P = CreateSampling(Sys, 1);
P.pts = traj.param';
P.traj = traj;

% define our formula
% and
%QMITL_Formula('phi', '(s1[t] > 0) and (s2[t] > 0)');

% eventually
QMITL_Formula('phi', 'ev (s1[t] > 0)');

time2 = traj.time(1):0.001:traj.time(end);

val1 = QMITL_Eval(Sys, phi, P, traj)
val2 = QMITL_Eval2(Sys, phi, traj, time2)

plot(traj.time, traj.X(1,:), time2, val2);

% plot and
%plot(traj.time, traj.X(1,:), traj.time, traj.X(2,:), time2, val2);
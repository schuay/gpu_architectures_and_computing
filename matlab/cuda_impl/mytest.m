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


%%
%% first test AND operation
%%
sig1 = readSignal('and-test-sig1.txt');
sig2 = readSignal('and-test-sig2.txt');

% breach seems to do not support 2 input signals with different
% time scales. Thus we have to merge the two time scales and
% interpolate the values
traj.time = union(sig1.time, sig2.time);
traj.X(1,:) = interp1(sig1.time, sig1.X, traj.time, 'linear');
traj.X(2,:) = interp1(sig2.time, sig2.X, traj.time, 'linear');

traj.param = Sys.p;

% taken from init_examples.m
P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

% define our formula
% and
QMITL_Formula('phi', '(s1[t] > 0) and (s2[t] > 0)');

QMITL_Eval(Sys, phi, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, phi, traj);

% save the result
writeSignal('and-test-breach-result.txt', [val2.time; val2.X]);

figure(1);
plot(sig1.time, sig1.X, '-sb',...
     sig2.time, sig2.X, '-sr',...
     val2.time, val2.X, '-sg');
title('sig1 (blue), sig2 (red), (s1[t] > 0) and (s2[t] > 0) (green)');
grid on;


%% 
%% second AND test
%%

% create time sequence and two signals
t = [0:0.4:3*pi];

a = -.5 .* cos(t/(pi)) + 0.7;
y1 = a .* sin(t);

y2 = 0.6 .* cos(t);

% fill signal in our system 
traj.time = t;
traj.X = [y1 ; y2];

traj.param = Sys.p;
P.traj = traj;

% we just reuse our old formula
QMITL_Eval(Sys, phi, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, phi, traj);

% plot it
figure(2);
plot(t, y1, '-xb',...
     t, y2, '-xr',...
     val2.time, val2.X, '-xg');
 
% save our signals
writeSignal('and_test2-sig1.txt', [t ; y1]);
writeSignal('and_test2-sig2.txt', [t ; y2]);
writeSignal('and_test2-breach-result.txt', [val2.time ; val2.X]);



%%
%% first eventually test
%%
sig1 = readSignal('eventually-test-sig.txt');

traj.time = sig1.time;
traj.X = sig1.X;

traj.param = Sys.p;
P.traj = traj;

% define eventually formula
QMITL_Formula('phi_ev', 'ev (s1[t] > 0)');

QMITL_Eval(Sys, phi_ev, P, traj);
val = QMITL_Eval2raw(Sys, phi_ev, traj);

figure(3);
plot(traj.time, traj.X, '-xb',...
     val.time, val.X, '-sg');

% save result
writeSignal('eventually-test-breach-result.txt', [val.time ; val.X]);
 
%%
%% second eventually test
%%
t = [0:0.4:3*pi];
a = 0.5 .* cos(t/(pi)) + 0.7;
y = a .* sin(t);

traj.time = t;
traj.X = y;
traj.param = Sys.p;
P.traj = traj;

% reuse phi_ev
QMITL_Eval(Sys, phi_ev, P, traj);
val = QMITL_Eval2raw(Sys, phi_ev, traj);

% save signals
writeSignal('eventually-test2-sig.txt', [t ; y]);
writeSignal('eventually-test2-breach-result.txt', [val.time, val.X]);

figure(4);
plot(t, y, '-xb',...
     val.time, val.X, '-sr');

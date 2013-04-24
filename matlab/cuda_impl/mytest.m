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
QMITL_Formula('until', '(s1[t] > 0) until (s2[t] > 0)');

QMITL_Eval(Sys, phi, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, until, traj);

% save the result
writeSignal('and-test-breach-result.txt', [val2.time; val2.X]);

figure(1);
subplot(2,1,1);
plot(sig1.time, sig1.X, '-sb',...
     sig2.time, sig2.X, '-sr',...
     val2.time, val2.X, '-sg');
title('sig1 (blue), sig2 (red), (s1[t] > 0) and (s2[t] > 0) (green)');
grid on;

gpu_result = readSignal('and-test-gpu-result.txt');
subplot(2,1,2);
plot(gpu_result.time, gpu_result.X, '-xg');
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
val2 = QMITL_Eval2raw(Sys, until, traj);

% plot it
figure(2);
%subplot(2,1,1);
plot(t, y1, '-xb',...
     t, y2, '-xr',...
     val2.time, val2.X, '-xg');
grid on;

%gpu_result = readSignal('and-test2-gpu-result.txt');
%subplot(2,1,2);
%plot(gpu_result.time, gpu_result.X, '-xg');
%grid on;

% save our signals
writeSignal('and-test2-sig1.txt', [t ; y1]);
writeSignal('and-test2-sig2.txt', [t ; y2]);
writeSignal('and-test2-breach-result.txt', [val2.time ; val2.X]);



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
subplot(2,1,1);
plot(traj.time, traj.X, '-xb',...
     val.time, val.X, '-sg');
grid on;

% plot gpu result
gpu = readSignal('eventually-test-gpu-result.txt');
subplot(2,1,2);
plot(gpu.time, gpu.X, '-xg');
grid on;

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
writeSignal('eventually-test2-breach-result.txt', [val.time; val.X]);

figure(4);
subplot(2,1,1);
plot(t, y, '-xb',...
     val.time, val.X, '-sr');
grid on;

gpu = readSignal('eventually-test2-gpu-result.txt');
subplot(2,1,2);
plot(gpu.time, gpu.X, '-xg');
grid on;


%%
%% 3rd AND test (bigger one)
%%

y2_periodes = 2; %5
y1_periodes_per_y2_periode = 5; %20
points_per_seq = 100; % 1000

% timeing: (sec)
%              AND         EV
% 1 000 000:   0.610579    0.295895
%   100 000:   0.095462    0.028115
%    10 000:   0.017985    0.008726
%     1 000:   0.014277    0.006949
%       100:   0.012529    0.006845


end_seq = 2 * pi * y2_periodes * y1_periodes_per_y2_periode;
t = [0 : end_seq / points_per_seq : end_seq];   
y1 = 5 .* sin(t);

ty2 = [t(1):t(end)/y2_periodes:t(end)];
v = -5.1;
y2 = [];
for i=ty2
    y2 = [y2 v];
    v = v .* -1;
end

traj.time = t;
traj.X = [ y1 ; interp1(ty2, y2, t, 'linear') ];

tarj.param = Sys.p;
tic;
val = QMITL_Eval2raw(Sys, phi, traj);
fprintf('Execution time for AND with %i points: %g s\n',...
    points_per_seq, toc);
whos val

writeSignal(strcat('and-test3-',points_per_seq,'-sig1.txt'),...
    [traj.time, traj.X(1,:)]);
writeSignal(strcat('and-test3-',points_per_seq,'-sig2.txt'),...
    [traj.time, traj.X(2,:)]);
writeSignal(strcat('and-test3-',points_per_seq,'-breach-reslut.txt'),...
    [ val.time ; val.X ]);

clear val
clear traj
clear y2
clear y1
clear ty2

%figure(5);
%plot(t, traj.X(1,:), '-b',...
%     t, traj.X(2,:), '-r',...
%     val.time, val.X, '-g');
%grid on;

y2 = [ 5.1 0.1 ];
ty2 = [ t(1) t(end) ];
y2i = interp1(ty2, y2, t, 'linear');

y = 5 .* y2i .* sin(t);

traj.time = t;
traj.X = y;
traj.param = Sys.p;

tic;
val = QMITL_Eval2raw(Sys, phi_ev, traj);
fprintf('Execution time for EV with %i points: %g s\n', ...
    points_per_seq, toc);

whos val

writeSignal(strcat('eventually-test3-',points_per_seq,'-sig.txt'),...
    [ traj.time ; traj.X ]);
writeSignal(strcat('eventually-test3-',points_per_seq,'-breach-result.txt'),...
    [ val.time ; val.X ]);

%figure(6);
%plot(traj.time, traj.X, '-b',...
%     val.time, val.X, '-g');
%grid on;


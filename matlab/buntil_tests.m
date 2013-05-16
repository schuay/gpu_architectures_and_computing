if ~ exist('Sys', 'var')
    if ~ exist('init_test_cases', 'file')
        addpath('util')
    end
    init_test_cases
end

sig1 = readSignal([TRACE_PATH 'sig01.trace']);
sig2 = readSignal([TRACE_PATH 'sig02.trace']);

% breach seems to do not support 2 input signals with different
% time scales. Thus we have to merge the two time scales and
% interpolate the values
traj.time = union(sig1.time, sig2.time);
traj.X = [ ...
    interp1(sig1.time, sig1.X, traj.time, 'linear') ;...
    interp1(sig2.time, sig2.X, traj.time, 'linear')  ...
    ];

traj.param = Sys.p;

% taken from init_examples.m
P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

% define our formula
% and
QMITL_Formula('phi', '(s1[t] > 0) until_[0.5,1.5] (s2[t] > 0)');

QMITL_Eval(Sys, phi, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, phi, traj);

% save the result
%writeSignal([TRACE_PATH 'and_sig01_sig02.breach.trace'],...
%    [val2.time; val2.X]);
val2.name = [TRACE_PATH 'buntil_0.5_1.5_sig01_sig02.breach.trace'];
resultArray = [resultArray val2];

% draw the signal traces
figure(1);
subplot(2,1,1);
plot(sig1.time, sig1.X, '-sb',...
     sig2.time, sig2.X, '-sr',...
     val2.time, val2.X, '-sg');
title('sig1 (blue), sig2 (red), (s1[t] > 0) until (s2[t] > 0) (green)');
axis([0 3 -2 4]);
grid on;

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
subplot(2,1,1);
plot(t, y1, '-xb',...
     t, y2, '-xr',...
     val2.time, val2.X, '-xg');
axis([0 10 -0.8 1.2]);
grid on;

% save our signals
writeSignal([TRACE_PATH 'sig03.trace'], [t ; y1]);
writeSignal([TRACE_PATH 'sig04.trace'], [t ; y2]);
%writeSignal('and-test2-breach-result.txt', [val2.time ; val2.X]);

val2.name = [TRACE_PATH 'buntil_0.5_1.5_sig03_sig04.breach.trace'];
resultArray = [resultArray val2];

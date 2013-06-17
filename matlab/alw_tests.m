if ~ exist('Sys', 'var')
    if ~ exist('init_test_cases', 'file')
        addpath('util')
    end
    init_test_cases
end

sig1 = readSignal([TRACE_PATH 'sig05.trace']);

traj.time = sig1.time;
traj.X = sig1.X;

traj.param = Sys.p;
P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

% define formula
QMITL_Formula('phi_not', 'alw (s1[t] > 0)');

QMITL_Eval(Sys, phi_not, P, traj);
val = QMITL_Eval2raw(Sys, phi_not, traj);

% save result
%writeSignal('eventually-test-breach-result.txt', [val.time ; val.X]);
val.name = [TRACE_PATH 'alw_sig05.breach.trace'];
resultArray = [resultArray val];

% plot the signals
figure (1);
subplot(2,1,1);
plot(sig1.time, sig1.X, '-sb',...
     val.time, val.X, '-sg');
title('sig1 (blue), "alw (s1[t] > 0)" (green)');
grid on;
%%
%% second test
%%
t = [0:0.4:3*pi];
a = 0.5 .* cos(t/(pi)) + 0.7;
y = a .* sin(t);

traj.time = t;
traj.X = y;
traj.param = Sys.p;
P.traj = traj;

QMITL_Eval(Sys, phi_not, P, traj);
val = QMITL_Eval2raw(Sys, phi_not, traj);

% save signals
writeSignal([TRACE_PATH 'sig06.trace'], [t ; y]);
val.name = [TRACE_PATH 'alw_sig06.breach.trace'];
resultArray = [resultArray val];
figure (2);
subplot(2,1,1);
plot(t, y, '-sb',...
     val.time, val.X, '-sg');
title('sig1 (blue), "alw (s1[t] > 0)" (green)');
grid on;

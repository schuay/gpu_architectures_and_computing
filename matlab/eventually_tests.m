if ~ exist('Sys', 'var')
    if ~ exist('init_test_cases', 'file')
        addpath('util')
    end
    init_test_cases
end


%%
%% first eventually test
%%
fprintf(' EVENTUALLY test 1\n');


sig1 = readSignal([TRACE_PATH 'sig05.trace']);

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
axis([0 3 -1 3.5]);
grid on;

% plot gpu result
gpuacTrace = [TRACE_PATH 'ev_sig05.gpuac.trace'];
if exist(gpuacTrace, 'file')
    gpu = readSignal(gpuacTrace);
    subplot(2,1,2);
    plot(gpu.time, gpu.X, '-xg');
    axis([0 3 -1 3.5]);
    grid on;
end

% save result
%writeSignal('eventually-test-breach-result.txt', [val.time ; val.X]);
val.name = [TRACE_PATH 'ev_sig05.breach.trace'];
resultArray = [resultArray val];


%%
%% second eventually test
%%
fprintf(' EVENTUALLY test 2\n');

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
writeSignal([TRACE_PATH 'sig06.trace'], [t ; y]);
val.name = [TRACE_PATH 'ev_sig06.breach.trace'];
resultArray = [resultArray val];

figure(4);
subplot(2,1,1);
plot(t, y, '-xb',...
     val.time, val.X, '-sr');
axis([0 10 -1 1.5]);
grid on;

gpuacTrace = [TRACE_PATH 'ev_sig06.gpuac.trace'];
if exist(gpuacTrace, 'file')
    gpu = readSignal(gpuacTrace);
    subplot(2,1,2);
    plot(gpu.time, gpu.X, '-xg');
    axis([0 10 -1 1.5]);
    grid on;
end


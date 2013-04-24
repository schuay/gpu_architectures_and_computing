%%
%% simple benchmark
%% TODO: make this more robust
%%

% init
% clear workspace
clear all

%addpath('../..');   % path to breach
InitBreach;
vars = {'s1', 's2'};
params = {'p0', 'p1','p2', 'p3'};                                                            % generic parameters  
p0 = [ 0 0, 0 0 0 0 ];
Sys = CreateExternSystem('myBenchmark1', vars, params, p0);
P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

% define our formulas
QMITL_Formula('phi', '(s1[t] > 0) and (s2[t] > 0)');
QMITL_Formula('phi_ev', 'ev (s1[t] > 0)');


y2_periodes = 2; %5
y1_periodes_per_y2_periode = 5; %20
points_per_seq = 100; % 1000

% timeing: (sec)
%              AND         EV
% 1 000 000:   0.610 579    0.295 895
%   100 000:   0.095 462    0.028 115
%    10 000:   0.017 985    0.008 726
%     1 000:   0.014 277    0.006 949
%       100:   0.012 529    0.006 845


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

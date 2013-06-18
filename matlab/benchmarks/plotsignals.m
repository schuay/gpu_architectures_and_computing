
if ~ exist('Sys', 'var')
    if ~ exist('init_test_cases', 'file')
        addpath('../util')
    end
    init_test_cases
end



sig1 = createSig1(2, 7, 1000);

traj.time = sig1.t;
traj.X = [ sig1.y1 ; sig1.y2 ];
traj.param = Sys.p;

P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

QMITL_Formula('phiand', '(s1[t] > 0) and (s2[t] > 0)');
QMITL_Formula('phiuntil', '(s1[t] > 0) until (s2[t] > 0)');

QMITL_Eval(Sys, phi, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, phiand, traj);
valuntil = QMITL_Eval2raw(Sys, phiuntil, traj);

figure(1);
subplot(2,1,1);
plot(sig1.t, sig1.y1, '-b',...
     sig1.t, sig1.y2, '-r',...
     val2.time, val2.X, '-g');
 
title('bechmark sig 1 (blue), sig 2 (red), (s1[t] > 0) and (s2[t] > 0) (green)');
grid on;

%figure(2);
subplot(2,1,2);
plot(sig1.t, sig1.y1, '-b',...
     sig1.t, sig1.y2, '-r',...
     valuntil.time, valuntil.X, '-g');
%axis 
title('bechmark sig 1 (blue), sig 2 (red), (s1[t] > 0) until (s2[t] > 0) (green)');
grid on;



sig2 = createSig2(15, 1000);

traj.time = sig2.t;
traj.X = [ sig2.y1 ];
traj.param = Sys.p;

P = CreateParamSet(Sys);
P.pts = traj.param';
P.traj = traj;

QMITL_Formula('phiev', 'ev (s1[t] > 0)');
QMITL_Formula('phialw', 'alw (s1[t] > 0)');

%QMITL_Eval(Sys, phiev, P, traj);  % for some reason we have to call this first
val2 = QMITL_Eval2raw(Sys, phiev, traj);
val3 = QMITL_Eval2raw(Sys, phialw, traj);



figure(3);
%subplot(2,1,1);
plot(sig2.t, sig2.y1, '-b',...
     val2.time, val2.X, '-g',...
     val3.time, val3.X, '-y');
title('benchmark sig 3 (blue), eventually (s1[t] > 0) (green), always (s1[t] > 0) (yellow)');
grid on;


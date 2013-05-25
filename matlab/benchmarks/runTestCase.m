function [ result ] = runTestCase( Sys, formular, traj )
%RUNTESTCASE run one testcase on the defined input parameter
%   run testcase with the given formular and the defined traj array
%
%   Sys       Breach System
%   formular  Breach STL formular for this testcase
%   traj      signal traces 
%             traj.X    ... signals
%             traj.time ... time domain
%

    QMITL_Formula('stl_formular', formular);
            
    traj.param = Sys.p;
            
    P = CreateParamSet(Sys);
    P.pts = traj.param';
    P.traj = traj;
            
    QMITL_Eval(Sys, stl_formular, P, traj); % for some reason we have to call this first
    
    tic;
    result.val = QMITL_Eval2raw(Sys, stl_formular, traj);
    result.time = toc;
    
end


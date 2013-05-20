function [ ] = benchmark ( test )
% BENCHMARK runs the given benchmark and measures the time for it
%   run the specified benchmark and measure the time breach
%   needs for it
%
%   test    testcase name

    % initialization stuff
    InitBreach;
    vars = {'s1', 's2'};
    params = {'p0', 'p1','p2', 'p3'};  % generic parameters  
    p0 = [ 0 0, 0 0 0 0 ];
    Sys = CreateExternSystem('myTest', vars, params, p0);

    % define path to trace files
    TRACE_PATH = 'traces/';
  
    
    switch test
        case 'simpleAND'
            QMITL_Formula('s1_and_s2', '(s1[t] > 0) and (s2[t] > 0)');
            sig1 = createSig1(5, 25, 1000);
            
            traj.time = sig1.t;
            traj.X = [ sig1.y1 ; sig1.y2 ];
            traj.param = Sys.p;
            
            P = CreateParamSet(Sys);
            P.pts = traj.param';
            P.traj = traj;
            
            QMITL_Eval(Sys, s1_and_s2, P, traj); % for some reason we have to call this first
            tic;
            val = QMITL_Eval2raw(Sys, s1_and_s2, traj);
            fprintf('testcase %s, finished. time: %g s\n',...
                test, toc);
    
        otherwise
            fprintf(2, 'test case "%s" not defined\n', test);
    end
end
function [ returncode ] = benchmark ( test, signalFileNameBase, resultFileName )
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
    %TRACE_PATH = 'traces/';
    % define default return code
    % returncodes: 0 ... ok
    %              1 ... test case not defined
    returncode = 0;
    
    switch test
        case 'AND-1000'
            sig1 = createSig1(5, 25, 1000);
            
            traj.time = sig1.t;
            traj.X = [ sig1.y1 ; sig1.y2 ];
            result = runTestCase(Sys, '(s1[t] > 0) and (s2[t] > 0)', traj);

            fprintf('testcase %s, finished. time: %g s\n',...
                    test, result.time);

        case 'AND-10000'
            sig1 = createSig1(10, 50, 10000);
            
            traj.time = sig1.t;
            traj.X = [ sig1.y1 ; sig1.y2 ];
            result = runTestCase(Sys, '(s1[t] > 0) and (s2[t] > 0)', traj);

            fprintf('testcase %s, finished. time: %g s\n',...
                    test, result.time);

        case 'AND-100000'
            sig1 = createSig1(10, 50, 100000);
            
            traj.time = sig1.t;
            traj.X = [ sig1.y1 ; sig1.y2 ];
            result = runTestCase(Sys, '(s1[t] > 0) and (s2[t] > 0)', traj);

            fprintf('testcase %s, finished. time: %g s\n',...
                    test, result.time);

        otherwise
            returncode = 1;
            fprintf(2, 'test case "%s" not defined\n', test);
            return
    end
    
    if nargin > 1
        [rows cols] = size(traj.X);
        
        for i = 1:rows
            filename = sprintf('%s_sig%d.trace', signalFileNameBase, i);
            writeSignal( filename, [ traj.time ; traj.X(i,:) ] );
        end
    end
    
    if nargin > 2
        filename = sprintf('%s.breach.trace', resultFileName);
        writeSignal( filename, [ result.val.time ; result.val.X ] );
    end
end
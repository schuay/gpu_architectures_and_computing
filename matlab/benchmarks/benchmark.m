function [ returncode ] = benchmark ( test, formula, signalFileNameBase,...
                                      resultFileName )
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
            [result, traj] = binary_op_test(Sys, createSig1(5, 25, 1000), formula);
        case 'AND-10000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 10000), formula);
        case 'AND-100000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 100000), formula);
        case 'AND-1000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 1000000), formula);
        case 'AND-5000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 5000000), formula);
            
        case 'OR-1000'
            [result, traj] = binary_op_test(Sys, createSig1(5, 25, 1000), formula);
        case 'OR-10000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 10000), formula);
        case 'OR-100000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 100000), formula);
        case 'OR-1000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 1000000), formula);
        case 'OR-5000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 5000000), formula);

        case 'EVTL-1000'
            [result, traj] = unary_op_test(Sys, createSig2( 10, 1000 ), formula);
        case 'EVTL-10000'
            [result, traj] = unary_op_test(Sys, createSig2( 20, 10000 ), formula);
        case 'EVTL-100000'
            [result, traj] = unary_op_test(Sys, createSig2( 40, 100000 ), formula);
        case 'EVTL-1000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 1000000 ), formula);
        case 'EVTL-5000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 5000000 ), formula);
            
        case 'EVTL_2_10-1000'
            [result, traj] = unary_op_test(Sys, createSig2( 10, 1000 ), formula);
        case 'EVTL_10_100-10000'
            [result, traj] = unary_op_test(Sys, createSig2( 20, 10000 ), formula);
        case 'EVTL_50_500-100000'
            [result, traj] = unary_op_test(Sys, createSig2( 40, 100000 ), formula);
        case 'EVTL_100_1000-1000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 1000000 ), formula);
        case 'EVTL_100_1000-5000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 5000000 ), formula);
            
        case 'ALW-1000'
            [result, traj] = unary_op_test(Sys, createSig2( 10, 1000), formula);
        case 'ALW-10000'
            [result, traj] = unary_op_test(Sys, createSig2( 20, 10000), formula);
        case 'ALW-100000'
            [result, traj] = unary_op_test(Sys, createSig2( 40, 100000), formula);
        case 'ALW-1000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 1000000), formula);
        case 'ALW-5000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 5000000), formula);
            
         case 'ALW_2_10-1000'
            [result, traj] = unary_op_test(Sys, createSig2( 10, 1000), formula);
        case 'ALW_10_100-10000'
            [result, traj] = unary_op_test(Sys, createSig2( 20, 10000), formula);
        case 'ALW_50_500-100000'
            [result, traj] = unary_op_test(Sys, createSig2( 40, 100000), formula);
        case 'ALW_100_1000-1000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 1000000), formula);
        case 'ALW_100_1000-5000000'
            [result, traj] = unary_op_test(Sys, createSig2( 80, 5000000), formula);
            
         
        case 'UNTIL-1000'
            [result, traj] = binary_op_test(Sys, createSig1(5, 25, 1000), formula);
        case 'UNTIL-10000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 10000), formula);
        case 'UNTIL-100000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 100000), formula);
        case 'UNTIL-1000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 1000000), formula);
        case 'UNTIL-5000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 5000000), formula);

        case 'UNTIL_2_10-1000'
            [result, traj] = binary_op_test(Sys, createSig1(5, 25, 1000), formula);
        case 'UNTIL_10_100-10000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 10000), formula);
        case 'UNTIL_50_500-100000'
            [result, traj] = binary_op_test(Sys, createSig1(10, 50, 100000), formula);
        case 'UNTIL_100_1000-1000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 1000000), formula);
        case 'UNTIL_100_1000-5000000'
            [result, traj] = binary_op_test(Sys, createSig1(20, 80, 5000000), formula);

        
        case 'AND-rand-1000'
            [result, traj] = binary_op_test(Sys, createRandomSig(1000, 10), formula);
        case 'AND-rand-10000'
            [result, traj] = binary_op_test(Sys, createRandomSig(10000, 10), formula);
        case 'AND-rand-100000'
            [result, traj] = binary_op_test(Sys, createRandomSig(100000, 10), formula);
        case 'AND-rand-1000000'
            [result, traj] = binary_op_test(Sys, createRandomSig(1000000, 100), formula);
        case 'AND-rand-5000000'
            [result, traj] = binary_op_test(Sys, createRandomSig(5000000, 100), formula);
            
        case 'OR-rand-1000'
            [result, traj] = binary_op_test(Sys, createRandomSig(1000, 10), formula);
        case 'OR-rand-10000'
            [result, traj] = binary_op_test(Sys, createRandomSig(10000, 10), formula);
        case 'OR-rand-100000'
            [result, traj] = binary_op_test(Sys, createRandomSig(100000, 10), formula);
        case 'OR-rand-1000000'
            [result, traj] = binary_op_test(Sys, createRandomSig(1000000, 100), formula);
        case 'OR-rand-5000000'
            [result, traj] = binary_op_test(Sys, createRandomSig(5000000, 100), formula);
            
        case 'EVTL-rand-1000'
            [result, traj] = unary_op_test(Sys, createRandomSig(1000, 10), formula);
        case 'EVTL-rand-10000'
            [result, traj] = unary_op_test(Sys, createRandomSig(10000, 10), formula);
        case 'EVTL-rand-100000'
            [result, traj] = unary_op_test(Sys, createRandomSig(100000, 10), formula);
        case 'EVTL-rand-1000000'
            [result, traj] = unary_op_test(Sys, createRandomSig(1000000, 100), formula);
        case 'EVTL-rand-5000000'
            [result, traj] = unary_op_test(Sys, createRandomSig(5000000, 100), formula);
            
        case 'NOT-rand-1000'
            [result, traj] = unary_op_test(Sys, createRandomSig(1000, 10), formula);
        case 'NOT-rand-10000'
            [result, traj] = unary_op_test(Sys, createRandomSig(10000, 10), formula);
        case 'NOT-rand-100000'
            [result, traj] = unary_op_test(Sys, createRandomSig(100000, 10), formula);
        case 'NOT-rand-1000000'
            [result, traj] = unary_op_test(Sys, createRandomSig(1000000, 100), formula);
        case 'NOT-rand-5000000'
            [result, traj] = unary_op_test(Sys, createRandomSig(5000000, 100), formula);
            
        
        case 'UNTIL_1_2-1000'
            [result, traj] = binary_op_test(Sys, createSig1(5, 25, 1000), formula);
        
            
        otherwise
            returncode = 1;
            fprintf(2, 'test case "%s" not defined\n', test);
            return
    end
    
    printResult(test, result);

    
    if nargin > 2
        [rows cols] = size(traj.X);
        
        for i = 1:rows
            filename = sprintf('%s_sig%d.trace', signalFileNameBase, i);
            writeSignal( filename, [ traj.time ; traj.X(i,:) ] );
        end
    end
    
    if nargin > 3
        filename = sprintf('%s.breach.trace', resultFileName);
        writeSignal( filename, [ result.val.time ; result.val.X ] );
    end
end

function [ result ] = runTestCase( Sys, formula, traj )
%RUNTESTCASE run one testcase on the defined input parameter
%   run testcase with the given formula and the defined traj array
%
%   Sys       Breach System
%   formula   Breach STL formula for this testcase
%   traj      signal traces 
%             traj.X    ... signals
%             traj.time ... time domain
%

    QMITL_Formula('stl_formula', formula);
            
    traj.param = Sys.p;
            
    P = CreateParamSet(Sys);
    P.pts = traj.param';
    P.traj = traj;
            
    QMITL_Eval(Sys, stl_formula, P, traj); % for some reason we have to call this first
    
    tic;
    result.val = QMITL_Eval2raw(Sys, stl_formula, traj);
    result.time = toc;
    
end



function [ ] = printResult(test, result)
    fprintf('Breach: testcase %s, finished. time: %g s\n',...
            test, result.time);
    
end


function [result, t] = unary_op_test(Sys, sig1, op)
    traj.time = sig1.t;
    traj.X = sig1.y1;
    result = runTestCase(Sys, [ op ' (s1[t] > 0)'], traj);
    t = traj;
end

    
function [result, t] = binary_op_test(Sys, sig1, op)
    traj.time = sig1.t;
    traj.X = [ sig1.y1 ; sig1.y2 ];
    result = runTestCase(Sys, ['(s1[t] > 0) ' op ' (s2[t] > 0)'], traj);
    t = traj;
end

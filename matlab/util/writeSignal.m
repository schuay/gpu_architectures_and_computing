function [ out ] = writeSignal( filename, v )
%WRITESIGNAL Summary of this function goes here
%   Detailed explanation goes here
    dlmwrite(filename, v', 'delimiter', ' ', 'precision', '%10.12f');
    out = v;
end


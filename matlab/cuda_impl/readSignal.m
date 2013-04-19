function [ out ] = readSignal( filename )
%READSIGNAL Summary of this function goes here
%   Detailed explanation goes here

    v = dlmread(filename);
    out = v';

end


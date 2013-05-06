function [ out ] = readSignal( filename )
%READSIGNAL Summary of this function goes here
%   Detailed explanation goes here

    v = dlmread(filename);
    v = v';
    out.time = v(1,:);
    out.X = v(2,:);
    out.dX = v(3,:);
end


function [ result ] = createRandomSig( num_points, time_end )
%CREATERANDOMSIG1 create two signals with random values
%   num_points    number of points to calculate
%   time_end      end of the time periode (staring from 0)
%

    result.t = 0 : time_end / num_points : time_end;
    result.y1 = rand(1, length(result.t));
    result.y2 = rand(1, length(result.t));
end
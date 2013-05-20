function [ result ] = createSig2( y1_per, num_points )
%CERATESIG2 create a benchmark signal for eventually
%   y1       sinus wave with decreasing amplitude (starting 
%            from 5 till 0.1)
%   
%   y1_per     number of periodes
%   num_points number of points for the whole signal 

    end_seq = 2 * pi * y1_per;
    t = [ 0 : end_seq / num_points : end_seq ];
    
    y2 = [ 5.1 0.1 ];
    y2_t = [ t(1) t(end) ];
    
    y = interp1(y2_t, y2, t, 'linear') .* sin(t);
    
    result.y1 = y;
    result.t = t;
end


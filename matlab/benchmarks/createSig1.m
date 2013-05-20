function [ output_args ] = createSig1( y2_per,...
                                           y1_per_in_y2,...
                                           num_points)
%createSigPair1 create tow signals based on the given parameter
%   y1 is a sinus wave with amplitude 5
%   y2 is a triang.signal between -5.1 and +5.1
%   both signals have a common time domain
%
%   y2_per         defines how many times one periode the created 
%                  signal should include
%   y1_per_in_y2   how many periodes of y1 should be within one
%                  y2 periode
%   num_points     defines the resulution of the signal, number
%                  of points the signal should include
%

    % end of our time domain
    end_seq = 2 * pi * y2_per * y1_per_in_y2;
        
    % time scale
    t = [ 0 : end_seq / num_points : end_seq ];   
    y1 = 5 .* sin(t);

    % time scale for y2 creation
    t_y2 = [t(1):t(end)/y2_per:t(end)];
    v = -5.1;
    y2 = [];    
    for i=t_y2
        y2 = [y2 v];
        v = v .* -1;
    end

    output_args.y1 = y1;
    % interpolate y2 to the common time domain t
    output_args.y2 = interp1(t_y2, y2, t, 'linear');
    output_args.t = t;

end


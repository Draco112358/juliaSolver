function round_ud(args...)
    nargin = length(args)
    if nargin<2
        out=round.(args[1]);
    else
        out=round.(args[1]*10^args[2])/10^args[2];
    end
    return out

end
%{
# Movie parameters for parametric models
-> stimulus.Movie
-----
params : longblob
%}

classdef MovieParams < dj.Part
    
    properties(SetAccess=protected)
        master= stimulus.Movie
    end
    
end
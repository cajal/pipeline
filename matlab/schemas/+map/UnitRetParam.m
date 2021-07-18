%{
    # parameters for computing unit retinotopy
    unitret_id  : int           # method id for generating unit retinotopy
    ---
    sigma       : decimal(4,2)         # sigma used for gaussian smoothing, 0 for no smoothing
%}
classdef UnitRetParam < dj.Lookup
    properties
        contents = {
            1 0
            2 0.5
            3 1
            4 1.5
            5 2
        }
    end
end
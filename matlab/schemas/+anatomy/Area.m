%{
#Brain Area
 brain_area : varchar(12)   # short brain area name
%}
classdef Area < dj.Lookup
    properties
         contents = {
            'V1'
            'P'
            'POR'
            'PM'
            'AM'
            'A'
            'RL'
            'AL'
            'LI'
            'LM'
        }
    end
end
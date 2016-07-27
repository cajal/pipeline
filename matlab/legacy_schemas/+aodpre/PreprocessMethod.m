%{
aodpre.PreprocessMethod (lookup) # trace preprocessing
preprocess_id   : tinyint                # pre
---
preprocess_name             : char(8)                       # brief description to be used in switch statements, etc
%}

classdef PreprocessMethod < dj.Relvar
    methods
        function fill(self)
            self.inserti({
                0   'raw'
                1   '-1pc'
                })
        end
    end
end

%{
rf.Lens (lookup) # objective lens list
lens : char(4)  # objective lens
-----
%}

classdef Lens < dj.Relvar
    methods
        function fill(self)
            self.insert({
                '16x'
                '25x'
                })
        end
    end
end
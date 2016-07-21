%{
pre.Slice (lookup) # slice numbers
slice : tinyint   # slice number
-----
%}

classdef Slice < dj.Relvar
    methods
        function fill(self)
            for i=1:16
                self.inserti({i})
            end
        end
    end
end
%{
pre.SegmentMethod (lookup) # cell segmentation method
segment_method         : tinyint # id of the method 
-----
method_name            : char(8) # name of the method for switch statements
%}

classdef SegmentMethod < dj.Relvar
    methods 
        function fill(self)
            self.inserti({
                1 'manual'
                2 'nmf'
                });
        end
    end
end
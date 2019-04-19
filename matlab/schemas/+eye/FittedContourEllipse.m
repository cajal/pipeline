%{
# 
-> eye.FittedContour
frame_id                    : int                           # frame id with matlab based 1 indexing
---
center=null                 : tinyblob                      # center of the ellipse in (x, y) of image
major_r=null                : float                         # major radius of the ellipse
%}


classdef FittedContourEllipse < dj.Computed
    % Implemented in Python
    methods(Access=protected)
        function makeTuples(self, key)
        end
    end
end
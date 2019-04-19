%{
# 
-> eye.ManuallyTrackedContours
frame_id                    : int                           # frame id with matlab based 1 indexing
---
contour=null                : longblob                      # eye contour relative to ROI
%}


classdef ManuallyTrackedContoursFrame < dj.Manual
    % Implemented in Python
end
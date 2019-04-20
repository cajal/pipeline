%{
# 
-> eye.ManuallyTrackedContoursFrame
---
roi=null                    : longblob                      # roi of eye
gauss_blur=null             : float                         # bluring of ROI
exponent=null               : tinyint                       # exponent for contrast enhancement
dilation_iter=null          : tinyint                       # number of dilation and erosion operations
min_contour_len=null        : tinyint                       # minimal contour length
running_avg_mix=null        : float                         # weight a in a * current_frame + (1-a) * running_avg
%}


classdef ManuallyTrackedContoursParameter < dj.Manual
    % Implemented in Python
end
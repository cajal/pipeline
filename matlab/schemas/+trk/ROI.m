%{
trk.ROI (manual) # table that stores the correct ROI of the Eye in the video
-> rf.Eye
x_roi           : int                    # x coordinate of roi
y_roi           : int                    # y coordinate of roi
---
%}


classdef ROI < dj.Relvar
end
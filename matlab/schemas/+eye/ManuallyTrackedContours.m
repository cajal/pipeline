%{
# 
-> eye.Eye
---
tracking_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
min_lambda=null             : float                         # minimum mixing weight for current frame in running average computation (1 means no running avg was used)
%}


classdef ManuallyTrackedContours < dj.Manual
    % Implemented in Python
end
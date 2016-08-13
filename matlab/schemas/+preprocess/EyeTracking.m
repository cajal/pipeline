%{
preprocess.EyeTracking (computed) # 
-> preprocess.Eye
-> preprocess.TrackingParameters
---
tracking_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
%}


classdef EyeTracking < dj.Relvar 

end
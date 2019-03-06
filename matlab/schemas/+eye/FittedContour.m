%{
# 
-> eye.ManuallyTrackedContours
---
tracking_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
%}


classdef FittedContour < dj.Computed
% Implemented in Python
    methods(Access=protected)
        function makeTuples(self, key)
        end
    end
end
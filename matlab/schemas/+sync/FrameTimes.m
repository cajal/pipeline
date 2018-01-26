%{
# Frame times with 2pMaster clock
-> experiment.Scan
---
frame                       : double                        # Frame idx of Scan Image
ts                          : timestamp                     # real time of frame
counter_ts                  : double                        # counter time
%}

classdef FrameTimes < dj.Manual
end
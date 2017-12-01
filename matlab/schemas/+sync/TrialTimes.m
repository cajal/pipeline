%{
# Trial times in 2pMaster clock
-> experiment.Scan
---
trial                       : double                        # Trial idx of stimulus
ts                          : timestamp                     # real time of frame
counter_ts                  : double                        # counter time
%}

classdef TrialTimes < dj.Manual
end
%{
vis.MovieStillCond (manual) # a still frame condition
-> vis.Condition
---
-> vis.MovieStill
pre_blank_period            : float                         # (s)
duration                    : float                         # (s)
%}


classdef MovieStillCond < dj.Relvar
end
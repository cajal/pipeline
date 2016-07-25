%{
vis.Condition (manual) # trial condition -- one condition per trial. All stimulus conditions refer to Condition.
-> vis.Session
cond_idx        : smallint unsigned      # condition index
---
%}


classdef Condition < dj.Relvar
end
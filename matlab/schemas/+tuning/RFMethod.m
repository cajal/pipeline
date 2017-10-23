%{
# 
rf_method                   : tinyint                       # rf computation method
---
stim_selection              : varchar(64)                   # stim to use.  If no stimulus, the parts will be missing.
algorithm                   : varchar(30)                   # short name for the computational approach for computing RF
%}


classdef RFMethod < dj.Lookup
end
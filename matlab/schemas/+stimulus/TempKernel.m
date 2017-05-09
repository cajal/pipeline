%{
# choice of temporal kernel for smoothing
temp_kernel : varchar(16) 
---
temp_kernel_comment="" : varchar(255)
%}

classdef TempKernel < dj.Lookup
    
    properties 
        contents = {
            'hamming'       'good smoothing for given bandwidth'
            'half-hamming'  'causal, preserves all frequencies'
            }
    end
end
%{
# 
spike_method                : tinyint                       # spike inference method
---
name                        : varchar(16)                   # short name to identify the spike inference method
details                     : varchar(255)                  # more details
language                    : enum('matlab','python')       # implementation language
%}


classdef SpikeMethod < dj.Lookup
end
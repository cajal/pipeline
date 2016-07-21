%{
preprocess.SpikeMethod (lookup) # 
spike_method    : smallint               # spike inference method
---
spike_method_name           : varchar(16)                   # short name to identify the spike inference method
spike_method_details        : varchar(255)                  # more details about
language                    : enum('matlab','python')       # implementation language
%}


classdef SpikeMethod < dj.Relvar
end
%{
tuning.CaKernel (lookup) # options for calcium response kinetics.
kernel          : tinyint                # calcium option number
---
transient_shape             : enum('exp','onAlpha')         # calcium transient shape
latency=0                   : float                         # (s) assumed neural response latency
tau                         : float                         # (s) time constant (used by some integration functions
explanation                 : varchar(255)                  # explanation of calcium response kinents
%}


classdef CaKernel < dj.Relvar
end

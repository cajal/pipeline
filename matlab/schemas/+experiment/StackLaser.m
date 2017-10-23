%{
# laser parameters for the stack
-> experiment.Stack
---
wavelength                  : int                           # (nm)
max_power                   : float                         # (mW) to brain
gdd                         : float                         # gdd setting
%}


classdef StackLaser < dj.Manual
end
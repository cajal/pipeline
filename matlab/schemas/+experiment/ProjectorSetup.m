%{
# projector set up
-> Projector
-> ProjectorConfig
-> Rig
---
display_width       : float         # projected display width in cm
display_height      : float         # projected display height in cm
target_distance     : float         # distance from mouse to the display in cm
%}

classdef ProjectorSetup < dj.Lookup
end
%{
vis.FlashingBar (manual) # flashing bar
-> vis.Condition
---
pre_blank                   : float                         # (s) blank period preceding trials
luminance                   : float                         # (cd/m^2) mid-value luminance
contrast                    : float                         # (0-1) Michelson contrast of values 0..255
bg_color                    : tinyint unsigned              # background color 1-254
orientation                 : decimal(4,1)                  # (degrees) 0=horizontal,  90=vertical
offset                      : float                         # normalized by half-diagonal
width                       : float                         # normalized by half-diagonal
trial_duration              : float                         # (s) ON time of flashing bar
pattern_frequency           : float                         # (Hz) will be rounded to the nearest fraction of fps
%}


classdef FlashingBar < dj.Relvar
end
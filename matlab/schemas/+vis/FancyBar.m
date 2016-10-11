 %{
vis.FancyBar (manual) # Moving Bar stimulus that keeps the size and speed of the bar constant relative to the mouse’s perspective. 
-> vis.Condition
---
pre_blank                   : double                        # (s) blank period preceding trials
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
bar_width                   : float                         # Degrees
grid_width                  : float                         # Degrees
bar_speed                   : float                         # Bar speed in deg/s
flash_speed                 : float                         # cycles/sec temporal frequency of the grid flickering
style                       : enum('grating','checkerboard')# selection beween a bar with a flashing checkeboard or a moving grating
grat_width                  : float                         # in cycles/deg
grat_freq                   : float                         # in cycles/sec
axis                        : enum('vertical', 'horizontal')# the direction of bar movement
%}


classdef FancyBar < dj.Relvar
end
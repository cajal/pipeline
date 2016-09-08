 %{
vis.FancyBar (manual) # pink noise with periods of motion and orientation$
-> vis.Condition
---
pre_blank                   : double                        # (s) blank period preceding trials
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
bar_width                   : float                         # Degrees
grid_width                  : float                         # Degrees
bar_speed                   : float                         # Bar speed in °/s
flash_speed                 : float                         # cycles/sec temporal frequency of the grid flickering
grating                     : float                         # selection beween grating (1) or flashing (0)
grat_width                  : float                         # in cycles/deg
grat_speed                  : float                         # in cycles/deg
axis                        : float                         # the axis of the bar movement
%}


classdef FancyBar < dj.Relvar
end
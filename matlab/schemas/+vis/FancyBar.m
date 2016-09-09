 %{
vis.FancyBar (manual) # Moving Bar stimulus that keeps the size and speed of the bar constant relative to the mouse’s perspective. The bar can be either a Flashing checkeboard (grating=0) or a Moving grating (grating=1)
-> vis.Condition
---
pre_blank                   : double                        # (s) blank period preceding trials
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
bar_width                   : float                         # Degrees
grid_width                  : float                         # Degrees
bar_speed                   : float                         # Bar speed in deg/s
flash_speed                 : float                         # cycles/sec temporal frequency of the grid flickering
grating                     : float                         # selection beween grating (1) or flashing (0)
grat_width                  : float                         # in cycles/deg
grat_speed                  : float                         # in cycles/deg
axis                        : enum('vertical', 'horizontal')# the direction of bar movement
%}


classdef FancyBar < dj.Relvar
end
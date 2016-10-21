%{
vis.SingleDot (manual) # single dot to map receptive field
-> vis.Condition
-----
luminance                   : float                         # cd/m^2
contrast                    : float                         # Michelson's 0-1
bg_color                    : smallint                      # (0-255) the index of the background color
dot_color                   : smallint                      # (0-255) the index of the dot color
dot_x                       : float                         # (fraction of the screen diagonal) position of dot on x axis
dot_y                       : float                         # (fraction of the screen diagonal) position of dot on y axis
dot_xsize                   : float                         # (fraction of the screen diagonal) width of dots
dot_ysize                   : float                         # (fraction of the screen diagonal) height of dots
dot_shape                   : enum('rect','oval')           # shape of the dot
dot_time                    : smallint                      # time of each dot persists
%}

classdef SingleDot < dj.Relvar
end
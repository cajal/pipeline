%{
# Dot stimulus, for receptive field mapping
-> vis.Condition
---
rng_seed                    : blob                          # RNG seed
luminance                   : float                         # cd/m^2
contrast                    : float                         # Michelson's 0-1
bg_color                    : smallint                      # (0-255) the index of the background color
dot_color                   : smallint                      # (0-255) the index of the dot color
tex_xdim                    : smallint                      # texture dimension
tex_ydim                    : smallint                      # texture dimension
dot_xsize                   : smallint                      # (pixels) width of dots
dot_ysize                   : smallint                      # (pixels) height of dots
frame_downsample=60         : smallint                      # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
dots_per_frame=1            : smallint                      # number of new dots displayed in each frame
linger_frames               : smallint                      # the number of frames each dot persists
%}


classdef DotMap < dj.Manual
end
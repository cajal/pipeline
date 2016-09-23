%{
vis.Trippy (manual) # randomized curvy dynamic gratings
-> vis.Condition
---
version                     : tinyint                       # trippy version
rng_seed                    : double                        # random number generate seed
packed_phase_movie          : longblob                      # phase movie before spatial and temporal interpolation
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
tex_ydim                    : smallint                      # (pixels) texture dimension
tex_xdim                    : smallint                      # (pixels) texture dimension
duration                    : float                         # (s) trial duration
frame_downsample            : tinyint                       # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
xnodes                      : tinyint                       # x dimension of low-res phase movie
ynodes                      : tinyint                       # y dimension of low-res phase movie
up_factor                   : tinyint                       # spatial upscale factor
temp_freq                   : float                         # (Hz) temporal frequency if the phase pattern were static
temp_kernel_length          : smallint                      # length of Hanning kernel used for temporal filter. Controls the rate of change of the phase pattern.
spatial_freq                : float                         # (cy/degree) approximate max. The actual frequencies may be higher.
%}


classdef Trippy < dj.Relvar
end
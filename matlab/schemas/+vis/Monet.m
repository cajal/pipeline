%{
vis.Monet (manual) # pink noise with periods of motion and orientation$
-> vis.Condition
---
-> vis.MonetLookup
rng_seed                    : double                        # random number generate seed
luminance                   : float                         # (cd/m^2)
contrast                    : float                         # michelson contrast
tex_ydim                    : smallint                      # (pixels) texture dimension
tex_xdim                    : smallint                      # (pixels) texture dimension
spatial_freq_half           : float                         # (cy/deg) spatial frequency modulated to 50 percent
spatial_freq_stop           : float                         # (cy/deg), spatial lowpass cutoff
temp_bandwidth              : float                         # (Hz) temporal bandwidth of the stimulus
ori_on_secs                 : float                         # seconds of movement and orientation
ori_off_secs                : float                         # seconds without movement
n_dirs                      : smallint                      # number of directions
ori_bands                   : tinyint                       # orientation width expressed in units of 2*pi/n_dirs
ori_modulation              : float                         # mixin-coefficient of orientation biased noise
speed                       : float                         # (degrees/s)
frame_downsample            : tinyint                       # 1=60 fps, 2=30 fps, 3=20 fps, 4=15 fps, etc
%}


classdef Monet < dj.Relvar
end
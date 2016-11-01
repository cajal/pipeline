%{
vis.Matisse (manual)  # conditions for the matisse stimulus
-> vis.Condition
----
-> vis.BaseNoise128
pre_blank_period       :decimal(5,3)  #  (seconds)
duration               :decimal(5,3)  #  (seconds)
pattern_width          :smallint      #  pixel size of the resulting pattern
pattern_aspect         :float         #  the aspect ratio of the pattern 
pattern_upscale        :tinyint       #  integer upscale factor of the pattern
ori                    :decimal(4,1)  #  degrees. 0=horizontal, then clockwise
outer_ori_delta        :decimal(4,1)  #  degrees. Differerence of outer ori from inner.  
ori_coherence          :decimal(4,1)  #  1=unoriented noise. pi/ori_coherence = bandwidth of orientations. 
aperture_x             :decimal(4,3)  #  x position of the aperture in units of pattern widths: 0=center, 0.5=right edge 
aperture_y             :decimal(4,3)  #  y position of the aperture in units of pattern widths: 0=center, 0.5/pattern_aspect = bottom edge
aperture_r             :decimal(4,3)  #  aperture radius expressed in units pattern widths
aperture_transition    :decimal(3,3)  #  aperture transition width
annulus_alpha          :decimal(3,2)  #  aperture annulus alpha 
inner_contrast         :decimal(4,3)  #  pattern contrast in inner region
outer_contrast         :decimal(4,3)  #  pattern contrast in outer region
%}

classdef Matisse < dj.Relvar
end
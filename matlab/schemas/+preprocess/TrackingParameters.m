%{
preprocess.TrackingParameters (lookup) # table that stores the paths for the params for pupil_tracker
-> preprocess.EyeQuality
---
perc_high                   : float                         # upper percentile for bright pixels
perc_low                    : float                         # lower percentile for dark pixels
perc_weight                 : float                         # threshold will be perc_weight*perc_low + (1- perc_weight)*perc_high
relative_area_threshold     : float                         # enclosing rotating rectangle has to have at least that amount of area
ratio_threshold             : float                         # ratio of major and minor radius cannot be larger than this
error_threshold             : float                         # threshold on the RMSE of the ellipse fit
min_contour_len             : int                           # minimal required contour length (must be at least 5)
margin                      : float                         # relative margin the pupil center should not be in
contrast_threshold          : float                         # contrast below that threshold are considered dark
speed_threshold             : float                         # eye center can at most move that fraction of the roi between frames
dr_threshold                : float                         # maximally allow relative change in radius
%}


classdef TrackingParameters < dj.Relvar
end
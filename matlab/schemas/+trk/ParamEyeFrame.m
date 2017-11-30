%{
# table that stores the paths for the params for pupil_tracker
pupil_tracker_param_id      : int                           # id for param collection
---
convex_weight_high=null     : float                         # parameter for tracking
convex_weight_low=null      : float                         # parameter for tracking
thres_perc_high=null        : float                         # parameter for tracking
thres_perc_low=null         : float                         # parameter for tracking
pupil_left_limit=null       : float                         # parameter for tracking
pupil_right_limit=null      : float                         # parameter for tracking
min_radius=null             : float                         # parameter for tracking
max_radius=null             : float                         # parameter for tracking
centre_dislocation_penalty  : float                         # parameter for tracking
distance_sq_pow             : float                         # parameter for tracking
%}


classdef ParamEyeFrame < dj.Lookup
end
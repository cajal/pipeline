%{
rf.EyeFrame (manual) # eye tracking info for each frame of a movie
-> rf.Eye
frame                       : int                           # frame number in movie
---
ismanual = 0                : boolean                       # TRUE if pupil was manually detected
isblink = 0                 : boolean                       # TRUE if pupil was manually detected
detection_tried = 0         : boolean                       # TRUE if automatic detection was tried
detection_failed = 0        : boolean                       # TRUE if automatic detection failed
pupil_x = NULL              : float                         # pupil x position
pupil_y = NULL              : float                         # pupil y position
pupil_r = NULL              : float                         # pupil radius
fit_method = NULL           : enum('threshold and fit','imextendedmax and fit') # automatic method used to detect eye position
fit_params = NULL           : mediumblob                 # parameters for detection (thresholds, etc)
frame_time                  : float                         # time of frame in seconds, with same t=0 as patch and ball data
eye_frame_ts=CURRENT_TIMESTAMP    : timestamp               # automatic
%}

classdef EyeFrame < dj.Relvar
    
    
    methods
        
    end
end
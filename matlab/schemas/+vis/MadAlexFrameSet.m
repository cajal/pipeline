%{
vis.MadAlexFrameSet (lookup) # holds training and testing frames for MadAlex stimulus
frame_set_id    : tinyint                # identifier for the frame set
---
uniques                     : longblob                      # integer array with unique image ids
repeats_shared              : longblob                      # integer array with repeated image ids
repeats_nonshared           : longblob                      # integer array with repeated image ids
%}


classdef MadAlexFrameSet < dj.Relvar
end
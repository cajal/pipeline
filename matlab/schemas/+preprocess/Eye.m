%{
preprocess.Eye (imported) # eye velocity and timestamps
-> experiment.Scan
---
-> preprocess.EyeQuality
eye_roi                     : tinyblob                      # manual roi containing eye in full-size movie
eye_time                    : longblob                      # timestamps of each frame in seconds, with same t=0 as patch and ball data
total_frames                : int                           # total number of frames in movie.
eye_ts=CURRENT_TIMESTAMP    : timestamp                     # automatic
%}


classdef Eye < dj.Relvar 


end
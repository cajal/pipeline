%{
rf.Eye (imported) # eye velocity and timestamps
-> rf.Scan
---
eye_roi                     : tinyblob                      # manual roi containing eye in full-size movie
eye_total_frames=NULL       : int                           # total number of frames in movie.
eye_time                    : longblob                      # timestamps of each frame in seconds, with same t=0 as patch and ball data
eye_quality=NULL            : int                           # quality of movie (0=unusable, 1=manual pupil detection only, 2=poor pupil detection, 3=good pupil detection)
eye_ts=CURRENT_TIMESTAMP    : timestamp                     # automatic
%}

classdef Eye < dj.Relvar
    
    
    methods
        function makeTuples(self, key, eyeT)
        tuple = key;
        tuple.eye_total_frames = length(eyeT); 
        tuple.eye_time = eyeT;
        self.insert(tuple);
        end
    end
end



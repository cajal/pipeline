%{
# Population analyis for each clip
-> fuse.ScanDone
-> stimulus.Clip
---
frame_mean             : mediumblob      # frame mean
%}

classdef PopStats < dj.Imported
    
    methods
       keySource = aggr(fuse.ScanDone * stimulus.Clip & (stimulus.Movie & 'movie_class="cinema"'), stimulus.Trial, 'count(*)->n') & 'n>=2'
    end
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
           
            
            
        end
    end
    
end
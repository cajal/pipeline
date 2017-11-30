%{
# Average clip statistics
-> stimulus.MovieClip
---
center_std_mean             : float      # standard deviation of frame center means
center_mean_opticflow       : float      # average frame center optic flow
%}

classdef AvgClipStats < dj.Imported
    
    properties
        keySource = stimulus.MovieClip & movies.ClipStats & movies.OpticalFlow
    end
    
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            center_mean = fetch1(movies.ClipStats & key,'center_mean');
            center_flow = fetch1(movies.OpticalFlow & key,'center_magnitude');
            
            key.center_std_mean = std(center_mean);
            key.center_mean_opticflow = mean(center_flow);
            
            insert( obj, key );
        end
    end
    
end
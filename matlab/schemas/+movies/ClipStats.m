%{
# Clip statistics
-> stimulus.MovieClip
---
%}

classdef ClipStats < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            movie_dir = '';
            filenames = export(stimulus.MovieClip & key, movie_dir);
            
            insert( obj, key );
        end
    end
    
end
%{
vis.MovieClip (lookup) # clips from movies
-> vis.Movie
clip_number     : int                    # clip index
---
file_name                   : varchar(255)                  # full file name
clip                        : longblob                      # 
%}


classdef MovieClip < dj.Relvar
end
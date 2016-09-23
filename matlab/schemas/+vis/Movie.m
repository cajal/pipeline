%{
vis.Movie (lookup) # movies used for generating clips and stills
movie_name      : char(8)                # short movie title
---
path                        : varchar(255)                  # 
movie_class                 : enum('mousecam','object3d','madmax') # 
original_file               : varchar(255)                  # 
file_template               : varchar(255)                  # filename template with full path
file_duration               : float                         # (s) duration of each file (must be equal)
codec="-c:v libx264 -preset slow -crf 5": varchar(255)      # 
movie_description           : varchar(255)                  # full movie title
%}


classdef Movie < dj.Relvar
end
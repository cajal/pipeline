%{
# object identities in clips from movies
-> movies.Object
clip_number     : int                                       # clip index
---
file_name                   : varchar(255)                  # full file name
clip                        : longblob                      #
parent_file_name                   : varchar(255)           # parent file name
%}

classdef ObjectClip < dj.Part
    
    properties(SetAccess=protected)
        master= movies.Object
    end
    
    methods
                
    end
end


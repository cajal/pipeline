%{
# Bad movie clips
-> stimulus.MovieClip
---
judge                : enum('human','machine')  # was it a man or a machine?
notes                : varchar(256)             # a little explanation for the exclusion
%}

classdef ClipIgnore < dj.Manual
     
end
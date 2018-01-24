%{
# Inspected movie clips that are not that bad
-> stimulus.MovieClip
---
judge                : enum('human','machine')  # was it a man or a machine?
notes                : varchar(256)             # a little explanation for the inclusion process
%}

classdef ClipInspected < dj.Manual
end
%{
# list of speakers in various setups
id                      : int               # some number that identifies the speaker
rig                     : char              # setup name
location                : tinyint           # 1 = Left, 2 = Right
%}

classdef Speakers < dj.Manual
end


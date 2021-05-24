%{
# stim type, visual or auditory
stim_type_id                : tinyint               # 1=visual, 2=auditory
---
stim_type                   : varchar(32)           # name of the type
%}


classdef StimTypes < dj.Manual
end
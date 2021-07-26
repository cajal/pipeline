%{
# current piezostimulators configured in the setup
rig                     : char              # name of setup
---
speaker_id              : int               # some number that identifies the speaker
location                : tinyint           # 1=Left, 2=Right
%}

classdef CurrentPiezostimulators < dj.Lookup
end
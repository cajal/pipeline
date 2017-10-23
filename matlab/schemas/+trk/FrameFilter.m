%{
# single filters to reject frames
filter_id                   : tinyint                       # id of the filter
---
filter_name                 : char(50)                      # descriptive name of the filter
%}


classdef FrameFilter < dj.Lookup
end
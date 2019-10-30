%{
# color options for projector channels
color_id            : tinyint                   # color id
---
color               : varchar(32)               # color name
%}

classdef Projector < dj.Lookup
end
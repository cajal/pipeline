%{
# projector specifications
projector_id        : tinyint                               # projector id
---
pixel_width         : smallint                              # number of pixels in width
pixel_height        : smallint                              # number of pixels in height
%}

classdef Projector < dj.Lookup
end
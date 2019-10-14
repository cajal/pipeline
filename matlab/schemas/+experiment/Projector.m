%{
experiment.Projector (lookup) # 
projector_id        : tinyint                       # projector id
---
pixel_width         : smallint                      # number of pixels in width
pixel_height        : smallint                      # number of pixels in height
red                 : enum('UV', 'Green', 'None')   # color to be used for red channel
green               : enum('UV', 'Green', 'None')   # color to be used for green channel
blue                : enum('UV', 'Green', 'None')   # color to be used for blue channel
refresh_rate        : tinyint                       # refresh rate in Hz
%}


classdef Projector < dj.Lookup
end
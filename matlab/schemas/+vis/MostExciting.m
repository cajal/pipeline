%{
# most exciting images generated from multiple models
mei_id                      : int                           # image id
-> experiment.BrainArea
-> experiment.Layer
model                       : varchar(12)                   # 
---
image                       : longblob                      # image
std                         : float                         # std of image
mean                        : float                         # mean of image
%}


classdef MostExciting < dj.Manual
end
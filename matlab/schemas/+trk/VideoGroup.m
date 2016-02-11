%{
trk.VideoGroup (lookup) # table that groups videos into groups that can be tracked by the same SVM
videogroup_id   : tinyint                # id of the video group
---
group_name                  : char(20)                      # name of the group
%}


classdef VideoGroup < dj.Relvar
end
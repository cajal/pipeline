%{
# current setup parameters for each rig
-> experiment.Rig
---
monitor_distance            : float                         # (cm) eye-to-monitor distance
monitor_size=19             : float                         # (inches) size diagonal dimension
monitor_aspect=1.25         : float                         # physical aspect ratio of monitor
resolution_x=1280           : smallint                      # (pixels)
resolution_y=1024           : smallint                      # display resolution along y
%}


classdef Setup < dj.Lookup
end
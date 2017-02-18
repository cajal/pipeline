%{
experiment.DisplayGeometry (manual) # geometry
-> experiment.Session
---
monitor_distance            : decimal(4,1)                  # (cm) eye-to-monitor distance
monitor_size                : decimal(5,2)                  # (inches) size diagonal dimension
monitor_aspect              : decimal(4,3)                  # physical aspect ratio of monitor
resolution_x                : smallint                      # (pixels) display resolution along x
resolution_y                : smallint                      # (pixels) display resolution along y
fps                         : decimal(5,2)                  # display refresh rate
%}

classdef DisplayGeometry < dj.Relvar
end

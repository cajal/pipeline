%{
vis.Session (manual) # Visual stimulus session, populated by the stimulus program.
-> mice.Mice
psy_id          : smallint unsigned      # unique psy session number
---
stimulus="unused"           : varchar(255)                  # experiment type
monitor_distance            : float                         # (cm) eye-to-monitor distance
monitor_size=19             : float                         # (inches) size diagonal dimension
monitor_aspect=1.25         : float                         # physical aspect ratio of monitor
resolution_x=1280           : smallint                      # (pixels)
resolution_y=1024           : smallint                      # display resolution along y
psy_ts=CURRENT_TIMESTAMP    : timestamp                     # automatic
%}


classdef Session < dj.Relvar
end

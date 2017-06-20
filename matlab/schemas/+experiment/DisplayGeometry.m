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
display_timestamp = CURRENT_TIMESTAMP  : timestamp  # automatic
%}

classdef DisplayGeometry < dj.Relvar
    
    methods(Static)
        function migrate
            data = rmfield(fetch(vis.Session*experiment.Session & preprocess.Sync, ...
                'monitor_distance', 'monitor_size', 'monitor_aspect', ...
                'resolution_x', 'resolution_y', '60->fps', 'psy_ts->display_timestamp'), 'psy_id');
            inserti(experiment.DisplayGeometry, data)
        end
    end
end

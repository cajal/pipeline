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
    
    methods
        function sz = getMonitorSize(self, type)
            % gets X,Y monitor size
            
            if nargin<2
                type = 'cm';
            end

            [m_sz, m_as, m_ds] = fetch1(self,'monitor_size','monitor_aspect','monitor_distance');
            sz = nan(2,1);
            sz(2) = sqrt(m_sz^2/(m_as^2+1));
            sz(1) = m_as*sz(2);
            
            % get correct output
            switch type
                case 'inches'
                   
                case 'cm'
                    sz = sz*2.54;
                case 'degrees'
                    sz = atand(sz/(m_ds)/2*2.54)*2;
                otherwise 
                    disp('Unknown metric')
            end
            
        end
    end
end

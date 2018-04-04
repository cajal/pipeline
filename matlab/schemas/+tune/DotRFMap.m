%{
# RF map.
-> tune.DotRF
-> fuse.ScanSetUnit
---
response_map         : blob       # average response for each location
center_x             : float      # (fraction of x) center of receptive field on x, from Gaussian fits 
center_y             : float      # (fraction of x) center of receptive field on y, from Gaussian fits 
snr                  : float      # snr between rf and periphery 
p_value              : float      # bootstrap significance
gauss_fit            : mediumblob # gauss fitting parameters
%}

classdef DotRFMap < dj.Computed
    
    methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
    end
    
    methods
		function plot(obj)
            keys = fetch(obj);
            figure
            for key = keys'
                [gaussfit, map, p, x_loc,y_loc] = fetch1(tune.DotRFMap & key,...
                    'gauss_fit','response_map','p_value','center_x','center_y');
                plot(tune.DotRF & key, gaussfit, nanmax(map,[],3))
                title(sprintf('cell:%d animal:%d scan:%d p:%.3f\nx:%.2f y:%.2f',...
                    key.unit_id, key.animal_id,key.scan_idx, p,x_loc,y_loc))
                set(gcf,'name','Cell RF')
                pause
                clf
            end
        end
    end
end
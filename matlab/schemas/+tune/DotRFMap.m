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
            %	 self.insert(key)
        end
    end
    
    methods
        function plot(obj, varargin)
            params.fun = @(x) nanmax(x,[],3);
            params.hold = true;
            params.area = false;
            params = ne7.mat.getParams(params,varargin);
            
            keys = fetch(obj);
            for ikey =1:length(keys)
                if ~params.hold
                    figure
                end
                key = keys(ikey);
                [gaussfit, map, p, x_loc,y_loc] = fetch1(tune.DotRFMap & key,...
                    'gauss_fit','response_map','p_value','center_x','center_y');
                plot(tune.DotRF & key, gaussfit, params.fun(map),params)
            
                if ~params.hold && params.area
                    area = fetch1(anatomy.AreaMembership & key,'brain_area');
                        title(sprintf('area:%s cell:%d animal:%d session:%d scan:%d',...
                    area,key.unit_id, key.animal_id,key.session,key.scan_idx))
                	set(gcf,'name',sprintf('RF cell:%d animal:%d session:%d scan:%d',...
                       key.unit_id, key.animal_id,key.session,key.scan_idx))
                else
                        title(sprintf('cell:%d animal:%d session:%d scan:%d \n p:%.3f x:%.2f y:%.2f',...
                    key.unit_id, key.animal_id,key.session,key.scan_idx, p,x_loc,y_loc))
                    set(gcf,'name',sprintf('RF area:%s cell:%d animal:%d session:%d scan:%d',key.unit_id, key.animal_id,key.session,key.scan_idx))
                end
                if ikey~=length(keys) && params.hold
                    pause
                    clf
                end
            end
        end
        
        function [xdeg, ydeg, keys] = getDegrees(obj,flat_corrected)
            % [xdeg, ydeg, keys] = getDegrees(obj)
            %
            % getDegrees converts rf distances from the center to degress
            % from center
            % flat_corrected accounts for flat monitor with shortest
            % distance to the eye in the center of the monitor
            
            [x,y, keys] = fetchn( obj,'center_x','center_y');
            [aspect, distance, diag_size] = fetch1(experiment.DisplayGeometry & obj,'monitor_aspect','monitor_distance','monitor_size');
            x_size = sind(atand(aspect))*diag_size;
            if nargin>1 && flat_corrected
                x2deg = @(xx) atand(x_size*xx/distance);
                xdeg = x2deg(x);
                ydeg = x2deg(y);
            else
                max_deg = atand(x_size/2/distance)*2;
                xdeg = x*max_deg;
                ydeg = y*max_deg;
            end
        end
        
        
    end
    
    
end
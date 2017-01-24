%{
tuning.DotRFMap (computed) # receptive field from the dot stimulus
-> tuning.DotRF
-> preprocess.SpikesRateTrace
-----
response_map         : blob       # average response for each location
center_x             : float      # (fraction of x) center of receptive field on x, from Gaussian fits 
center_y             : float      # (fraction of x) center of receptive field on y, from Gaussian fits 
snr                  : float      # snr between rf and periphery 
p_value              : float      # bootstrap significance
gauss_fit            : mediumblob # gauss fitting parameters
%}

classdef DotRFMap < dj.Relvar
	
    methods
		function plot(obj)
            keys = fetch(obj);
            for key = keys'
                figure
                [gaussfit, map, p] = fetch1(tuning.DotRFMap & key,'gauss_fit','response_map','p_value');
                plot(tuning.DotRF & key, gaussfit, map)
                title(sprintf('cell:%d animal:%d scan:%d p:%.3f',...
                    key.trace_id, key.animal_id,key.scan_idx, p))
                set(gcf,'name','Cell RF')
            end
        end
    end
end
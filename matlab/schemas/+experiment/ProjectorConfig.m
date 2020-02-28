%{
# projector configuration
projector_config_id         : tinyint                       # projector config    
---
-> ProjectorColor.proj(channel_1="color_id")                # channel 1 means 1st color channel. Usually red
-> ProjectorColor.proj(channel_2="color_id")                # channel 2 means 2nd color channel. Usually green
-> ProjectorColor.proj(channel_3="color_id")                # channel 3 means 3rd color channel. Usually blue
refresh_rate                : float                         # refresh rate in Hz
%}

classdef ProjectorConfig < dj.Lookup
end

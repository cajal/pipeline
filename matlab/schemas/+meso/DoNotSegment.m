%{
# field/channels that should not be segmented (used for web interface only)
-> experiment.Scan
-> shared.Field
-> shared.Channel
%}


classdef DoNotSegment < dj.Manual
end
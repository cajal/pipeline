%{
# Photodiode voltage for different pixel intensities
-> experiment.Session
---
luminance               : mediumblob      # luminance in cd/m^2
px_value                : mediumblob      # pixel intensity values
pd_voltage              : mediumblob      # photodiode voltage at specified pixel values
%}

classdef DisplayLUT < dj.Lookup
end


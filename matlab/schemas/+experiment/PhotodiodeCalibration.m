%{
# Photodiode luminance calibration
-> experiment.Rig
trial                   : bigint          # repetitions
pixel_value             : mediumint       # control pixel value
---
luminance               : double          # luminance in cd/m^2
pd_voltage              : double          # photodiode voltage at specified luminance in V
ts                      : timestamp       # timestamp
%}

classdef PhotodiodeCalibration < dj.Manual
end


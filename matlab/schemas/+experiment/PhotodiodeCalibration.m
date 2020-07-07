%{
# Photodiode luminance calibration
-> experiment.Rig
trial                   : bigint          # repetitions
pixel_value             : mediumint       # control pixel value
blanking                : tinyint         # 0 if calib done with blanking off, 1 if done with blanking on
---
luminance               : double          # luminance in cd/m^2
pd_voltage              : double          # photodiode voltage at specified luminance in V
ts                      : timestamp       # timestamp
notes                   : varchar(255)    # comments about the calibration condition
monitor_type            : varchar(16)     # LCD or LC4500
illumination            : varchar(16)     # e.g. RGB
power_per_area          : double          # power density of the illumination
%}

classdef PhotodiodeCalibration < dj.Manual
end


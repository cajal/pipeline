%{
# Monitor luminance calibration
-> experiment.Scan
---
pixel_value             : mediumblob      # control pixel value (0-255)
luminance               : mediumblob      # luminance in cd/m^2
amplitude               : float           # lum = Amp*pixel_value^gamma + offset
gamma                   : float           # 
offset                  : float           # 
mse                     : float           # 
ts                      : timestamp       # timestamp
%}

classdef MonitorCalibration < dj.Manual
end


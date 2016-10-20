%{
experiment.LaserCalibrationPowerMeasurement (manual) # 
-> experiment.LaserCalibration
wavelength      : int                    # wavelength of the laser
attenuation      : tinyint                # power setting 
bidirectional   : tinyint                # 0 if off 1 if on
gdd             : int                    # GDD setting on the laser
---
power                       : float                         # power in mW
%}


classdef LaserCalibrationPowerMeasurement < dj.Relvar
end
%{
experiment.LaserCalibration (manual) # stores measured values from the laser power calibration
-> experiment.Rig
calibration_ts=CURRENT_TIMESTAMP: timestamp                #automatic
---
-> experiment.Lens
-> experiment.Person
-> experiment.Software           
%}


classdef LaserCalibration < dj.Relvar
end
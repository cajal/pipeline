%{
experiment.LaserCalibration (manual) # stores measured values from the laser power calibration
-> experiment.Rig
calibration_ts  : timestamp              # calibration timestamp -- automatic
---
-> experiment.Lens
-> experiment.Person
-> experiment.Software
%}


classdef LaserCalibration < dj.Relvar
end

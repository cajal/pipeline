%{
experiment.LaserCalibration (manual) # stores measured values from the laser power calibration
-> experiment.Rig
calibration_ts  : timestamp              # calibration timestamp -- automatic
---
-> experiment.Lens
-> experiment.Person
-> experiment.Software
pockel          : mediumint              # pockel setting
notes           : varchar(1024)          #
%}


classdef LaserCalibration < dj.Relvar
end

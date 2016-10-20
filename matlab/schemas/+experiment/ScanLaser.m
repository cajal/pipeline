%{
experiment.ScanLaser (manual) # Laser parameters
-> experiment.Scan
---
laser_wavelength            : float                         # (nm)
laser_power                 : float                         # (mW) to brain
laser_gdd                   : float                         # gdd setting
%}


classdef ScanLaser < dj.Relvar
end
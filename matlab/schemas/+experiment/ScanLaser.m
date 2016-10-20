%{
experiment.ScanLaser (manual) # Laser parameters for the scan
-> experiment.Scan
---
wavelength                  : float                         # (nm)
power                       : float                         # (mW) to brain
gdd                         : float                         # gdd setting
%}


classdef ScanLaser < dj.Relvar
end

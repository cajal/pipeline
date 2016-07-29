%{
experiment.FOV (lookup) # field-of-view sizes for all lenses and magnifications
-> experiment.Rig
-> experiment.Lens
mag             : decimal(5,2)           # ScanImage zoom factor
fov_ts          : datetime               # fov measurement date and time
---
height                      : decimal(5,1)                  # measured width of field of view along axis of pipette (medial/lateral on mouse)
width                       : decimal(5,1)                  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
%}


classdef FOV < dj.Relvar
end
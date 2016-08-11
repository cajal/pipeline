%{
rf.FOV (lookup) # field-of-view sizes for all lenses and magnifications
-> rf.Lens
mag             : decimal(5,2)           # ScanImage zoom factor
---
height                      : decimal(5,1)                  # measured width of field of view along axis of pipette (medial/lateral on mouse)
width                       : decimal(5,1)                  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
take=1                      : tinyint                       # 
fov_date                    : date                          # fov measurement date
INDEX(lens)
%}

classdef FOV < dj.Relvar
end

%{
rf.FOV (manual) # field of view sizes for lenses
setup :  tinyint  # two-photon setup number 
-> rf.Lens
take : tinyint  # measurement number 
mag  : decimal(5,2)  # magnification
-----
width  : decimal(5,1)  # (um) horizontal FOV size 
height : decimal(5,1)  # (um) vertical FOV size
time = CURRENT_TIMESTAMP :timestamp
%}

classdef FOV < dj.Relvar
end
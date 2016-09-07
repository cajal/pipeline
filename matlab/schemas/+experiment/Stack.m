%{
experiment.Stack (manual) # scanimage scan info
-> experiment.Session
stack_idx       : smallint               # number of TIFF stack file
---
bottom_z                    : int                           # z location at bottom of the stack
surf_z                      : int                           # z location of surface
laser_wavelength            : int                           # (nm)
laser_power                 : int                           # (mW) to brain
stack_notes                 : varchar(4095)                 # free-notes
filename                    : varchar(255)                  # file base name
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}
classdef Stack < dj.Relvar
end

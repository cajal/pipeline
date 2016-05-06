%{
rf.Stack (manual) # scanimage scan info
-> rf.Session
stack_idx        : smallint               # number of TIFF stack file
---
-> rf.Site
bottom_z                    : int                           # z location at bottom of the stack
surf_z                      : int                           # z location of surface
laser_wavelength            : int                           # (nm)
laser_power                 : int                           # (mW) to brain
stack_notes                 : varchar(4095)                 # free-notes
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}
classdef Stack < dj.Relvar
end
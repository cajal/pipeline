%{
# scanimage scan info
-> experiment.Session
stack_idx                   : smallint                      # number of TIFF stack file
---
-> experiment.Lens
-> experiment.BrainArea
-> experiment.Software
laser_wavelength            : int                           # (nm)
laser_power                 : int                           # (mW) to brain
filename                    : varchar(255)                  # file base name
bottom_z                    : int                           # z location at bottom of the stack
surf_z                      : int                           # z location of surface
stack_notes                 : varchar(4095)                 # free-notes
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}


classdef OldStack < dj.Manual
end
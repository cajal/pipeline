%{
experiment.Scan (manual) # scanimage scan info
-> experiment.Session
scan_idx        : smallint               # number of TIFF stack file
---
-> experiment.Lens
-> experiment.BrainArea
-> experiment.Aim
-> experiment.Software
laser_wavelength            : float                         # (nm)
laser_power                 : float                         # (mW) to brain
filename                    : varchar(255)                  # file base name
behavior_filename           : varchar(255)                  # pupil movies, whisking, locomotion, etc.
depth=0                     : int                           # manual depth measurement
scan_notes                  : varchar(4095)                 # free-notes
site_number=0               : tinyint                       # site number
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}


classdef Scan < dj.Relvar
end

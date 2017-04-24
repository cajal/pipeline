%{
# scanimage scan info
-> experiment.Session
scan_idx                    : smallint                      # number of TIFF stack file
---
-> experiment.Lens
-> experiment.BrainArea
-> experiment.Aim
filename                    : varchar(255)                  # file base name
depth=0                     : int                           # manual depth measurement
scan_notes                  : varchar(4095)                 # free-notes
site_number=0               : tinyint                       # site number
-> experiment.Software
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}


classdef Scan < dj.Manual
end
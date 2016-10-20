%{
experiment.Session (manual) # imaging session
-> mice.Mice
session         : smallint               # session index for the mouse
---
-> experiment.Rig
-> experiment.Person
-> experiment.Anesthesia
session_date                : date                          # date
scan_path                   : varchar(255)                  # file path for TIFF stacks
behavior_path               : varchar(255)                  # pupil movies, whisking, locomotion, etc.
craniotomy_notes            : varchar(4095)                 # free-text notes
session_notes               : varchar(4095)                 # free-text notes
archive                     : varchar(255)                  # archive disk label
session_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}


classdef Session < dj.Relvar
end

%{
rf.Session (manual) # session
-> common.Animal
session         : smallint               # session index
---
-> rf.Fluorophore
-> rf.Lens
session_date                : date                          # date
scan_path                   : varchar(255)                  # file path for TIFF stacks
hd5_path                    : varchar(255)                  # file path for HD5 files
file_base                   : varchar(255)                  # file base name
anesthesia="awake"          : enum('isoflurane','fentanyl','awake') # per protocol
craniotomy_notes            : varchar(4095)                 # free-text notes
session_notes               : varchar(4095)                 # free-text notes
session_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
INDEX(lens)
%}

classdef Session < dj.Relvar
end

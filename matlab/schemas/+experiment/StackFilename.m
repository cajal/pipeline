%{
# filenames that compose one stack (used in resonant and 3p scans)
-> experiment.Stack
filename_idx                : tinyint                       # id of the file
---
filename                    : varchar(255)                  # file base name
surf_depth=0                : float                         # ScanImage's z at cortex surface
%}


classdef StackFilename < dj.Manual
end
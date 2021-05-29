%{
# scan stim type, visual or auditory
-> experiment.Scan
-> experiment.StimTypes
---
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}


classdef ScanStimType < dj.Manual
end
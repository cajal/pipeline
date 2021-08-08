%{
# scan stim type, visual or auditory
-> experiment.Scan
-> experiment.StimTypes
---
speaker_id                  : int                           # id of the speaker
location                    : tinyint                       # 1 = Left, 2 = Right
calib_trial                 : int                           # calibration trial used in this scan
piezostimulator_id          : int                           # id of piezo stimulator
scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
%}


classdef ScanStimType < dj.Manual
end
%{
vis.ScanConditions (manual) # Link between conditions and scan_idx
-> vis.Condition
-> experiment.Scan
---
%}

classdef ScanConditions < dj.Relvar
end
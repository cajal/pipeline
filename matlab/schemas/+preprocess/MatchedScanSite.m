%{
# an entry in this table indicates that two scans have been recorded at the same location
-> preprocess.ExtractRaw
other_scan_idx              : smallint                      # second dependent scan idx
---
match_threshold=0.7         : float                         # minimal cosine between masks for match
%}


classdef MatchedScanSite < dj.Manual
end
%{
vis.MonetLookup (lookup) # cached noise maps to save computation time
moving_noise_version: smallint           # algorithm version; increment when code changes
moving_noise_paramhash: char(10)         # hash of the lookup parameters
---
params                      : blob                          # cell array of params
cached_movie                : longblob                      # [y,x,frames]
moving_noise_lookup_ts=CURRENT_TIMESTAMP: timestamp         # automatic
%}


classdef MonetLookup < dj.Relvar
end
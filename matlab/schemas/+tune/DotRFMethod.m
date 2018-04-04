%{
tune.DotRFMethod (lookup) # options for dot maping.
rf_method                : tinyint                # method number
---
onset_delay              : float                         # response onset delay in msec
response_duration        : float                         # response window in msec
rf_filter                : float                         # gaussian filter (in degrees)
rf_sd                    : tinyint                       # rf standard deviation for snr computation
shuffle                  : int                           # shuffling number for bootstrap
snr_method               : enum('var','zscore','dprime') # method for measuring snr
notes                    : varchar(255)                  # details for each setting
%}

classdef DotRFMethod < dj.Lookup
end
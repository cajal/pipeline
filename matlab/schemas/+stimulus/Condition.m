%{
# stimulus condition
condition_hash              : char(20)                      # 120-bit hash (The first 20 chars of MD5 in base64)
---
stimulus_type               : varchar(255)                  # class name of the special stimulus condition table class
stimulus_version            : varchar(255)                  # specified by the property `version` of the stimulus class
%}

classdef Condition < dj.Manual
end

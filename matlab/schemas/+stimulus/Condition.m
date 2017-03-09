%{
stimulus.Condition (manual) # stimulus condition
condition_hash              : char(20)               # 120-bit hash (The first 20 chars of MD5 in base64)
---
special_name        : varchar(255)                  # class name of the special stimulus condition table class
special_variation   : varchar(255)                  # the variation property of the same class
%}

classdef Condition < dj.Relvar
end
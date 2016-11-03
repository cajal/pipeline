%{
experiment.Aim (lookup) # Declared purpose of the scan
aim             : varchar(36)            # short name for the purpose of the scan
---
aim_description             : varchar(255)                  # 
%}


classdef Aim < dj.Relvar
end

%{
experiment.Software (lookup) # 
software             : varchar(20)                # name of the software
version             : char(10)                # version
---
%}


classdef Software < dj.Relvar
end
%{
experiment.Anesthesia (lookup) # Anesthesia type
anesthesia     : char(20)               # anesthesia short name
---
anesthesia_description             : varchar(2048)                 # 
%}


classdef Anesthesia < dj.Relvar
end
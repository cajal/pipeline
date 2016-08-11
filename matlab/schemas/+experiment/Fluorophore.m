%{
experiment.Fluorophore (lookup) # calcium-sensitive indicators
fluorophore     : char(10)               # fluorophore short name
---
dye_description             : varchar(2048)                 # 
%}


classdef Fluorophore < dj.Relvar
end
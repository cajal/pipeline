%{
experiment.Aim (lookup) # Recording aim
aim     : varchar(255)               # aim short name
---
aim_description             : varchar(255)                 # 
%}


classdef Aim < dj.Relvar
end
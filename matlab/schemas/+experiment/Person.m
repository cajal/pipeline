%{
experiment.Person (lookup) # Users
username     : char(12)               # lab member
---
full_name             : varchar(2048)                 # 
%}


classdef Person < dj.Relvar
end
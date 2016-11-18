%{
lab.Paths (lookup) # Path translation
global       : varchar(255)               # global path name
---
linux        : varchar(255)               # linux path name
windows      : varchar(255)               # windows path name
mac          : varchar(255)               # mac path name
location     : varchar(255)               # computer path
%}


classdef Paths < dj.Relvar
end
%{
experiment.VisProtocol (lookup) # visual protocol used
-> experiment.Person
vis_protocol        : varchar(255)               # brief name
---
vis_filename                : varchar(255)                  # file base name
discription                  : varchar(255)                 # 
%}


classdef VisProtocol < dj.Relvar
end

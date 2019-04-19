%{
# 
vis_protocol                : varchar(255)                  # 
-> experiment.Person
---
stim_version=1              : tinyint                       # 
vis_filename=null           : varchar(255)                  # 
discription=null            : varchar(255)                  # 
timestamp=null              : timestamp                     # 
%}


classdef VisProtocol < dj.Lookup
end
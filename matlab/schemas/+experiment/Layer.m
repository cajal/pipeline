%{
# 
layer                       : char(12)                      # short name for cortical area
---
layer_description           : varchar(255)                  # 
z_start=null                : float                         # starting depth
z_end=null                  : float                         # deepest point
%}


classdef Layer < dj.Lookup
end
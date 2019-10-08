%{
# population statistics parameters
opt_idx                     : smallint                      # 
---
rsz=10                      : smallint                      # movie resizing factor
algorithm=null              : enum('opticalFlowFarneback','opticalFlowLK') # 
params=null                 : varchar(255)                  # 
process=null                : enum('yes','no')              # 
%}


classdef OpticFlowOpt < dj.Lookup
end
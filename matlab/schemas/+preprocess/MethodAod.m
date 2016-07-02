%{
preprocess.MethodAod (lookup) # 
-> preprocess.Method
---
description                 : varchar(60)                   # 
high_pass_stop=null         : float                         # (Hz)
low_pass_stop=null          : float                         # (Hz)
subtracted_princ_comps      : tinyint                       # number of principal components to subtract
%}


classdef MethodAod < dj.Relvar
end
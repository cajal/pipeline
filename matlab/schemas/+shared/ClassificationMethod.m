%{
# methods to classify extracted masks
classification_method       : tinyint                       # 
---
name                        : varchar(16)                   # 
details                     : varchar(255)                  # 
language                    : enum('matlab','python')       # implementation language
%}


classdef ClassificationMethod < dj.Lookup
end
%{
# methods for mask extraction for multi-photon scans
segmentation_method         : tinyint                       # 
---
name                        : varchar(16)                   # 
details                     : varchar(255)                  # 
language                    : enum('matlab','python')       # implementation language
%}


classdef SegmentationMethod < dj.Lookup
end
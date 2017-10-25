%{
# classification of segmentation masks
mask_type                   : varchar(32)                   # cell type
%}


classdef MaskType < dj.Lookup
end
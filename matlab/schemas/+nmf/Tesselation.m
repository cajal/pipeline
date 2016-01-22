%{
nmf.Tesselation (manual) # my newest table
-> nmf.Segment 
rstart                : int # row start index of the mask
rend                  : int # row end index of the mask
cstart                : int # col start index of the mask
cend                  : int # col end index of the mask
-----
%}

classdef Tesselation < dj.Relvar
end
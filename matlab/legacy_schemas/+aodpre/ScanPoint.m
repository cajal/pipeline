%{
aodpre.ScanPoint (imported) # my newest table
-> aodpre.Set
point_id : smallint #  scan point numer
-----
x :  float   # (um) 
y :  float   # (um)
z :  float   # (um) 
%}

classdef ScanPoint < dj.Relvar
end
%{
vis.Matisse2Cache (manual) # cached movies for the the Matisse2 stimulus
cond_hash : char(20)               # 120-bit hash (The first 20 chars of MD5 in base64)
-----
movie :longblob    #  stored movie
%}

classdef Matisse2Cache < dj.Relvar
end
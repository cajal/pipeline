%{
preprocess.CirclesLookup (lookup) # my newest table
# add primary key here
radius : smallint # radius of the circles (pixels)
width : smallint # width of the rectangle over which circles lookup is created (pixels)
height : smallint # height of rectangle
-----
# add additional attributes
basepath : varchar(1024) # folder where lookup file is stored
filename : varchar(32) # lookup table is stored here as a binary file, it
                       # is an array of bytes
%}

classdef CirclesLookup < dj.Relvar
end
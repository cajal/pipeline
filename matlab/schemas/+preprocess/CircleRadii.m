%{
preprocess.CircleRadii (lookup) # my newest table
# add primary key here
radius  :   smallint    # radius of the cluster
-----
# add additional attributes
%}

classdef CircleRadii < dj.Relvar
end
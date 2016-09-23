%{
preprocess.ClusterPeaks (lookup) # my newest table
# add primary key here
peakorder=1   :   tinyint       # 1=cluster around absolute maximum,
                                # 2=cluster around the next peak after removing the previous cluster elements
-----
# add additional attributes
%}

classdef ClusterPeaks < dj.Relvar
end
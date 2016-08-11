%{
experiment.PMTFilterSet (lookup) # microscope filter sets: dichroic and PMT Filters
pmt_filter_set  : varchar(16)            # short name of microscope filter set
---
primary_dichroic            : varchar(255)                  # passes the laser  (excitation/emission separation)
secondary_dichroic          : varchar(255)                  # splits emission spectrum
filter_set_description      : varchar(4096)                 # A detailed description of the filter set
%}


classdef PMTFilterSet < dj.Relvar
end
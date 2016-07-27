%{
experiment.PMTFilterSetChannel (lookup) # PMT description including dichroic and filter
-> experiment.PMTFilterSet
pmt_channel     : tinyint                # pmt_channel
---
color                       : enum('green','red','blue')    # 
pmt_serial_number           : varchar(40)                   # 
spectrum_center             : smallint unsigned             # (nm) overall pass spectrum of all upstream filters
spectrum_bandwidth          : smallint unsigned             # (nm) overall pass spectrum of all upstream filters
pmt_filter_details          : varchar(255)                  # more details, spectrum, pre-amp gain, pre-amp ADC filter
%}


classdef PMTFilterSetChannel < dj.Relvar
end
%{
experiment.FluorophoreEmissionSpectrum (lookup) # calcium-sensitive indicators Spectrum
-> experiment.Fluorophore
loaded     : tinyint               # 
---
wavelength             : longblob                 # 
fluorescence           : longblob                 #
%}


classdef FluorophoreEmissionSpectrum < dj.Relvar
end
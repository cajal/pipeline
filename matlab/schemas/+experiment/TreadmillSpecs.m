%{
experiment.TreadmillSpecs (lookup) # methods for extraction from raw data for either AOD or Galvo data
-> experiment.Rig
treadmill_start_date          : date              # first day in use on this rig
---
diameter                        : float             # treadmill diameter where mouse sits in cm
counts_per_revolution = 8000    : int               # number of encoder counts per treadmill revolution
treadmill_notes                 : varchar(255)      # additional info about treadmill
%}


classdef TreadmillSpecs < dj.Relvar
end
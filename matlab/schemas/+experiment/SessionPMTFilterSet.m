%{
experiment.SessionPMTFilterSet (manual) # Fluorophores expressed in prep for the imaging session
-> experiment.Session
---
-> experiment.PMTFilterSet
%}


classdef SessionPMTFilterSet < dj.Relvar
end
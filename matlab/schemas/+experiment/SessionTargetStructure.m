%{
experiment.SessionTargetStructure (manual) # specifies which neuronal structure was imaged
-> experiment.Session
-> experiment.Fluorophore
-> experiment.Compartment
---
%}


classdef SessionTargetStructure < dj.Relvar
end
%{
experiment.SessionFluorophore (manual) # Fluorophores expressed in prep for the imaging session
-> experiment.Session
-> experiment.Fluorophore
---
notes                       : varchar(255)                  # additional information about fluorophore in this scan
%}


classdef SessionFluorophore < dj.Relvar
end
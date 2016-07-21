%{
vis2p.Mice (manual) # 
mouse_id        : smallint unsigned      # m) Unique identification number
---
mouse_gender="male"         : enum('unknown','female','male') # 
mouse_strain="C57BL6/J"     : enum('NesCre-ReYFP','Viat-Ai9','SST-Ai9','Viaat','Ai9','PV','PV-Ai9','PV-AAVArch','SST-AAVArch','SST-ChR2','Nestin-Ai9','C57BL6/J') # m) strain of mouse
mouse_dob                   : date                          # m) mouse date of birth
mouse_dod=null              : date                          # m) Euthanasia date
mouse_notes                 : varchar(1023)                 # m) few words for the mouse
headpost_date=null          : date                          # m) date that the mouse was headposted
%}


classdef Mice < dj.Relvar
     methods
        function self = Mice(varargin)
           self.restrict(varargin{:})
        end
    end
end
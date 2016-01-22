%{
mice.PcrPrograms (manual) # PCR programs for detecting transgenes

pcr_name                  : enum('Cre','tdTomato','YFP','GCaMP3','TVA','KOPRCS+','KOPRCS-')   # transgene tested by pcr
---
program=""        : varchar(1000)           # pcr program details

program_notes=""    : varchar(4096)             # other comments 
program_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef PcrPrograms < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.PcrPrograms')
	end

	methods
		function self = PcrPrograms(varargin)
			self.restrict(varargin)
		end
	end
end
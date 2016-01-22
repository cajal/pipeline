%{
mice.PcrPrimers (manual) # primer sets used in each PCR reaction

-> mice.PcrPrograms
-> mice.Primers 
---

pcr_notes=""    : varchar(4096)             # other comments 
pcr_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef PcrPrimers < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.PcrPrimers')
	end

	methods
		function self = PcrPrimers(varargin)
			self.restrict(varargin)
		end
	end
end

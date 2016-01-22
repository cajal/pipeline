%{
mice.Genotyping (manual) # transgenes tested to genotype each line

-> mice.Lines
-> mice.PcrPrograms 
---

genotyping_notes=""    : varchar(4096)             # other comments 
genotyping_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Genotyping < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Genotyping')
	end

	methods
		function self = Genotyping(varargin)
			self.restrict(varargin)
		end
	end
end

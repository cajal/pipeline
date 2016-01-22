%{
mice.Primers (manual) # info about each primer set

primer_set          : varchar(20)               # primer set
---
f_primer            : varchar(100)               # forward primer
r_primer            : varchar(100)               # reverse primer
band                : int                       # band size
allele              : varchar(20)               # allele tested


primer_notes=""    : varchar(4096)             # other comments 
primer_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Primers < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Primers')
	end

	methods
		function self = Primers(varargin)
			self.restrict(varargin)
		end
	end
end

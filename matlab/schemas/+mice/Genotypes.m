%{
mice.Genotypes (manual) # info about each mouse's genotype$
-> mice.Mice
-> mice.Lines
---
genotype="unknown"          : enum('homozygous','heterozygous','hemizygous','positive','negative','wild type','unknown') # animal's genotype
genotype_notes              : varchar(4096)                 # other comments
genotype_ts=CURRENT_TIMESTAMP: timestamp                    # automatic
%}



classdef Genotypes < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Genotypes')
	end

	methods
		function self = Genotypes(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end

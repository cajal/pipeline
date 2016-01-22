%{
mice.Founders (manual) # Additional info about founder mice$
-> mice.Mice
-> mice.Lines
---
source                      : varchar(100)                  # source of mouse (lab, company)
doa=null                    : date                          # date of arrival
founder_notes               : varchar(4096)                 # other comments
founder_ts=CURRENT_TIMESTAMP: timestamp                     # automatic
%}



classdef Founders < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Founders')
	end

	methods
		function self = Founders(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end

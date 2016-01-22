%{
mice.Death (manual) # info about each mouse's death

-> mice.Mice
---
dod=null            : date                      # date of death

death_notes=""    : varchar(4096)             # other comments 
death_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Death < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Death')
	end

	methods
		function self = Death(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end

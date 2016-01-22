%{
mice.Parents (manual) # parent-child relationships between mice

-> mice.Mice         
parent_id          : varchar(20)           # id number of parent
---

relation_notes=""    : varchar(4096)             # other comments 
relation_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Parents < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Parents')
	end

	methods
		function self = Parents(varargin)
			self.restrict(varargin)
        end
        function makeTuples(self,key)
            self.insert(key)
        end
	end
end

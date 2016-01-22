%{
mice.Lanes (manual) # lanes on each gel

-> mice.Gels
-> mice.Mice
-> mice.PcrPrograms
lane_id             : tinyint                # lane number
---

lane_notes=""    : varchar(4096)             # other comments 
lane_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Lanes < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Lanes')
	end

	methods
		function self = Lanes(varargin)
			self.restrict(varargin)
		end
	end
end

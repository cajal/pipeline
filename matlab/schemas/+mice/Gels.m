%{
mice.Gels (manual) # electrophoresis gels used for genotyping

gel_id              : varchar(100)           # gel id
---
gel_date=null       : date                  # date gel was imaged
image=null         : blob          # gel image

gel_notes=""    : varchar(4096)             # other comments 
gel_ts=CURRENT_TIMESTAMP : timestamp        # automatic
%}



classdef Gels < dj.Relvar

	properties(Constant)
		table = dj.Table('mice.Gels')
	end

	methods
		function self = Gels(varargin)
			self.restrict(varargin)
		end
	end
end

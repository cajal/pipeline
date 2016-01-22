%{
common.WholeCell (manual) # whole cell path
-> common.WholeCellSession
patch_num       : tinyint    # the number of the patched cell
rec_num         : tinyint    # whole cell recording number for this animal
---
wc_notes=""                 : varchar(511)                  # free-hand notes
special_filename=""      : varchar(255)                  # only necessary for deviations from naming convention
wc_ts=CURRENT_TIMESTAMP     : timestamp                     # automatic
%}


classdef WholeCell < dj.Relvar

	properties(Constant)
		table = dj.Table('common.WholeCell')
	end

	methods
		function self = WholeCell(varargin)
			self.restrict(varargin)
		end
	end
end

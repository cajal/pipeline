%{
common.WholeCellSession (manual) # in-vivo whole cell patch recording session

-> common.Animal
---
wc_path="/xiaolong/"        : varchar(511)                  # the folder with .ERD or .WCP files for processing
wc_session_date             : date                          # recording date
wc_session_notes=""         : varchar(511)                  # arbitrary notes
%}


classdef WholeCellSession < dj.Relvar

	properties(Constant)
		table = dj.Table('common.WholeCellSession')
	end

	methods
		function self = WholeCellSession(varargin)
			self.restrict(varargin)
		end
	end
end

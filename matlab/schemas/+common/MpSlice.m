%{
common.MpSlice (manual) # brain slice for in-vitro patching 
-> common.Animal
mp_slice                  : smallint       # brain slice number for this animal
-----
slice_date                : date            #
brain_area                : varchar(80)     # e.g. left barrel cortex. free text for now.
thickness = 300           : float           # (um) slice thickness
experimenter              : varchar(80)     # who did the slicing
slice_notes = ""          : varchar(4095)   # any other notes
slice_time = CURRENT_TIMESTAMP : timestamp  # automatic but editable
%}

classdef MpSlice < dj.Relvar

	properties(Constant)
		table = dj.Table('common.MpSlice')
	end

	methods
		function self = MpSlice(varargin)
			self.restrict(varargin)
		end
	end
end

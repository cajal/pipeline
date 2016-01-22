%{
common.BrainSliceImage (manual) # Zeiss images of brain images

-> common.Animal
slice_id        : tinyint               # sequential slice numbers
---
slice_filepath              : varchar(511)                  # full path (Mac)
first_slice=0               : tinyint                       # if 0, there must be a slice with slice_num-1
%}

classdef BrainSliceImage < dj.Relvar

	properties(Constant)
		table = dj.Table('common.BrainSliceImage')
	end
end

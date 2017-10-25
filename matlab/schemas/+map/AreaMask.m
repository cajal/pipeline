%{
# clips from movies
animal_id                   : int                           # id number
session                     : smallint                      # session index for the mouse
area                        : enum('A','POR','P','PM','AM','RL','AL','LI','LM','V1') # area name
scan_idx                    : smallint                      # number of TIFF stack file
slice                       : tinyint                       # slice in scan
---
mask=null                   : mediumblob                    # 
align_params=null           : mediumblob                    # 
%}


classdef AreaMask < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end
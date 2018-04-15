%{
# 
-> tune.STAQual
---
x=null                      : float                         # x coordinate (mean of Gaussian fit)
y=null                      : float                         # y coordinate (mean of Gaussian fit)
radius=null                 : float                         # 2*sigma border of major axis of Gaussian fit
%}


classdef STAExtent < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end
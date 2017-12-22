%{
# 
-> tune.MonetOracle
-> `pipeline_fuse`.`__activity__trace`
-> stimulus.Monet
---
trials                      : int                           # number of trials used
pearson                     : float                         # per unit oracle pearson correlation over all movies
%}


classdef MonetOraclePearson < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end
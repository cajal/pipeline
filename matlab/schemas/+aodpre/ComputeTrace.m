%{
aodpre.ComputeTrace (computed) #  compute calcium traces from timeseries
-> aodpre.Set
-----
%}

classdef ComputeTrace < dj.Relvar & dj.AutoPopulate

	properties
		popRel  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			self.insert(key)
		end
	end

end
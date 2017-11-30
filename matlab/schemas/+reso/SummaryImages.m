%{
# summary images for each slice and channel after corrections
-> reso.MotionCorrection
-> shared.Channel
%}


classdef SummaryImages < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end
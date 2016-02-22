%{
trk.Quality (computed) # quality assessment of tracking using Jake's tracked frames as ground truth
-> rf.Eye
---
pos_err                     : float                         # mean Euclidean distance between pupil positions
r_corr                      : float                         # correlation of radii
excess_frames               : int                           # number of frames detected by tracking but not in Jake's data
missed_frames               : int                           # number of frames detected by Jake but no by tracking
total_frames                : int                           # total number of frames in the video
%}


classdef Quality < dj.Relvar & dj.AutoPopulate

	properties
		popRel = rf.Eye  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end
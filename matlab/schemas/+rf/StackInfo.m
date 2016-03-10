%{
stk.StackInfo (imported) # header information
-> stk.Stack
---
nchannels                   : tinyint                       # number of recorded channels
nslices                     : int                           # number of slices (hStackManager_numSlices)
frames_per_slice            : int                           # number of frames per slice (hStackManager_framesPerSlice)
px_width                    : smallint                      # pixels per line
px_height                   : smallint                      # lines per frame
zoom                        : decimal(4,1)                  # zoom factor
um_width                    : float                         # width in microns
um_height                   : float                         # height in microns
slice_pitch                 : float                         # (um) distance between slices (hStackManager_stackZStepSize)
%}

classdef StackInfo < dj.Relvar & dj.AutoPopulate

	properties
		popRel  % !!! update the populate relation
	end

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			self.insert(key)
		end
	end

end
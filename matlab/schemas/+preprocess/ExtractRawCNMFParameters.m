%{
# Arguments used to demix and deconvolve the scan with CNMF
-> preprocess.ExtractRaw
---
num_components              : smallint                      # estimated number of components
ar_order                    : tinyint                       # order of the autoregressive process for impulse function response
merge_threshold             : float                         # overlapping masks are merged if temporal correlation greater than this
num_processes=null          : smallint                      # number of processes to run in parallel, null=all available
num_pixels_per_process      : int                           # number of pixels processed at a time
block_size                  : int                           # number of pixels per each dot product
init_method                 : enum('greedy_roi','sparse_nmf','local_nmf') # type of initialization used
soma_radius_in_pixels=null  : blob                          # estimated radius for a soma in the scan
snmf_alpha=null             : float                         # regularization parameter for SNMF
num_background_components   : smallint                      # estimated number of background components
init_on_patches             : tinyint                       # whether to run initialization on small patches
patch_downsampling_factor=null: tinyint                     # how to downsample the scan
percentage_of_patch_overlap=null: float                     # overlap between adjacent patches
%}


classdef ExtractRawCNMFParameters < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 %self.insert(key)
		end
	end

end
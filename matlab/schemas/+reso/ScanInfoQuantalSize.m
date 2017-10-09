%{
# quantal size in images
-> reso.ScanInfo
-> `pipeline_shared`.`#slice`
-> `pipeline_shared`.`#channel`
---
min_intensity               : int                           # min value in movie
max_intensity               : int                           # max value in movie
intensities                 : longblob                      # intensities for fitting variances
variances                   : longblob                      # variances for each intensity
quantal_size                : float                         # variance slope, corresponds to quantal size
zero_level                  : int                           # level corresponding to zero (computed from variance dependence)
quantal_frame               : longblob                      # average frame expressed in quanta
median_quantum_rate         : float                         # median value in frame
percentile95_quantum_rate   : float                         # 95th percentile in frame
%}


classdef ScanInfoQuantalSize < dj.Imported

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
			 self.insert(key)
		end
	end

end
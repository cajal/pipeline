%{
# Calcium activity for the whole scan (multiple scan fields)
-> `pipeline_experiment`.`scan`
-> `pipeline_shared`.`#pipeline_version`
-> `pipeline_shared`.`#segmentation_method`
-> `pipeline_shared`.`#spike_method`
---
-> fuse.Pipe
%}


classdef ScanDone < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
    end
    
    methods
        function [average_images,keys] = getSummaryImageAverage(self)
           average_images = [];
           keys = [];
           if exists(reso.SummaryImagesAverage & self)
               [average_images,keys] = fetchn(reso.SummaryImagesAverage & self,'average_image');
           elseif meso.SummaryImagesAverage & self
               [average_images,keys] = fetchn(meso.SummaryImagesAverage & self,'average_image');
           end
            
        end
        
    end

end
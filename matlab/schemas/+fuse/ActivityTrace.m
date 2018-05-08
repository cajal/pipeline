%{
# Trace corresponding to <module>.Activity.Trace
-> fuse.Activity
unit_id                     : int                           # unique per scan & segmentation method
%}


classdef ActivityTrace < dj.Computed
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            % 		%!!! compute missing fields for key here
            % 			 self.insert(key)
        end
    end
    
    methods
        
        function [Spikes, frame_times, keys] = getAdjustedSpikes(obj,type)
            
            if nargin<2; type = 'soma';end
            
            % get frame times
            frame_times = fetch1(stimulus.Sync & obj,'frame_times');
            
            % get traces
            if strcmp(fetch1(experiment.Session & obj,'rig'),'2P4')
                [traces, ms_delay, keys] = fetchn(meso.ActivityTrace * (proj(meso.ScanSetUnitInfo,'ms_delay') & ...
                    proj(meso.ScanSetUnit & (meso.MaskClassificationType & struct('type',type)) & obj)) & obj,'trace','ms_delay');
                nfields = fetch1(proj(meso.ScanInfo & obj,'nfields/nrois->depths'),'depths');
            else
                [traces, ms_delay, keys] = fetchn(reso.ActivityTrace * (proj(reso.ScanSetUnitInfo,'ms_delay') & ...
                    proj(reso.ScanSetUnit & (reso.MaskClassificationType & struct('type',type)) & obj)) & obj,'trace','ms_delay');
                nfields = fetch1(reso.ScanInfo & obj,'nfields');
            end
            
            % find minimum trace length
            min_trace_length = min(cellfun(@(x) size(x,1),traces));
            traces = cell2mat(cellfun(@(x) single(x(1:min_trace_length)),traces,'uni',0)');
            
            % adjust to correct length
            frame_times = frame_times(1:nfields:end);
            frame_times = frame_times(1:min_trace_length)';
            
            % Interpolate all spikes to the first timepoint
            Spikes = nan(size(traces));
            for itrace = 1:size(traces,2)
                Spikes(:,itrace) = interp1(frame_times + ms_delay(itrace)/1000,traces(:,itrace),frame_times,'linear',nan);
            end
        end
    end
end
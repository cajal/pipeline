%{
# Compute frequency of abnormal activity
-> experiment.Scan
---
dff: longblob          # quality intesity for all fields, length(trace) x nFields
mean_dff: longblob     # mean quality intensity trace (df/f) across all fields
prominences: longblob  # peak amplitudes of global epileptiform events
widths: longblob       # half widths of global epileptiform events
peak_locs: longblob    # locations of peaks
peak_locs_abn: longblob # locations of abnormal peaks
time: longblob         # time frame of a particular trace, in secs
record_length: float   # total recording time, in secs
event_freq : float     # mean frequency of epilepticform events in the scan, nevents per second
%}


classdef AbnormalEventFrequency < dj.Computed
    
    properties
        keySource = experiment.Scan & (pro(reso.QualityMeanIntensity) | pro(meso.QualityMeanIntensity));
    end

	methods(Access=protected)

		function makeTuples(self, key)
            
            if exists(reso.QualityMeanIntensity & key)
                pipe = 'reso';
            elseif exists(meso.QualityMeanIntensity & key)
                pipe = 'meso';
            else
                return
            end
                
            qual_mean_intensity = eval([pipe '.QualityMeanIntensity']);
            scan_info = eval([pipe '.ScanInfo']);
    
            fps = fetch1(scan_info & key, 'fps');
            traces = squeeze(extractValues(fetchn(qual_mean_intensity & key & 'channel=1','intensities')));
            
            key.time = (1:size(traces,1))/fps;
            key.dff = bsxfun(@rdivide, bsxfun(@minus, traces, mean(traces)), mean(traces));
            key.mean_dff = mean(key.dff,2);
            [~,key.peak_locs,key.widths, key.prominences] = findpeaks(key.mean_dff,key.time);
            key.peak_locs_abn = key.peak_locs(key.prominences' > 0.2 & key.widths < 0.4);
            key.event_freq = length(key.peak_locs_abn)/range(key.time);
            key.record_length = range(key.time);
            
            self.insert(key);
		end
	end

end
%{
aodpre.Sync (imported) # stimulus synchronization
-> aodpre.Scan
-----
-> psy.Session
first_trial                 : int                           # first trial index from psy.Trial overlapping recording
last_trial                  : int                           # last trial index from psy.Trial overlapping recording
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
%}

classdef Sync < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = aodpre.Scan
	end

	methods(Access=protected)

		function makeTuples(self, key)
            file = fetch1(aodpre.Scan & key, 'hdf5_file');
            temporal = aodReader(getLocalPath(file), 'Temporal'); 
            display_fps = 60;
            sync_info = stims.analysis.sync(struct('animal_id', key.mouse_id), temporal(:,1), temporal.Fs, display_fps);
			self.insert(dj.struct.join(key, sync_info))
		end
	end

end
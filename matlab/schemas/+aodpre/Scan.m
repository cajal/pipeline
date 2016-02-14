%{
aodpre.Scan (imported) # synchronized AOD scans
-> vis2p.Scans
---
-> psy.Session
hdf5_file                   : varchar(255)                  # raw data file
sampling_rate               : double                        # sampling rate of the signal
first_trial                 : int                           # first trial index from psy.Trial overlapping recording
last_trial                  : int                           # last trial index from psy.Trial overlapping recording
signal_start_time           : double                        # (s) signal start time on stimulus clock
signal_duration             : double                        # (s) signal duration on stimulus time
%}

classdef Scan < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = (vis2p.Scans & 'aim="Cajal"') - aodpre.Ignore 
    end

	methods(Access=protected)

		function makeTuples(self, key)
            [~,name] = fetch1(vis2p.Experiments*vis2p.Scans & key, 'directory','file_name' );
            dirs = dir(fullfile(getLocalPath('M:\Mouse\'), [name(1:10) '*']));
            names = vertcat(dirs.name);
            % find the session that started immediatelly before the recording
            timediff = str2double(names(:,[12 13 15 16 18 19]))- str2double( name([12 13 15 16 18 19]));
            timeind = find(timediff<0);
            [~,i] = max(timediff(timeind));
            itime = timeind(i);
            tuple = key;
            tuple.hdf5_file = fullfile(getLocalPath('M:\Mouse\'), ...
                dirs(itime).name, 'AODAcq', name);
            temporal = aodReader(tuple.hdf5_file, 'Temporal'); 
            fps = 60;
            sync_info = stims.analysis.sync(...
                struct('animal_id', key.mouse_id), temporal(:,1), temporal.Fs, fps);
            tuple = dj.struct.join(tuple, sync_info);
            traces = aodReader(tuple.hdf5_file, 'Functional');
            sz = traces.reshapedSize;
            tuple.signal_duration = sz(1)/traces.Fs;
            tuple.sampling_rate = traces.Fs;
                        
			self.insert(tuple)
            makeTuples(aodpre.Trace, key)
		end
	end

end

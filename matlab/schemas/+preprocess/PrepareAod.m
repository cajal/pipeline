%{
preprocess.PrepareAod (imported) # information about AOD scans
-> preprocess.Prepare
---
%}


classdef PrepareAod < dj.Relvar
	methods

		function makeTuples(self, key)
	        % get volume info
            [path, name] = fetch1(experiment.Session * experiment.Scan & key,'scan_path','filename');
            
            % find the session that started immediatelly before the recording
            dirs = dir(fullfile(getLocalPath(path), [name(1:10) '*']));          
            names = datenum(vertcat(dirs.name),'yyyy-mm-dd_HH-MM-SS');
            timediff = names-datenum(name,'yyyy-mm-dd_HH-MM-SS');
            timeind = find(timediff<0);
            [~,i] = max(timediff(timeind));
            itime = timeind(i);
            key.hdf5_file = fullfile(getLocalPath(path), dirs(itime).name, 'AODAcq', name);
            
            % insert
			self.insert(key)
		end
	end

end
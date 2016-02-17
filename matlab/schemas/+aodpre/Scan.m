%{
aodpre.Scan (imported) # synchronized AOD scans
-> vis2p.Scans
---
hdf5_file                   : varchar(255)                  # raw data file
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
            timediff = str2double(names(:,[12 13 15 16 18 19]))- str2double(name([12 13 15 16 18 19]));
            timeind = find(timediff<0);
            [~,i] = max(timediff(timeind));
            itime = timeind(i);
            tuple = key;
            tuple.hdf5_file = fullfile(getLocalPath('M:\Mouse\'), dirs(itime).name, 'AODAcq', name);
            self.insert(tuple)
        end
    end
    
end

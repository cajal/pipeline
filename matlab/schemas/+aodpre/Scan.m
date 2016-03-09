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
            % find the session that started immediatelly before the recording
            names = datenum(vertcat(dirs.name),'yyyy-mm-dd_HH-MM-SS');
            timediff = names-datenum(name,'yyyy-mm-dd_HH-MM-SS');
            timeind = find(timediff<0);
            [~,i] = max(timediff(timeind));
            itime = timeind(i);
            tuple = key;
            tuple.hdf5_file = fullfile(getLocalPath('M:\Mouse\'), dirs(itime).name, 'AODAcq', name);
            self.insert(tuple)
        end
    end
    
end

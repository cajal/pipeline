function [dat dt t0] = loadHWS(fn,group,waveform)
% Function for loading data written by HWS from NI Labview
dat = [];

fid = H5F.open(fn,0,0);

gid = H5G.open(fid,'/');

% iterate through all the wfm_group
for i = 0 : H5G.get_num_objs(gid)-1
    
    % check the name attribute of the wfm_group/id group
    id_group = H5G.open(gid,[H5G.get_objname_by_idx(gid,i) '/id']);    
    a = H5A.open_name(id_group,'name');
    name = H5A.read(a,H5A.get_type(a))'; name(end) = [];
    H5A.close(a);

    if strcmp(name,group)  % if the group name matches the desired one
        
        % iterate through all the traces and check their names
        traces_group = H5G.open(gid,[H5G.get_objname_by_idx(gid,i) '/traces']);        
        for j = 0 : H5G.get_num_objs(traces_group)-1
            
            % open a trace and get its name
            trace = H5G.open(traces_group,H5G.get_objname_by_idx(traces_group,j));            
            a = H5A.open_name(trace,'name');
            name = H5A.read(a,H5A.get_type(a))'; name(end) = [];
            H5A.close(a);
            
            if strcmp(name,waveform)
                xaxis = H5G.open(trace, 'x-axis');
                
                try
                    a = H5A.open_name(xaxis,'increment');
                    dt = H5A.read(a,H5A.get_type(a));
                    H5A.close(a);
                catch
                    dt = 0;
                end

                try
                    a = H5A.open_name(xaxis,'ref_time');
                    datatypeID = H5T.copy('H5T_NATIVE_UINT64');
                    t0 = H5A.read(a,datatypeID); 
                    H5A.close(a);
                catch
                    t0 = 0;
                end
                
                H5G.close(xaxis);
                
                yaxis = H5G.open(trace, 'y-axis/data_vector');
                dset = H5D.open(yaxis,'data');
                datatypeID = H5T.copy('H5T_NATIVE_DOUBLE');
                dat = H5D.read(dset,datatypeID,'H5S_ALL',0,0);
                H5D.close(dset);
                H5G.close(yaxis);                
            end
            
            H5G.close(trace);
        end        
        
        H5G.close(traces_group);
    end
    
    H5G.close(id_group);
end
H5G.close(gid);

H5F.close(fid);



return

info = hdf5info(fn);
dat = [];

matchedGroup = 0;
matchedWfm = 0;

for i = 1:length(info.GroupHierarchy.Groups)
    % waveform name is saved in the second group of each of the root groups
    if strcmp(info.GroupHierarchy.Groups(i).Groups(2).Attributes.Value.Data,group)
        matchedGroup = i;       
    end
end

assert(matchedGroup ~= 0, 'Could not find requested waveform group');
    
for i = 1:length(info.GroupHierarchy.Groups(matchedGroup).Groups(3).Groups)
    if strcmp(info.GroupHierarchy.Groups(matchedGroup).Groups(3).Groups(i).Attributes(1).Value.Data,waveform)
        matchedWfm = i;
    end
end

assert(matchedWfm ~= 0, 'Could not find requested waveform in group');
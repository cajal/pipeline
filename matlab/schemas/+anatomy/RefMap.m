%{
# Reference map
-> mice.Mice
ref_idx                 : smallint        # reference map index for each animal
---
-> experiment.Scan
pxpitch                 : double          # estimated pixel pitch of the reference map (microns per pixel)
ref_table               : varchar(256)    # reference table
ref_map                 : mediumblob      # reference map
%}

classdef RefMap < dj.Manual
    methods
        function exit_tuple = createRef(obj,REF,pxpitch)
            
            if isstruct(REF)  % by default use OptImageBar as reference map if not supplied
                ret_keys = fetch(map.RetMapScan & REF,'ORDER BY ret_idx ASC, axis DESC');
                assert(~isempty(ret_keys),'Create a retinotopic map or specify a reference map explicitly!')
                REF = map.OptImageBar & ret_keys(1);
            end
                   
            % get primary key
            tuple = fetch(experiment.Scan & fetch(REF));
            ref_indexes = fetchn(obj & (mice.Mice & tuple),'ref_idx');
            if ~isempty(ref_indexes)
                tuple.ref_idx = max(ref_indexes)+1;
            else
                tuple.ref_idx = 1;
            end
            exit_tuple = tuple;
            
            % get map info
            if ismatrix(REF) && numel(REF)>4 % handle image ref_map
                tuple.ref_map = REF;
                tuple.ref_table = [];
                if nargin>2
                    tuple.pxpitch = pxpitch;
                else
                    tuple.pxpitch = 1;
                end
            else    % get map from coresponding table
                tuple.ref_table = REF.className;
                switch REF.className
                    case 'experiment.Scan'
                        tuple.pxpitch = 1;
                        tuple.ref_map = [];
                    case 'map.OptImageBar'
                        % get the intrinsic vessel map
                        [maps,   pxpitch] = fetchn(REF,'vessels','pxpitch');
                        tuple.ref_map = maps{1};
                        tuple.pxpitch = pxpitch(1);
                        assert(~isempty(tuple.ref_map),'No maps found!');
                    case 'meso.SummaryImagesAverage'
                        tuple.ref_map = fetch1(REF,'average_image');
                        ref_height = fetch1(meso.ScanInfoField & REF,'um_height');
                        tuple.pxpitch = ref_height/size(tuple.ref_map,1);
                        assert(~isempty(tuple.ref_map),'No maps found!');
                end
                
            end
            
            
            % insert
            insert(obj,tuple)
        end
    end
end


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

classdef RefMap < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create ref map
            insert( obj, key );
        end
    end
    
    methods
        function createRef(obj,REF,pxpitch)
            
            % get primary key
            tuple = fetch(experiment.Scan & REF);
            ref_indexes = fetchn(obj & (mice.Mice & tuple),'ref_idx');
            if ~isempty(ref_indexes)
                tuple.ref_idx = max(ref_indexes)+1;
            else
                tuple.ref_idx = 1;
            end
            
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
                    case 'map.OptImageBar'
                        % get the intrinsic vessel map
                        [maps,   pxpitch] = fetchn(REF,'vessels','pxpitch');
                        tuple.ref_map = maps{1};
                        tuple.pxpitch = pxpitch(1);
                    case 'meso.SummaryImagesAverage'
                        tuple.ref_map = fetch1(REF,'average_image');
                        ref_height = fetch1(meso.ScanInfoField & REF,'um_height');
                        tuple.pxpitch = ref_height/size(tuple.ref_map,1);
                end
                
            end
            assert(~isempty(tuple.ref_map),'No maps found!');
            
            % insert
            makeTuples(obj,tuple)
        end
    end
end


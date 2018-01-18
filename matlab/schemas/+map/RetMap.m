%{
# Retinotopy map
-> mice.Mice
ret_idx                 : smallint        # retinotopy map index for each animal
---
%}

classdef RetMap < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create ref map
            insert( obj, key );
        end
    end
    
    methods
        function tuple = createRetRef(self,keys,ret_idx)
            
            % select animal_id
            tuple = fetch(mice.Mice & keys);
            assert(length(tuple)==1,'Specify one animal_id!')
            
            % find opt maps if not provided
            if ~isfield(keys(1),'scan_idx')
                keys = [];
                for axis = {'horizontal','vertical'}
                    key.axis = axis{1};
                    hkeys = fetch(map.OptImageBar & tuple & key,'ORDER BY session, scan_idx');
                    if ~isempty(hkeys)
                        if length(hkeys)>1
                            f = clf;
                            for ikey = 1:length(hkeys)
                                subplot(ceil(sqrt(length(hkeys))),ceil(sqrt(length(hkeys))),ikey)
                                plot(map.OptImageBar & hkeys(ikey),'figure',f,'subplot',true,'exp',2,'sigma',2)
                                title(num2str(ikey))
                            end
                            
                            fprintf('\n Multiple maps found for animal %d \n',hkeys(1).animal_id);
                            ikey = input('Select key:');
                        else
                            ikey = 1;
                        end
                        keys{end+1} = hkeys(ikey);
                    else
                        fprintf('No %s map found!',axis{1})
                    end
                end
                if ~isempty(keys)
                    keys = cell2mat(keys);
                else
                    disp 'No maps found, please specify...'
                    return
                end
            end
            
            % set reference index
            if nargin<3 && ~exists(self & (map.RetMapScan & keys))
                ret_idx = max([0;fetchn(self & (mice.Mice & keys),'ret_idx')])+1;
            elseif nargin<3
                ret_idx = fetch1(self & (map.RetMapScan & keys),'ret_idx');
            end
            tuple.ret_idx = ret_idx;
            
            % insert if not found
            if ~exists(self & tuple)
                makeTuples(self, tuple)
            end
            
            % insert dependent keys
            for key = keys(:)'
                key.ret_idx = ret_idx;
                if ~exists(map.RetMapScan & key)
                    makeTuples(map.RetMapScan, key);
                else
                    str = struct2cell(key);
                    fprintf('\nTuple already exists! \n AnimalId: %d Session: %d ScanIdx: %d Axis: %s RetIdx: %d\n',str{:});
                end
            end
        end
    end
end


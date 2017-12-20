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
        function createRet(self,keys,ret_idx)
            
            tuple = fetch(mice.Mice & keys);
            
            % set reference index
            if nargin<3 && ~exists(self & (map.RetMapScan & keys))
                ret_idx = max([0;fetchn(self & (mice.Mice & keys),'ret_idx')])+1;
            elseif nargin<3
                ret_idx = fetch1(self & (map.RetMapScan & keys),'ret_idx');
            end
            tuple.ret_idx = ret_idx;

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


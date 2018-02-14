%{
# Retinotopy map
-> mice.Mice
ret_idx                 : smallint        # retinotopy map index for each animal
---
%}

classdef RetMap < dj.Manual
    methods
        function createRet(self,keys,ret_idx)
            
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
                                title(sprintf('%d: session:%d scan_idx:%d',ikey,hkeys(ikey).session,hkeys(ikey).scan_idx))
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
                insert(self, tuple)
            end
            
            % insert dependent keys
            [keys.ret_idx] = deal(ret_idx);
            for key = keys(:)'
                if ~exists(map.RetMapScan & key)
                    insert(map.RetMapScan, key);
                else
                    str = struct2cell(key);
                    fprintf('\nTuple already exists! \n AnimalId: %d Session: %d ScanIdx: %d Axis: %s RetIdx: %d\n',str{:});
                end
            end
            
            % populate SignMap
            if ~exists(map.SignMap & tuple)
                extractSign(map.SignMap,tuple,'manual',0);
            end
        end
        
        
        function ret_key = getRetKey(self, key)
            if ~exists(self & key)
                createRet(map.RetMap,fetch(mice.Mice & key));
            end
            ret_key = rmfield(fetch(map.RetMapScan & (self & key) & 'axis="horizontal"'),'axis');
        end
        
        
        function background = getBackground(self, varargin)
            
            params.exp = 1.5;
            params.sigma = 2;
            
            params = ne7.mat.getParams(params,varargin);
            
            assert(exists(self), 'No retinotopy map exists!')
            
            % get horizontal map
            Hor = [];
            vessels = fetch1(map.OptImageBar & (map.RetMapScan & self) & 'axis="horizontal"','vessels');
            vessels = single(vessels);
            [Hor(:,:,1),Hor(:,:,2),Hor(:,:,3)] = plot(map.OptImageBar & (map.RetMapScan & self) ...
                & 'axis="horizontal"','exp',params.exp,'sigma',params.sigma);
            background = cat(4,repmat(vessels/max(vessels(:)),1,1,3),hsv2rgb(Hor));
           
            % get vertical map
            if exists(map.OptImageBar & (map.RetMapScan & self) & 'axis="vertical"')
                Ver = [];
                [Ver(:,:,1),Ver(:,:,2),Ver(:,:,3)] = plot(map.OptImageBar & (map.RetMapScan & self) ...
                    & 'axis="vertical"','exp',params.exp,'sigma',params.sigma);
                background = cat(4,background,hsv2rgb(Ver));
            end
            
            % get sign map
            if exists(map.SignMap & self)
                sign_map = fetch1(map.SignMap & self,'sign_map');
                background = cat(4,background,hsv2rgb(sign_map));
            end
        end
    end
end


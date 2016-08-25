%{
tuning.MonetRFMap (computed) # receptive fields from the monet stimulus
-> tuning.MonetRF
-> preprocess.SpikesRateTrace
-----
map : longblob
%}

classdef MonetRFMap < dj.Relvar 
    
    methods
        
        function save(self)
            path = '/Volumes/scratch01/RF-party';
            for key = self.fetch'
                disp(key)
                fullpath = getLocalPath(fullfile(path, ...
                    sprintf('%05d-%s-%s',key.animal_id, ...
                    fetch1(preprocess.Method & key, 'method_name'),...
                    fetch1(preprocess.SpikeMethod & key, 'short_name'))));
                if ~exist(fullpath, 'dir')
                    mkdir(fullpath);
                end
                
                map = fetch1(self & key, 'map');
                mx = max(abs(map(:)));
                map = round(map/mx*31.5 + 32.5);
                cmap = ne7.vis.doppler;
                for i=1:min(size(map,3),6)
                    im = reshape(cmap(map(:,:,i),:),[size(map,1) size(map,2) 3]);
                    f = fullfile(fullpath, sprintf('monet%03u-%u-%03u-%u.png', ...
                        key.scan_idx, key.slice, key.mask_id, i));
                    imwrite(im, f, 'png')
                end
            end
        end
        
        function makeTuples(self,key)
              self.insert(key)
        end
    end
    
end
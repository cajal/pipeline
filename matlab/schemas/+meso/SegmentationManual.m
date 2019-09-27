%{
# masks created manually
-> meso.Segmentation
%}


classdef SegmentationManual < dj.Computed
    
    properties
        popRel  = meso.SummaryImagesAverage
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            images = ne7.mat.normalize(fetch1(meso.SummaryImagesAverage & key, 'average_image'));
            
            % add second channel info
            key2 = key;
            channels = [1 2];
            key2.channel = channels(key.channel~=channels);
            if exists(meso.SummaryImagesAverage & key2);
               images = repmat(images,1,1,3);
               image2 = ne7.mat.normalize(fetch1(meso.SummaryImagesAverage & key2, 'average_image'));
               images(:,:,:,2) = repmat(image2,1,1,3);
               images(:,:,channels(key.channel~=channels),3) = images(:,:,1,1);
               images(:,:,channels(key.channel==channels),3) = image2;
            end
            
            % remove baseline
            masks = ne7.ui.paintMasks(images);
            assert(~isempty(masks), 'user aborted segmentation')
            key.segmentation_method = 1;
            r_key = rmfield(key,'pipe_version');
            r_key.compartment = 'unknown';
            
            % insert parents
            insert(meso.SegmentationTask,r_key);
            insert(meso.Segmentation,key);
            self.insert(key)
            
            % Insert Masks
            unique_masks = unique(masks);
            key.mask_id = 0;
            for mask = unique_masks(unique_masks>0)'
                key.mask_id = key.mask_id+1;
                key.pixels = find(masks==mask);
                key.weights = ones(size(key.pixels));
                insert(meso.SegmentationMask,key)
            end
        end 
    end
    
    methods
        function populateAll(self,keys)
            keys = fetch(experiment.Scan - (meso.Segmentation & keys) & keys);
            for key = keys'
                channels = unique(fetchn(meso.SummaryImagesAverage & key,'channel'));
                key.channel = channels(1);
                populate(self,key)
                
                if length(channels)>1
                    % insert parents
                    tuple = fetch(meso.SegmentationTask & key,'*');
                    tuple.channel = channels(2);
                    insert(meso.SegmentationTask,tuple);

                    tuple = fetch(meso.Segmentation & key,'*');
                    tuple.channel = channels(2);
                    insert(meso.Segmentation,tuple);

                    tuple = fetch(self & key,'*');
                    tuple.channel = channels(2);
                    self.insert(tuple)

                    % Insert Masks
                    keys = fetch(meso.SegmentationMask & key,'*');
                    for tuple = keys'
                        tuple.channel = channels(2);
                        insert(meso.SegmentationMask,tuple)
                    end
                end
            end
        end
    end
end

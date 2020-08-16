%{
# masks created manually
-> reso.Segmentation
---
%}


classdef SegmentationManual < dj.Computed
    
    properties
        popRel  = reso.SummaryImagesAverage
    end
    
    methods(Access=protected)
        function makeTuples(self, key)
            images = ne7.mat.normalize(fetch1(reso.SummaryImagesAverage & key, 'average_image'));
            
            % add second channel info
            key2 = key;
            channels = [1 2];
            key2.channel = channels(key.channel~=channels);
            if exists(reso.SummaryImagesAverage & key2);
               images = repmat(images,1,1,3);
               image2 = ne7.mat.normalize(fetch1(reso.SummaryImagesAverage & key2, 'average_image'));
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
            insert(reso.SegmentationTask,r_key);
            insert(reso.Segmentation,key);
            self.insert(key)
            
            % Insert Masks
            unique_masks = unique(masks);
            key.mask_id = 0;
            for mask = unique_masks(unique_masks>0)'
                key.mask_id = key.mask_id+1;
                key.pixels = find(masks==mask);
                key.weights = ones(size(key.pixels));
                insert(reso.SegmentationMask,key)
            end
        end 
    end
    
    methods
        function populateAll(self,keys)
            keys = fetch(experiment.Scan - (reso.Segmentation & keys) & keys);
            for key = keys'
                channels = unique(fetchn(reso.SummaryImagesAverage & key,'channel'));
                key.channel = channels(1);
                populate(self,key)
                
                if length(channels)>1
                    % insert parents
                    tuple = fetch(reso.SegmentationTask & key,'*');
                    tuple.channel = channels(2);
                    insert(reso.SegmentationTask,tuple);

                    tuple = fetch(reso.Segmentation & key,'*');
                    tuple.channel = channels(2);
                    insert(reso.Segmentation,tuple);

                    tuple = fetch(self & key,'*');
                    tuple.channel = channels(2);
                    self.insert(tuple)

                    % Insert Masks
                    keys = fetch(reso.SegmentationMask & key,'*');
                    for tuple = keys'
                        tuple.channel = channels(2);
                        insert(reso.SegmentationMask,tuple)
                    end
                end
            end
        end    
    end
    
    methods (Static)
        function editMasks(self,key)
              [masks ,tkeys]= fetchn(reso.SegmentationMask & key,'pixels');
              images = fetchn(reso.SummaryImagesAverage & key, 'average_image', 'ORDER BY channel');
              mask = zeros(size(images{1}));
              for imask = 1:length(masks)
                mask(masks{imask}) = tkeys(imask).mask_id;
              end
              % plot masks
              un = unique(mask(:));
              nmask = zeros(size(mask));
              for i = 1:length(un)
                nmask(mask==un(i)) = i;
              end
              masks = ne7.ui.paintMasks(images{1},nmask);
              assert(~isempty(masks), 'user aborted segmentation')
              reply = input('Do you want to update mask? Y/N [N]:','s');
              if isempty(reply)
                  reply = 'N' ;
              end
              if reply == 'Y'
                  key.segmentation_method = 1;
                  r_key = rmfield(key,'pipe_version');
                  r_key.compartment = 'unknown';
                  % insert parents
                  delQuick(self & key)
                  % Insert Masks
                  unique_masks = unique(masks);
                  key.mask_id = 0;
                  for mask = unique_masks(unique_masks>0)'
                    key.mask_id = key.mask_id+1;
                    key.pixels = find(masks==mask);
                    key.weights = ones(size(key.pixels));
                    insert(reso.SegmentationMask,key)
                  end
              end
        end 
        
        % take a full mask that was manually constructed over the entire
        % cell body and trim it down to pixels in the average image that represent presence of above threshold baseline fluoroscence 
        % mask in the database is a vector containing the pixel locations in a 2-D image.
        % Pixel locations increment column wise in the image, i..e pixel 1
        % is row 1, col 1 and pixel 2 is row 2, col 1
        function thresholdMasks(self,key,new_segmentation_method)
              self.clearOldThresholdMasks(self,key,new_segmentation_method) ;
              [masks ,tkeys]= fetchn(reso.SegmentationMask & key,'pixels'); % get all masks and their keys
              images = fetchn(reso.SummaryImagesAverage & key, 'average_image', 'ORDER BY channel'); % get an image that is averaged across time
              nmask = {} ;
              for imask = 1:length(masks)
                  nmask{imask} = masks{imask} ;
                  im = images{tkeys(imask).channel} ; % use channel from database
                  sel_im = im(nmask{imask}) ; % parts of image selected from image by this mask
                  sel_im = sel_im - mean(sel_im) ;
                  sd = std(sel_im) ;
                  sd_th = 0.0 ; % arbitrarily chosen as threshold above which mask pixels should be accepted
                  idx = find(sel_im < sd_th*sd) ;
                  nmask{imask}(idx) = [] ; % delete the entries from mask that is below th
              end
              
              % Insert Masks
              for mask_id = 1:length(tkeys)
                    nkey = tkeys(mask_id) ; 
                    nkey.segmentation_method = new_segmentation_method; % replace method by new method
                    nkey.pixels = nmask{mask_id}; % replace mask by new mask
                    nkey.weights = ones(size(nkey.pixels)) ;
                    k = reso.SegmentationTask & nkey ; % make sure parent tables have the new key
                    if ~count(k)
                        stkey = nkey ;
                        stkey = rmfield(stkey,'mask_id') ;
                        stkey = rmfield(stkey,'pixels') ;
                        stkey = rmfield(stkey,'weights') ;
                        stkey = rmfield(stkey,'pipe_version') ;
                        stkey.compartment = 'unknown' ;
                        insert(reso.SegmentationTask, stkey) ;
                    end
                    k = reso.Segmentation & nkey ;
                    if ~count(k)
                        skey = nkey ;
                        skey = rmfield(skey,'mask_id') ;
                        skey = rmfield(skey,'pixels') ;
                        skey = rmfield(skey,'weights') ;
                        insert(reso.Segmentation, skey) ;
                    end
                    insert(reso.SegmentationMask,nkey)
              end
        end 
        
        
        function clearOldThresholdMasks(self,key,new_segmentation_method)
              obj=vreso.getSchema ;
              key.segmentation_method = new_segmentation_method ; % replace the source method because we are searching for the new method tuples
              [sm ,tkeys]= fetchn(reso.SegmentationMask & key,'segmentation_method'); % get all masks and their keys
              for ii=1:length(sm)
                  if sm(ii) == new_segmentation_method
                      try
                          [msm,session,scan_idx,mskeys] = fetchn(obj.v.ResoMatch, 'segmentation_method', 'session', 'scan_idx') ;
                          for jj=1:length(mskeys)                          
                              if session(jj) == key.session && scan_idx(jj) == key.scan_idx && msm(jj) == new_segmentation_method
                                  delQuick(obj.v.ResoMatch & mskeys(jj))
                              end
                          end
                      catch
                      end
                      try
                        delQuick(reso.FluorescenceTrace & tkeys(ii)) ;
                      catch
                      end
                      try
                        delQuick(reso.Fluorescence & tkeys(ii)) ;
                      catch
                      end
                      try
                        delQuick(reso.SegmentationMask & tkeys(ii)) ;
                      catch
                      end
                      try
                        delQuick(reso.Segmentation & tkeys(ii)) ;
                      catch
                      end
                      try
                        delQuick(reso.SegmentationTask & tkeys(ii)) ;
                      catch
                      end
                  end
              end
        end
    end
end


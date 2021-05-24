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
        % key refers to the source segmentation method from which new masks
        % will be derived and inserted into SegmentationMask table
        % skey refers to the segmentation table and a tuple for
        % new_segmentation_mask
        function thresholdMasks(self,key,skey,new_segmentation_method)
              self.clearOldThresholdMasks(self,key,skey,new_segmentation_method) ;
              [masks ,tkeys]= fetchn(reso.SegmentationMask & key,'pixels'); % get all masks and their keys
              images = fetchn(reso.SummaryImagesAverage & key, 'average_image', 'ORDER BY channel'); % get an image that is averaged across time
              [th_type,th_value] = fetchn(reso.SegmentationMaskThreshold & skey, 'threshold_type', 'threshold_value') ;
              
              nmask = {} ;
              for imask = 1:length(masks)
                  nmask{imask} = masks{imask} ;
                  im = images{tkeys(imask).channel} ; % use channel from database
                  sel_im = im(nmask{imask}) ; % parts of image selected from image by this mask
                  mn = mean(sel_im) ;
                  sel_im = sel_im - mn ;
                  if (th_type==1)
                    sd = std(sel_im) ;
                    th = th_value*sd ; % threshold above which mask pixels should be accepted
                  else
                    th = th_value - mn ;
                  end                    
                  idx = find(sel_im < th) ;
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
        
        
        function clearOldThresholdMasks(self,key,skey,new_segmentation_method)
              obj=vreso.getSchema ;
              key.segmentation_method = new_segmentation_method ; % replace the source method because we are searching for the new method tuples
              [sm ,tkeys]= fetchn(reso.SegmentationMask & key,'segmentation_method'); % get all masks and their keys
              for ii=1:length(sm)
                  if sm(ii) == new_segmentation_method
                      try
                          [msm,session,scan_idx,mskeys] = fetchn(obj.v.ResoMatch, 'segmentation_method', 'session', 'scan_idx') ; % ResoMatch does not have scan as primary key
                          for jj=1:length(mskeys)                          
                              if session(jj) == key.session && scan_idx(jj) == key.scan_idx && msm(jj) == new_segmentation_method
                                  delQuick(obj.v.ResoMatch & mskeys(jj))
                              end
                          end
                      catch
                      end
                      delQuick(reso.FluorescenceTrace & tkeys(ii)) ;
                      delQuick(reso.Fluorescence & tkeys(ii)) ;
                      delQuick(reso.SegmentationMask & tkeys(ii)) ;
                  end
              end
        end
        

        
        % take a full mask that was manually constructed over the entire
        % cell body and make one new mask for each pixel in the full mask
        % mask in the database is a vector containing the pixel locations in a 2-D image.
        % Pixel locations increment column wise in the image, i..e pixel 1
        % is row 1, col 1 and pixel 2 is row 2, col 1
        % key refers to the source segmentation method from which new masks
        % will be derived and inserted into SegmentationMask table
        % skey refers to the segmentation table and a tuple for
        % new_segmentation_mask
        function masksPerPixel(self,key,skey,new_segmentation_method)
              self.clearOldMasksPerPixel(self,key,skey,new_segmentation_method) ;
              [masks ,tkeys]= fetchn(reso.SegmentationMask & key,'pixels'); % get all masks and their keys
              images = fetchn(reso.SummaryImagesAverage & key, 'average_image', 'ORDER BY channel'); % get an image that is averaged across time

              vres_obj = vreso.getSchema ;
              [msm, field, channel, pipe_version, mask_id, animal_id, session, scan_idx, mskeys] = ...
                  fetchn(vres_obj.v.ResoMatch, 'segmentation_method', 'field', 'channel', 'pipe_version', 'mask_id', 'animal_id', 'session', 'scan_idx') ; % ResoMatch does not have scan as primary key
              
%             can only do this if there are less than 2000 pixels in original mask and less than 15 masks
%             these limits come because mask_id is a small int and can only have numbers between -32K and +32K 
              if length(masks)>7
                fprintf('Cannot have more than 7 masks in a scan, first 7 out of %d will be used\n', length(masks)) ;
                masks = masks(1:7) ;
              end
              clearvars('stkey_list') ;
              stkey_count = 0 ;
              clearvars('skey_list') ;
              skey_count = 0 ;
              clearvars('nkey_list') ;
              nkey_count = 0 ;

%             process all masks in this scan              
              for imask = 1:length(masks)
%                check if this mask id is in vreso, brut forcing                 
                 maskinvreso = false ;
                 for jj=1:length(mskeys)                          
                      if animal_id(jj) == tkeys(imask).animal_id && ...
                              session(jj) == tkeys(imask).session && ...
                              scan_idx(jj) == tkeys(imask).scan_idx && ...
                              field(jj) == tkeys(imask).field && ...
                              channel(jj) == tkeys(imask).channel && ...
                              pipe_version(jj) == tkeys(imask).pipe_version && ...
                              msm(jj) == tkeys(imask).segmentation_method && ...
                              mask_id(jj) == tkeys(imask).mask_id
                          maskinvreso = true ;
                          break ;
                      end
                 end
                 
                 if maskinvreso
                     nmask = masks{imask} ;
                     samekeywithnewmethod = tkeys(imask) ;
                     samekeywithnewmethod.segmentation_method = new_segmentation_method; % replace method by new method
                     samekeywithnewmethod.pixels = nmask ;
                     samekeywithnewmethod.weights = ones(size(samekeywithnewmethod.pixels)) ;
                     insert(reso.SegmentationMask, samekeywithnewmethod) ;                 % store the parent mask as is
                     try
                          if length(nmask)>3999
                            fprintf('Cannot have more than 3999 pixels in each mask, will use first 3999 out of %d pixels\n',length(nmask)) ;
                            nmask = nmask(1:3999) ;
                          end
                          for jj = 1:length(nmask)
                                tmask(1) = nmask(jj) ;                      
                              % insert mast into database
                                nkey = tkeys(imask); 
                                nkey.mask_id = floor((nkey.mask_id*4E3)+jj) ;
                                nkey.segmentation_method = new_segmentation_method; % replace method by new method
                                nkey.pixels = tmask ; % replace mask by new mask
                                nkey.weights = ones(size(nkey.pixels)) ;
                                if stkey_count == 0
                                    k = reso.SegmentationTask & nkey ; % make sure parent tables have the new key
                                    if ~count(k)
                                        stkey = nkey ;
                                        stkey = rmfield(stkey,'mask_id') ;
                                        stkey = rmfield(stkey,'pixels') ;
                                        stkey = rmfield(stkey,'weights') ;
                                        stkey = rmfield(stkey,'pipe_version') ;
                                        stkey.compartment = 'unknown' ;
                                        stkey_count = stkey_count + 1 ;
                                        stkey_list(stkey_count) = stkey ;
                                    end
                                end
                                if skey_count == 0
                                    k = reso.Segmentation & nkey ;
                                    if ~count(k) && skey_count == 0
                                        skey = nkey ;
                                        skey = rmfield(skey,'mask_id') ;
                                        skey = rmfield(skey,'pixels') ;
                                        skey = rmfield(skey,'weights') ;
                                        skey_count = skey_count + 1 ;
                                        skey_list(skey_count) = skey ;
                                     end
                                end
                                nkey_count = nkey_count + 1 ;
                                nkey_list(nkey_count) = nkey ;
                          end
                      catch err
                          fprintf('Error Id: %s\nError Message: %s\n\n',err.identifier, err.message) ;
                     end
                 end
              end
              if stkey_count > 0
                insert(reso.SegmentationTask, stkey_list(1:stkey_count)) ;
              end
              if skey_count > 0
                insert(reso.Segmentation, skey_list(1:skey_count)) ;
              end
              if nkey_count > 0
                insert(reso.SegmentationMask,nkey_list(1:nkey_count)) ;    
              else
                fprintf('No ephys and optical matching masks found\n') ;
              end
        end
        
        function clearOldMasksPerPixel(self,key,skey,new_segmentation_method)
              obj=vreso.getSchema ;
              key.segmentation_method = new_segmentation_method ; % replace the source method because we are searching for the new method tuples
              [sm ,tkeys]= fetchn(reso.SegmentationMask & key,'segmentation_method'); % get all masks and their keys
              for ii=1:length(sm)
                  if sm(ii) == new_segmentation_method
                      try
                          [msm,session,scan_idx,mskeys] = fetchn(obj.v.ResoMatch, 'segmentation_method', 'session', 'scan_idx') ; % ResoMatch does not have scan as primary key
                          for jj=1:length(mskeys)                          
                              if session(jj) == key.session && scan_idx(jj) == key.scan_idx && msm(jj) == new_segmentation_method
                                  delQuick(obj.v.ResoMatch & mskeys(jj))
                              end
                          end
                      catch
                      end
                  end
              end
              delQuick(reso.ScanSetUnitInfo & key) ;
              delQuick(reso.ScanSetUnit & key) ;
              delQuick(reso.FluorescenceTrace & key) ;
              delQuick(reso.SegmentationMask & key) ;
              delQuick(fuse.ScanSetReso & key) ;
              delQuick(fuse.ScanSetUnit & key) ;
              delQuick(fuse.ScanSet & key) ;
              delQuick(reso.ScanSet & key) ;
              delQuick(reso.Fluorescence & key) ;
        end
        
    end
end








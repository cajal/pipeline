%{
# Simulated Filter Responses
-> stimulus.MovieClip
filter_idx            : smallint     # filter index
---
resp                  : mediumblob   # filter response to the center of the movie
%}

classdef SimResp < dj.Imported
    methods(Access=protected)
        function makeTuples(obj,key) %create clips
            % get video file
            filename = export(stimulus.MovieClip & key);
            vidReader = VideoReader(filename{1});
            
            % initialize
            iframe = 0;
            nframes =  floor(vidReader.Duration*vidReader.FrameRate);
            Mov = nan(vidReader.Height,vidReader.Width,nframes);
            
            % fun for each frame
            while hasFrame(vidReader)
                iframe = iframe+1;
                frame =  readFrame(vidReader);
                Mov(:,:,iframe) = frame(:,:,1);
            end
            Mov = Mov(:,:,1:iframe);

            % center ~20deg in ~100deg coverage
            sz = round(size(Mov,2)/5/2);
            ct = round(size(Mov)/2);
            Mov = Mov(ct(1)-sz(1):ct(1)+sz,ct(2)-sz:ct(2)+sz,:);
          
            % get filters
            filters = obj.getFilters(size(Mov,1));
            
            % get responses
            mov = repmat(permute(Mov/255,[1 2 4 3]),[1 1 size(filters,3) 1]);
            Resp = squeeze(mean(mean(bsxfun(@times,filters,mov),1),2));
            
            % separate by sign
            Resp(end+1:size(Resp,1)*2,:) = -Resp;
            Resp(Resp<0) = 0;
            
            for i = 1:size(Resp,1)
                tuple = key;
                tuple.filter_idx = i;
                tuple.resp = single(Resp(i,:)');
                insert( obj, tuple );
            end
        end
    end
    
    methods
       
        function filters = getFilters(self,target_size)
            Fil = load(getLocalPath('/lab/users/Manolis/Matlab/OLD/sparsenet/A16.mat'));
            [L, M]=size(Fil.A);sz=sqrt(L);
            res = target_size/sz;
            filters = nan(size(imresize(reshape(Fil.A(:,1),sz,sz),res),1),...
                size(imresize(reshape(Fil.A(:,1),sz,sz),res),1),5);
            for i = 1:M
                filters(:,:,i) = imresize(reshape(Fil.A(:,i),sz,sz),res);
            end    
        end
    end
end
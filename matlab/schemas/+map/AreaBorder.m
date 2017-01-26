%{
map.AreaBorder (imported) # clips from movies
-> map.OptImageBar
---
max_reversal                : mediumblob            # full file name
fixed_map                   : mediumblob            # improved phase map
border                      : mediumblob            # estimated area border
V1_mask                     : mediumblob            # mask of V1 area
LM_mask                     : mediumblob            # mask of LM area
%}

classdef AreaBorder < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = map.OptImageBar & 'axis="horizontal"' & (experiment.Scan & 'software="scanimage"')
    end
    
    methods(Access=protected)
        
        function makeTuples(obj,key) %create clips
            
            h = plot(map.OptImageBar & key,'sigma',20,'exp',2,'amp',0);
            
            gwin = 70; % gaussian filter 
            rfMap = normalize(h);
            fixedMap = convn(rfMap,gausswin(gwin)*gausswin(gwin)','same'); 
            [~,reversal] = max(fixedMap,[],2);
            
            % Set up fittype and options.
            [xData, yData] = prepareCurveData( [],reversal );
            ft = fittype( 'poly2' );
            opts = fitoptions( ft );
            opts.Lower = [-Inf -Inf -Inf];
            opts.Upper = [Inf Inf Inf];
            opts.Normalize = 'on';
            
            % Fit model to data.
            [fitresult, ~] = fit( xData, yData, ft, opts );
            border = mean(predint(fitresult,1:size(rfMap,1)),2);
            
            % create area masks
            mask = meshgrid(1:size(h,2),1:size(h,1));
            key.LM_mask = mask>border;
            key.V1_mask = mask<border;

            % insert data
            key.fixed_map = fixedMap;
            key.max_reversal = reversal;
            key.border = border;
            insert( obj, key );
            
        end
    end
    
    
    methods
        
        function plot(obj)
            
            key = fetch(obj);
            border = fetch1(obj,'border');
            nslices = fetch1(preprocess.PrepareGalvo & key,'nslices');
            f = figure;
            subplot(nslices+1,1,1)
            plot(map.OptImageBar & key,'sigma',20,'exp',2,'amp',0,'subplot',1,'figure',f)
            hold on
            plot(border,1:length(border),'r')
            
            for islice = 1:nslices
                key.slice = islice;
                subplot(nslices+1,1,islice+1)
                
                frame = fetch1(preprocess.PrepareGalvoAverageFrame & key,'frame');
                imagesc(frame)
                axis image
                axis off
                colormap gray
                hold on
                plot(border,1:length(border),'r')
            end
            
        end
        
    end
    
end
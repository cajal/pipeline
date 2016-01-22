%{
common.BrainSliceRegistration (imported) # registration points against the previous slice
-> common.BrainSliceImage
-----
n_points        : smallint  # the number of control points
input_points  : longblob   # control points
base_points     : longblob   # same points in the base image (previous slice)
%}

classdef BrainSliceRegistration < dj.Relvar & dj.AutoPopulate
    
    properties(Constant)
        table = dj.Table('common.BrainSliceRegistration')
        popRel = common.BrainSliceImage & 'first_slice=0'
    end
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            baseKey = key;
            baseKey.slice_id = key.slice_id - 1;
            if ~exists(common.BrainSliceImage & baseKey)
                warning 'no previous slice found... skipped'
            else
                disp 'reading images...'
                inputImg = imread(strtrim(fetch1(common.BrainSliceImage & key, ...
                    'slice_filepath')));
                baseImg  = imread(strtrim(fetch1(common.BrainSliceImage & baseKey,...
                    'slice_filepath')));
                inputxy = cache('input');
                basexy = cache('base');
                if isempty(inputxy)
                    [key.input_points, key.base_points] = cpselect(inputImg, baseImg, 'Wait', true);
                else
                    [key.input_points, key.base_points] = cpselect(inputImg, baseImg, inputxy, basexy, 'Wait', true);
                    cache
                end
                tform = cp2tform(key.input_points, key.base_points, 'similarity');
                clf
                disp 'displaying results'
                subplot 121
                imshowpair(inputImg(:,:,2), baseImg(:,:,2))
                title original
                subplot 122
                imshowpair(imtransform(inputImg(:,:,2), tform, 'xdata', [1 size(baseImg,2)], 'ydata', [1 size(baseImg,1)]), baseImg)
                title registered
                
                if strncmpi('y', input('Commit results? y|n >', 's'), 1)
                    key.n_points=size(key.input_points,1);
                    self.insert(key)
                end
                close all
            end
        end
    end
    
    methods        
        function refine(self)
            % refine an existing registration
            for key = fetch(self)'
                disp 'About to refine:'
                disp(key)
                cache
                [input,base] = fetch1(common.BrainSliceRegistration & key, 'input_points', 'base_points');
                cache('input',input)
                cache('base',base)
                del(common.BrainSliceRegistration & key)
                populate(common.BrainSliceRegistration, key);
                cache
            end
        end
    end
end


function ret = cache(field,value)
persistent CACHE
if ~nargout
    if ~nargin
        CACHE = struct;
    else
        CACHE.(field) = value;
    end
else
    if ~isempty(CACHE) && isfield(CACHE,field)
        ret = CACHE.(field);
    else
        ret = [];
    end
end    
end
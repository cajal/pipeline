function [loader, channel] = create_loader(key, do_squeeze)
if nargin < 2
    do_squeeze=1;
end
reader = preprocess.getGalvoReader(key);
switch fetch1(experiment.SessionFluorophore & key,'fluorophore')
    case 'Twitch2B'
        channel= [1,2];
        loader = @(islice, varargin)(twitch_loader(key, islice, reader, varargin{:}));
            
    otherwise
        channel = 1;  % TODO: change to more flexible choice
        if do_squeeze
            loader = @(islice, varargin)(squeeze(load_galvo_scan(key, islice, channel, reader, varargin{:})));
        else
            loader = @(islice, varargin)(load_galvo_scan(key, islice, channel, reader, varargin{:}));
        end
end


end


function scan = load_galvo_scan(key, islice, ichannel, reader, frames)
key.slice = islice;
fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);

if nargin < 4
    reader = preprocess.getGalvoReader(key);
end


if nargin < 5
    fprintf('\tLoading all frames\n');
    frames = 1:reader.nframes;
end

[r,c] = fetch1(preprocess.PrepareGalvo & key, 'px_height', 'px_width');


fprintf('\tLoading slice %d channel %d\n', islice, ichannel)
scan = zeros(r,c, 1, length(frames));
N = length(frames);

for iframe = frames
    if mod(iframe, 1000) == 0
        fprintf('\r\t\tloading frame %i (%i/%i)', iframe, iframe - frames(1) + 1, N);
    end
    scan(:, :, 1, iframe - frames(1) + 1) = fixMotion(fixRaster(reader(:,:,ichannel, islice, iframe)), iframe);
end
fprintf('\n');
end

function Y = twitch_loader(key, islice, reader, mask_range)
    if nargin < 4
        fprintf('\tLoading all frames\n');
        mask_range = 1:reader.nframes;
    end
    Y = squeeze(cat(3, load_galvo_scan(key, islice, 1, reader, mask_range), load_galvo_scan(key, islice, 2, reader, mask_range)));
    
% %     stride = 4;
% %     hs = hamming(2*stride+1);
% %     hs = hs./sum(hs);
% %     
%     fps = fetch1(preprocess.PrepareGalvo & key, 'fps');
%     ht = hamming(2*floor(fps/1)+1);
%     ht = ht./sum(ht);
%     
% %     y1 = imfilter(imfilter(Y1, hs, 'symmetric'), hs','symmetric');
% %     y2 = imfilter(imfilter(Y2, hs, 'symmetric'), hs','symmetric');
% %     y1 = y1(1:stride:end,1:stride:end,:);
% %     y2 = y2(1:stride:end,1:stride:end,:);
%     [d1,d2,fr] = size(Y1);
%     y1 = ne7.dsp.convmirr(reshape(Y1,[d1*d2,fr])',ht);
%     y2 = ne7.dsp.convmirr(reshape(Y2,[d1*d2,fr])',ht);
%     R = (y2-y1)./(y1+y2);
%     q = quantile(R, 0.01, 1);
%     b_free = 2716.16;
%     b_loaded = 616.00;
%     g_free = 2172.93;
%     g_loaded = 3738.65;
%     dg = g_loaded - g_free;
%     db = b_loaded - b_free;
% 
%     
%     gamma = median(-b_free/g_free * (q+1)./(q-1));
%     x = (-b_free + g_free*gamma - R.*(b_free + g_free*gamma))./(db - dg*gamma + R.*(db + dg*gamma));
%    
%     keyboard;
    end
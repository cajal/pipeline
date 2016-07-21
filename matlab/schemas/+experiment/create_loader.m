function [loader, channel] = create_loader(key)
reader = preprocess.getGalvoReader(key);
switch fetch1(experiment.SessionFluorophore & key,'fluorophore')
    case 'Twitch2B'
        channel= 0;
        loader = @(islice, mask_range)(twitch_loader(key, islice, mask_range, reader));
            
    otherwise
        channel = 1;  % TODO: change to more flexible choice
        loader = @(islice, mask_range)(squeeze(load_galvo_scan(key, islice, channel, mask_range, reader)));
end


end

function scan = load_galvo_scan(key, islice, ichannel, frames, reader)
key.slice = islice;
fixMotion = get_fix_motion_fun(preprocess.PrepareGalvoMotion & key);
fixRaster = get_fix_raster_fun(preprocess.PrepareGalvo & key);
if nargin < 5
    reader = preprocess.getGalvoReader(key);
end


if nargin < 4
    frames = 1:reader.nframes;
end

[r,c] = fetch1(preprocess.PrepareGalvo & key, 'px_height', 'px_width');


key.slice = islice;
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

function Y = twitch_loader(key, islice, mask_range, reader)
    Y1 = squeeze(load_galvo_scan(key, islice, 1, mask_range, reader));
    Y2 = squeeze(load_galvo_scan(key, islice, 2, mask_range, reader));
%     stride = 4;
%     hs = hamming(2*stride+1);
%     hs = hs./sum(hs);
%     
    fps = fetch1(preprocess.PrepareGalvo & key, 'fps');
    ht = hamming(2*floor(fps/4)+1);
    ht = ht./sum(ht);
    
%     y1 = imfilter(imfilter(Y1, hs, 'symmetric'), hs','symmetric');
%     y2 = imfilter(imfilter(Y2, hs, 'symmetric'), hs','symmetric');
%     y1 = y1(1:stride:end,1:stride:end,:);
%     y2 = y2(1:stride:end,1:stride:end,:);
    [d1,d2,fr] = size(Y1);
    y1 = ne7.dsp.convmirr(reshape(Y1,[d1*d2,fr])',ht);
    y2 = ne7.dsp.convmirr(reshape(Y2,[d1*d2,fr])',ht);
    R = y2./(y1+y2 + 1);
    q = quantile(R, 0.01, 1);
    b_free = 2716.16;
    b_loaded = 616.00;
    g_free = 2172.93;
    g_loaded = 3738.65;
    dg = g_loaded - g_free;
    db = b_loaded - b_free;
    
    gamma = -b_free/g_free*q./(q-1);
    keyboard;
    gamma = median(gamma);
    x = (g_free*gamma - R*(b_free + gamma*g_free))./(-dg*gamma + R*(db + gamma*dg));
    end
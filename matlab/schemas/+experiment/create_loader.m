function [loader, channel] = create_loader(key)
reader = preprocess.getGalvoReader(key);
switch fetch1(experiment.SessionFluorophore & key,'fluorophore')
    case 'Twitch2B'
        error('Twitch data is not supported at the moment.');
        channel= 0;
        loader = @(islice, mask_range)(twitch_loader(key, islice, mask_range, reader));
            %);
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
    Y1(Y1 < 1) = 1;
    Y2(Y2 < 1) = 1;
    sn = get_noise_fft(Y1);
    Y = Y2./bsxfun(@plus,Y1,sn);
    
end
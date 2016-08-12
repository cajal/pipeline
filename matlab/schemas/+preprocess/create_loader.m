function [loader, channel] = create_loader(key, do_squeeze)
if nargin < 2
    do_squeeze=1;
end
reader = preprocess.getGalvoReader(key);
fluorophore = fetch1(experiment.SessionFluorophore & key,'fluorophore');
switch fluorophore
    case 'Twitch2B'
        channel= [1,2];
        loader = @(islice, varargin)(twitch_loader(key, islice, reader, varargin{:}));
    otherwise
        if strcmp(fluorophore, 'RCaMP1a')
            fprintf('\tLoading channel 2 for red RCaMP1a\n');
            channel = 2;
        else
            fprintf('\tLoading channel 1 for green fluorophoes\n');
            channel = 1;  
        end
        
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
    end
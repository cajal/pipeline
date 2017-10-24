function stk = loadCorrectedStack(key)

xy = fetch1(stack.StackMotion & key,'motion_xy');

fprintf('Loading stack...');
reader = stack.getStackReader(key);
stk = reader(:, :, :, :, :);
nslices = reader.nslices;
nframes = reader.nframes;
nchannels = reader.nchannels;

fprintf('Aligning stack...');
for islice=1:nslices
    fprintf('Slice %d of %d\n', islice, nslices)
    for iframe=1:nframes
        for ichan=1:nchannels
            stk(:,:,ichan,islice,iframe) = int16(ne7.ip.correctMotion(single(stk(:,:,ichan,islice,iframe)), xy(:,iframe,islice)));
        end
    end
end
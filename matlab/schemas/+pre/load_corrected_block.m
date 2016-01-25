function block = load_corrected_block(key, reader, frames)
%
% BLOCK = LOAD_CORRECTED_BLOCK(KEY, READER, FRAMES)
%
% Loads frames index by FRAMES from READER and performs raster and motion correction. 
% KEY is used to identify the values for the correction. 
%

  block=double(reader(:,:,1,1,frames)); % TODO change for more slices and channels
  [row, col, ~, channels, T] = size(block);
  block = reshape(block, row, col,  channels, T); % TODO: fix that once slice/channel order is clear

  % If scan is bidirectional, run raster correction
  if fetch1(pre.ScanInfo & key, 'bidirectional')
    [rasterPhase, fillFraction] = fetch1(pre.AlignRaster*pre.ScanInfo & key, 'raster_phase', 'fill_fraction');
    block = ne7.ip.correctRaster(block, rasterPhase, fillFraction);
  end
        
  % get xy motion correction and raster correction data from
  % pre.AlignMotion
  xymotion = fetch1(pre.AlignMotion & key, 'motion_xy');

  % crop xymotion to number of frames in block
  xy = xymotion(:,frames);
  xy = reshape(xy, 2, 1, size(block, 4)); % add a slice/channel dimension
                                          % run motion correction
  block  = ne7.ip.correctMotion(block, xy);
end


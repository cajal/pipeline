%{
preprocess.PrepareGalvo (imported) # basic information about resonant microscope scans, raster correction
-> preprocess.Prepare
---
nframes_requested           : int                           # number of valumes (from header)
nframes                     : int                           # frames recorded
px_width                    : smallint                      # pixels per line
px_height                   : smallint                      # lines per frame
um_width                    : float                         # width in microns
um_height                   : float                         # height in microns
bidirectional               : tinyint                       # 1=bidirectional scanning
fps                         : float                         # (Hz) frames per second
zoom                        : decimal(4,1)                  # zoom factor
dwell_time                  : float                         # (us) microseconds per pixel per frame
nchannels                   : tinyint                       # number of recorded channels
nslices                     : tinyint                       # number of slices
slice_pitch                 : float                         # (um) distance between slices
fill_fraction               : float                         # raster scan fill fraction (see scanimage)
preview_frame               : longblob                      # raw average frame from channel 1 from an early fragment of the movie
raster_phase                : float                         # shift of odd vs even raster lines
%}


classdef PrepareGalvo < dj.Relvar
    
    methods
        
        function fixRaster = get_fix_raster_fun(self)
            % returns a function that corrects the raster
            [rasterPhase, fillFraction] = self.fetch1('raster_phase', 'fill_fraction');
            if rasterPhase == 0
                fixRaster = @(img) double(img);
            else
                fixRaster = @(img) ne7.ip.correctRaster(double(img), rasterPhase, fillFraction);
            end
        end

        
        function makeTuples(self, key, reader, movie)
            
            key.nframes_requested = reader.requested_frames;
            key.nframes = reader.nframes;
            sz = size(reader);
            key.px_height = sz(1);
            key.px_width  = sz(2);
                       
            %%%% compute field of view
            fastScanMultiplier = reader.header.hRoiManager_scanAngleMultiplierFast;
            slowScanMultiplier = reader.header.hRoiManager_scanAngleMultiplierSlow;
            zoom = reader.zoom;
            fov = experiment.FOV * pro(experiment.Session*experiment.Scan & key, 'rig', 'lens', 'session_date') & 'session_date>=fov_ts';
            mags = fov.fetchn('mag');
            [~, i] = min(abs(log(mags/zoom)));
            mag = mags(i); % closest measured magnification
            [key.um_width, key.um_height] = fetch1(fov & struct('mag', mag), 'width', 'height');
            key.um_width = key.um_width * mag/zoom * fastScanMultiplier;
            key.um_height = key.um_height * mag/zoom * slowScanMultiplier;
            
            key.slice_pitch = reader.slice_pitch;
            key.fps = reader.fps;
            key.bidirectional = reader.bidirectional;
            
            key.zoom = zoom;
            key.dwell_time = reader.dwell_time; 
            key.nchannels = reader.nchannels;
            key.nslices = reader.nslices;
            key.fill_fraction = reader.fill_fraction;
            
            % average initial frames from channel 1
            skipFrames = max(0, min(2000, key.nframes-5000));
            accumFrames = min(3000, key.nframes-skipFrames);
            movie = movie(:,:,1,ceil(end/2),skipFrames+(1:accumFrames));
            meanFrameValue =  squeeze(mean(mean(movie,1),2));
            if accumFrames < 500 || median(meanFrameValue) < 10 || ...
                    quantile(meanFrameValue,0.5) < 0.5*quantile(meanFrameValue,0.95)
                error 'Too many dark frames: recording did not record properly. Aborting preprocessing....'
            end
            key.preview_frame = single(mean(movie,5));
            clear movie
            
            % raster correction
            if key.bidirectional 
                taper = 10;  % the larger the number the thinner the taper
                sz = size(key.preview_frame);
                mask = atan(taper*hanning(sz(2)))'/atan(taper);
                im = bsxfun(@times, mask, key.preview_frame-mean(key.preview_frame(:)));
                key.raster_phase = ne7.ip.computeRasterCorrection(im, key.fill_fraction);
            else
                key.raster_phase = 0;
            end
            self.insert(key)
        end
        
    end
    
end

%{
pre.ScanInfo (imported) # header information
-> rf.Scan
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
%}

classdef ScanInfo < dj.Relvar & dj.AutoPopulate
    
    properties
        popRel = rf.Scan
    end
    
    
    methods(Access=protected)
        
        function makeTuples(self, key)
            reader = pre.getReader(key);            
            key.nframes_requested = reader.requested_frames;
            key.nframes = reader.nframes;
            sz = size(reader);
            key.px_height = sz(2);
            key.px_width  = sz(1);
                       
            %%%% compute field of view
            zoom = reader.zoom;
            fov = rf.FOV * pro(rf.Session*rf.Scan & key, 'setup', 'lens', 'session_date') & 'session_date>=fov_date';
            mags = fov.fetchn('mag');
            [~, i] = min(abs(log(mags/zoom)));
            mag = mags(i); % closest measured magnification
            [key.um_width, key.um_height] = fetch1(fov & struct('mag', mag), 'width', 'height');
            key.um_width = key.um_width * zoom/mag;
            key.um_height = key.um_height * zoom/mag;
            
            %%%%
            key.slice_pitch = reader.slice_pitch;
            key.fps = reader.fps;
            key.bidirectional = reader.bidirectional;
            
            key.zoom = zoom;
            key.dwell_time = reader.dwell_time; 
            key.nchannels = reader.nchannels;
            key.nslices = reader.nslices;
            key.fill_fraction = reader.fill_fraction;
            
            self.insert(key)
        end
    end
end

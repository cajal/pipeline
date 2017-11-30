%{
stack.StackInfo (imported) # header information
-> experiment.Stack
---
nchannels                   : tinyint                       # number of recorded channels
nslices                     : int                           # number of slices 
frames_per_slice            : int                           # number of frames per slice
px_width                    : smallint                      # pixels per line
px_height                   : smallint                      # lines per frame
zoom                        : decimal(4,1)                  # zoom factor
um_width                    : float                         # width in microns
um_height                   : float                         # height in microns
slice_pitch                 : float                         # (um) distance between slices (hStackManager_stackZStepSize)
%}

classdef StackInfo < dj.Relvar & dj.AutoPopulate

	properties
		popRel = experiment.Stack
	end

	methods(Access=protected)

		function makeTuples(self, key)
            reader = stack.getStackReader(key);
            assert(~reader.bidirectional,'Structural stacks must be unidirectional scanning')
            
            key.frames_per_slice = reader.nframes;
            sz = size(reader);
            key.px_height = sz(2);
            key.px_width  = sz(1);
                       
            %%%% compute field of view
            zoom = reader.zoom;
            fov = experiment.FOV * pro(experiment.Session*experiment.Stack & key, 'rig', 'lens', 'session_date') & 'session_date>=fov_ts';
            mags = fov.fetchn('mag');
            [~, i] = min(abs(log(mags/zoom)));
            mag = mags(i); % closest measured magnification
            [key.um_width, key.um_height] = fetch1(fov & struct('mag', mag), 'width', 'height');
            key.um_width = key.um_width * zoom/mag;
            key.um_height = key.um_height * zoom/mag;
            
            key.slice_pitch = reader.slice_pitch;
            
            key.zoom = zoom;
            key.nchannels = reader.nchannels;
            key.nslices = reader.nslices;
            self.insert(key)
		end
	end

end
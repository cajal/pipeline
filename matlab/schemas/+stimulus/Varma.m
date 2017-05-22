%{
# the Varma stimulus 
-> stimulus.Condition
----
fps                     :decimal(6,3)  #  monitor frame rate
noise_seed              :smallint      #  RNG seed
pre_blank_period        :decimal(5,3)  #  (seconds)
duration                :decimal(5,3)  #  (seconds)
pattern_width           :smallint      #  pixel size of the resulting pattern
pattern_aspect          :float         #  the aspect ratio of the pattern
pattern_upscale         :tinyint       #  integer upscale factor of the pattern
gaborpatchsize          :smallint      #  size of the gabor filter = gaborpatchsize*pattern_width
gabor_wlscale           :float         #  wavelength scale: lambda = Ny/gabor_wlscale
gabor_envscale          :float         #  stddev of the Gabor's Gaussian envelope = Ny/cond.gabor_envscale
gabor_ell               :float         #  specifies the ellipticity of support of the Gabor
gaussfilt_scale         :float         #  size of filter = Ny*gaussfilt_scale
gaussfilt_istd          :float         #  inverse standard deviation
gaussfiltext_scale      :float         #  size of filter = Ny*gaussfiltext_scale
gaussfiltext_istd       :float         #  inverse standard deviation
filt_noise_bw           :float         #  BW (in Hz) for filter for noise input
filt_ori_bw             :float         #  BW (in Hz) for filter for orientation field
filt_cont_bw            :float         #  BW (in Hz) for filter for external contrast field
filt_gammshape          :float         #  shape parameter of gamma distribution to generate inputs for external contrast field
filt_gammscale          :float         #  scale parameter of gamma distribution
movie                   :longblob      # actual movie to be used
%}

classdef Varma < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '1'    % update upon modifying the algorithm
    end
    
    
    methods(Static)
        
        function test()
            cond.fps = 60;
            cond.pre_blank_period   = 5.0;
            cond.noise_seed         = 10;
            cond.pattern_upscale    = 6;
            cond.pattern_width      = 32;
            cond.duration           = 30;
            cond.pattern_aspect     = 1.7;
            cond.gaborpatchsize     = 0.28;
            cond.gabor_wlscale      = 4;
            cond.gabor_envscale     = 6;
            cond.gabor_ell          = 1;
            cond.gaussfilt_scale    = 0.5;
            cond.gaussfilt_istd     = 0.5; % originally 2
            cond.gaussfiltext_scale = 1;
            cond.gaussfiltext_istd  = 1; % originally 2.4
            cond.filt_noise_bw       = 0.5;
            cond.filt_ori_bw         = 0.5;
            cond.filt_cont_bw        = 0.5;
            cond.filt_gammshape     = 0.35;
            cond.filt_gammscale     = 2;
            
            
            
            tic
            cond = stimulus.Varma.make(cond);
            toc
            
            v = VideoWriter('/Users/rajdbz/Reservoir/Code/StimulusDesign/TestFiles/Varma', 'MPEG-4');
            v.FrameRate = cond.fps;
            v.Quality = 100;
            open(v)
            writeVideo(v, permute(cond.movie, [1 2 4 3]));
            close(v)
        end
        
        
        function cond = make(cond)
            
            rng(cond.noise_seed)
            
            
            % fill out condition structucture -- all fields are used for computing the condition id
            
            % intitial buffer time in seconds
            init_buffer_period = 3; 
            
            % video size
            Nframes = round((cond.duration + init_buffer_period)*cond.fps);
            Nx = cond.pattern_width;
            Ny = round(Nx/cond.pattern_aspect);
            
            %  Gabor filter size
            gy = round(cond.pattern_width*cond.gaborpatchsize);
            gx = round(cond.pattern_width*cond.gaborpatchsize);
            
            % Generating the meshgrid for the gabor filter
            ymin = -(gy-1)/2; ymax = (gy-1)/2;
            xmin = -(gx-1)/2; xmax = (gx-1)/2;
            [y,x] = meshgrid(linspace(ymin,ymax,gy),linspace(xmin,xmax,gx));
            
            % Gabor parameters
            lambda  = Ny/cond.gabor_wlscale;    % wavelength
            kappa   = 2*pi/lambda;              % wavenumber
            phi     = 0;                        % phase offset
            sigm    = Ny/cond.gabor_envscale;   % standard deviation of the Gaussian envelope
            gamma   = cond.gabor_ell;           % spatial aspect ratio of the Gabor
            
            % Gaussian filter for smoothing
            fx          = round(Ny*cond.gaussfilt_scale);              % size of filter in pixels
            GaussFilt   = gausswin(fx,cond.gaussfilt_istd);
            GaussFilt   = GaussFilt*GaussFilt';
            GaussFilt   = GaussFilt/sum(GaussFilt(:));
            
            % filter for smoothing used to generate external contrast field
            fx          = round(Ny*cond.gaussfiltext_scale);              % size of filter in pixels
            GaussFiltExt   = gausswin(fx,cond.gaussfiltext_istd);
            GaussFiltExt   = GaussFiltExt*GaussFiltExt';
            GaussFiltExt   = GaussFiltExt/sum(GaussFiltExt(:));
            
            % Initialize the orientation and contrast fields
            OMap    = zeros(Ny,Nx,Nframes);
            CMap    = zeros(Ny,Nx,Nframes);
            CMapExt = zeros(Ny,Nx,Nframes);
            
            
            % Filter parameters
            % obtain the 2nd order filter parameter for the given filter BW
            wr_w = 2*pi*cond.filt_noise_bw/cond.fps;
            a_w  = 2 - cos(wr_w) - sqrt((2 - cos(wr_w))^2 - 1);
            
            wr_o = 2*pi*cond.filt_ori_bw/cond.fps;
            a_o  = 2 - cos(wr_o) - sqrt((2 - cos(wr_o))^2 - 1);
            
            wr_c = 2*pi*cond.filt_cont_bw/cond.fps;
            a_c  = 2 - cos(wr_c) - sqrt((2 - cos(wr_c))^2 - 1);
            
            a_w = 1 - a_w;
            a_o = 1 - a_o;
            a_c = 1 - a_c;
            
            % Initialize all the filter inputs
            Wold1 = randn(Ny + gy - 1, Nx + gx -1);  % white noise to be filtered
            Wold2 = randn(Ny + gy - 1, Nx + gx -1);  % white noise to be filtered
            
            OMapRawOld1 = randn(Ny,Nx) + 1i*randn(Ny,Nx);
            OMapRawOld2 = randn(Ny,Nx) + 1i*randn(Ny,Nx);
            
            k = cond.filt_gammshape/(gy*gx*2/a_c); % shape parameter for the gamma distribution
            % k is proportional to 1/(no. of pixels being filtered in space and time)
            % 2/a_c term is because that is the rough time scale for temporal averaging with the 2nd order IIR filter
            CMapExtOld1 = gamrnd(k,cond.filt_gammscale,[Ny,Nx]);
            CMapExtOld2 = gamrnd(k,cond.filt_gammscale,[Ny,Nx]);
            
            % generate filter for upscaling
            f_up = cond.pattern_upscale;
            kernel_sigma = f_up;
            sz = f_up*[Ny,Nx];
            [fy,fx] = ndgrid(...
                (-floor(sz(1)/2):floor(sz(1)/2-0.5))*2*pi/sz(1), ...
                (-floor(sz(2)/2):floor(sz(2)/2-0.5))*2*pi/sz(2));
            
            fmask = exp(-(fy.^2 + fx.^2)*kernel_sigma.^2/2);
            fmask = ifftshift(fmask);
            
            % Output
            Y       = zeros(Ny,Nx,Nframes);
            YsVid   = zeros(Ny*f_up,Nx*f_up,Nframes);
            
            
            for tt = 1:Nframes
                
                % Temporal filtering of complex Gaussian noise
                OMapRawNew      = 2*(1-a_o)*OMapRawOld1 - (1-a_o)^2*OMapRawOld2 + a_o*(randn(Ny,Nx) + 1i*randn(Ny,Nx));
                % Spatial filtering to generate orientation field
                OMapFilt        = convn(OMapRawNew,GaussFilt,'same');
                OMap(:,:,tt)    = angle(OMapFilt);
                CMap(:,:,tt)    = abs(OMapFilt);
                
                % Temporal filtering of gamma distributed noise
                CMapExtNew      = 2*(1-a_c)*CMapExtOld1 - (1-a_c)^2*CMapExtOld2 + a_c*gamrnd(k,cond.filt_gammscale,[Ny,Nx]);
                % Spatial filtering to generate external contrast field
                CMapExtFilt     = convn(CMapExtNew,GaussFiltExt,'same');
                CMapExt(:,:,tt) = CMapExtFilt;
                
                % Temporal filtering of white noise to generate input to
                % the Gabor filters
                Wnew = 2*(1-a_w)*Wold1 - (1-a_w)^2*Wold2 + a_w*randn(Ny + gy - 1, Nx + gx -1);
                
                % Gabor filtering of temporally filtered noise input W
                for ii = 1:Ny
                    for jj = 1:Nx
                        % centered indices for the input
                        lx = (gy-1)/2;
                        ly = (gx-1)/2;
                        ic = ii + lx; jc = jj + ly;
                        Wreq = Wnew(ic-lx:ic+ly,jc-lx:jc+ly);
                        
                        theta = OMap(ii,jj,tt);                     % orientation
                        cont  = CMap(ii,jj,tt)*CMapExt(ii,jj,tt);   % contrast
                        
                        yp = y*cos(theta) + x*sin(theta);
                        xp = -y*sin(theta) + x*cos(theta);
                        % Gabor defined at output pixel location (ii,jj) at time tt
                        F = cont*exp(-(yp.^2 + gamma^2*xp.^2)/(2*sigm^2)).*cos(kappa*yp + phi);
                        Y(ii,jj,tt) = Wreq(:)'*F(:);
                    end
                end
                
                % Filter states updated
                OMapRawOld2 = OMapRawOld1;
                OMapRawOld1 = OMapRawNew;
                CMapExtOld2 = CMapExtOld1;
                CMapExtOld1 = CMapExtNew;
                Wold2       = Wold1;
                Wold1       = Wnew;
                YsVid(:,:,tt) = upscale(Y(:,:,tt), fmask, f_up);
            end
            
            Tstart  = 1 + cond.fps*init_buffer_period;
            YsVid   = YsVid(:,:,Tstart:end);    
            
            if cond.gaussfilt_scale == 1.5
                K = 0.015;
            elseif cond.gaussfilt_scale == 0.5
                K = 0.03;
            else
                K       = 0.025;
            end
               
            cond.movie = uint8(round(255*(YsVid/2/K + 0.5)));
        end
    end

    
    methods
        function showTrial(self, cond)
            % verify that pattern parameters match display settings
            assert(~isempty(self.fps), 'Cannot obtain the refresh rate') 
            assert(abs(self.fps - cond.fps)/cond.fps < 0.05, 'incorrect monitor frame rate')
            assert((self.rect(3)/self.rect(4) - cond.pattern_aspect)/cond.pattern_aspect < 0.05, 'incorrect pattern aspect')
            
            % blank the screen if there is a blanking period
            if cond.pre_blank_period>0
                self.flip(struct('checkDroppedFrames', false, 'logFlips', false))
                WaitSecs(cond.pre_blank_period);
            end
            
            % play movie
            for i=1:size(cond.movie,3)
                tex = Screen('MakeTexture', self.win, cond.movie(:,:,i));
                Screen('DrawTexture',self.win, tex, [], self.rect)
                self.flip(struct('checkDroppedFrames', i>1))
                Screen('close',tex)
            end
        end
    end
    
end


function out = upscale(X, fmask, f_up)
X = upsample(X', f_up, round(f_up/2))*f_up;
X = upsample(X', f_up, round(f_up/2))*f_up;
out = ifft2(fmask.*fft2(X));
end
%{
# conditions for the matisse stimulus
-> stimulus.Condition
----
noise_seed             :smallint      #  controls the base noise pattern
base_noise             :longblob      #  base noise image before filtration
pre_blank_period       :decimal(5,3)  #  (seconds)
duration               :decimal(5,3)  #  (seconds)
pattern_width          :smallint      #  pixel size of the resulting pattern
pattern_aspect         :float         #  the aspect ratio of the pattern
ori                    :decimal(4,1)  #  degrees. 0=horizontal, then clockwise
outer_ori_delta        :decimal(4,1)  #  degrees. Differerence of outer ori from inner.
ori_coherence          :decimal(4,1)  #  1=unoriented noise. pi/ori_coherence = bandwidth of orientations.
aperture_x             :decimal(4,3)  #  x position of the aperture in units of pattern widths: 0=center, 0.5=right edge
aperture_y             :decimal(4,3)  #  y position of the aperture in units of pattern widths: 0=center, 0.5/pattern_aspect = bottom edge
aperture_r             :decimal(4,3)  #  aperture radius expressed in units pattern widths
aperture_transition    :decimal(3,3)  #  aperture transition width
annulus_alpha          :decimal(3,2)  #  aperture annulus alpha
inner_contrast         :decimal(4,3)  #  pattern contrast in inner region
outer_contrast         :decimal(4,3)  #  pattern contrast in outer region
image                  :longblob      #  actual image to present
second_photodiode=0         : tinyint                       # 1/-1=paint a photodiode white/black patch in the upper right corner
second_photodiode_time=0.0  : decimal(4,1)                  # time delay of the second photodiode relative to the stimulus onset
%}

classdef Matisse < dj.Manual & stimulus.core.Visual
    
    properties(Constant)
        version = '2'
    end
    
    methods(Static)
        
        function migrate
            % incremental migrate from old +vis
            special = vis.Matisse;  % old special condition table
            trial = vis.Trial;        % old trial table
            newSpecial = stimulus.Matisse;
            
            control = stimulus.getControl;
            control.clearAll
            scans = experiment.Scan & (preprocess.Sync*trial*special & 'trial_idx between first_trial and last_trial');
            remain = scans - stimulus.Trial * newSpecial;
            
            for scanKey = remain.fetch'
                disp(scanKey)
                params = fetch(special & (trial * preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*');
                hashes = control.makeConditions(newSpecial, rmfield(params, {'animal_id', 'psy_id', 'cond_idx', 'pattern_upscale'}));
                trials =  fetch(trial & special & (preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*', 'last_flip_count->last_flip');
                hashes = hashes(arrayfun(@(t) find([params.cond_idx]==t.cond_idx, 1, 'first'), trials));
                trials = rmfield(trials, {'psy_id', 'cond_idx'});
                [trials.condition_hash] = deal(hashes{:});
                insert(stimulus.Trial, dj.struct.join(trials, scanKey))
            end
        end
        
        
        function test()
            cond.noise_seed = 100;
            cond.pattern_width = 80;
            cond.pattern_aspect = 1.7;
            cond.ori = 30;
            cond.outer_ori_delta = 30;
            cond.ori_coherence = 2.5;
            cond.aperture_x = 0.2;
            cond.aperture_y = 0.1;
            cond.aperture_r = 0.2;
            cond.aperture_transition = 0.05;
            cond.annulus_alpha = 0.0;
            cond.outer_contrast = 1;
            cond.inner_contrast = 1;
            tic
            cond = stimulus.Matisse.make(cond);
            toc
            imshow(cond.image)
            imwrite(cond.image, '~/Desktop/im2.png')
        end
        
        
        function cond = make(cond)
            % fill out condition structucture -- all fields are used for computing the condition id
            assert(isscalar(cond), 'one condition at a time please')
            r = RandStream.create('mt19937ar','NormalTransform', ...
                'Ziggurat', 'Seed', cond.noise_seed);  % just trying to get matlab to stick with a specific RNG algorithm
            nx = cond.pattern_width;
            ny = round(cond.pattern_width/cond.pattern_aspect);
            cond.base_noise = int8(r.randn(ny, nx)/3*128);
            % precomputed fields for the condition -- not used for computinng condition id
            assert(isscalar(cond), 'one condition at a time please')
            img = double(cond.base_noise)/127*1.5;
            outer = upscale(img, cond.ori + cond.outer_ori_delta, cond.ori_coherence);
            inner = upscale(img, cond.ori, cond.ori_coherence);
            img = aperture(inner*cond.inner_contrast, outer*cond.outer_contrast, ...
                cond.aperture_x, cond.aperture_y, cond.aperture_r, cond.aperture_transition, cond.annulus_alpha);
            cond.image = uint8((img+0.5)*255);
        end
        
    end
    
    
    methods
        function showTrial(self, cond)
  
            % verify that pattern parameters match display settings
            assert((self.rect(3)/self.rect(4) - cond.pattern_aspect)/cond.pattern_aspect < 0.05, 'incorrect pattern aspect')
            
            % blank the screen if there is a blanking period
            opts.clearScreen = true;
            opts.checkDroppedFrames = false;
            if cond.pre_blank_period>0
                opts.logFlips = false;
                self.flip(opts)
                WaitSecs(cond.pre_blank_period);
            end
            
            % display the texture
            opts.logFlips = true;
            tex = Screen('MakeTexture', self.win, cond.image);
            Screen('DrawTexture', self.win, tex, [], self.rect)
            self.flip(opts)
            Screen('close', tex);
            WaitSecs(cond.duration);
        end
    end
    
end


function img = upscale(img, ori, coherence)
% Performs fast resizing of the image by the given integer factor with
% gaussian interpolation.

ori_mix = coherence > 1;  % how much of orientation to mix in
factor = 3;  % upscale factor

% upscale without interpolation
kernel_sigma = factor;
img = upsample(img', factor, round(factor/2))*factor;
img = upsample(img', factor, round(factor/2))*factor;

% interpolate using gaussian kernel with DC gain = 1
sz = size(img);
[fy,fx] = ndgrid(...
    (-floor(sz(1)/2):floor(sz(1)/2-0.5))*2*pi/sz(1), ...
    (-floor(sz(2)/2):floor(sz(2)/2-0.5))*2*pi/sz(2));
fmask = exp(-(fy.^2 + fx.^2)*kernel_sigma.^2/2);

% apply orientation selectivity
theta = mod(atan2(fx,fy) + ori*pi/180 + pi/2, pi) - pi/2;
fmask = ifftshift(fmask.*(1-ori_mix + sqrt(coherence)*ori_mix*hann(theta*coherence)));
img = real(ifft2(fmask.*fft2(img)));

end


function y = hann(q)
% circuar hanning mask with symmetric opposite lobes
y = (0.5 + 0.5*cos(q)).*(abs(q)<pi);
end


function img = aperture(inner, outer, x, y, radius, transition, annulus_alpha)
% add aperture and annulus
sz = size(inner);
aspect = sz(1)/sz(2);
[y, x] = ndgrid(linspace(-aspect/2,aspect/2,sz(1))-y, linspace(-.5, .5, sz(2))-x);
r = sqrt(y.*y + x.*x);
mask = 1./(1 + exp(-(r-radius)/(transition/4)));
img = inner.*(1-mask) + outer.*mask;
img = img.*(1 - annulus_alpha*(abs(r-radius)<transition/2));
end

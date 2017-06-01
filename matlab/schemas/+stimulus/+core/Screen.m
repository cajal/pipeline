% viz.core.Screen manages the visual stimulus display, incuding the coded
% photodiode signal in the corner of the screen for synchronization.

% -- Dimitri Yatsenko, 2012, 2016


classdef Screen < handle

    properties(Constant)
        flipSize  = [0.05 0.06];   %  the relative size of the photodiode texture
    end
    
    properties
        frameStep = 1;
    end
    
    properties(SetAccess=private)
        rect              % (pixels) window rectangle
        win               % window pointer
        fps               % frames per second
        frameInterval     % (seconds)
        flipCount         % the index of the last flip
        prevFlipTime      % (seconds)
        contrastEnabled   %  when false, disables contrast and brightness settings and uses default monitor settings
    end

    properties(Access=private)
        isOpened = false
        contrast     % currently set contrast
        luminance    % currently set luminance
        binaryGray      % if true, use black and white images (e.g. square gratings)
        gammaData       % gamma table loaded from file
        savedSettings   % saved settings to restore upon closing
        flipTimes       % the flip times of the recent flips
        flipTex         % photodiode textures
        flipRect        % photodiode rectangle
    end
    
    methods
        
        function open(self)
            if ~self.isOpened
                disp 'Configuring display...'
                AssertOpenGL
                sca
                % pix screen with the largest screen number
                screen = max(Screen('Screens'));
                [self.win, self.rect] = Screen('OpenWindow',screen,127,[],[],[],[],[], ...
                    mor(kPsychNeedFastBackingStore,kPsychNeed16BPCFloat));
                AssertGLSL
                fprintf 'Screen Rectangle:'
                disp(self.rect)
                self.fps = Screen(screen, 'FrameRate', []);
                self.frameInterval = Screen('GetFlipInterval', self.win);
                Priority(MaxPriority(self.win));

                % Set luminance and contrast
                if self.contrastEnabled
                    disp 'Loading gamma'
                    self.savedSettings.gammaTable = Screen('ReadNormalizedGammaTable',self.win);
                    self.gammaData = load('~/stimulation/gammatable.mat');
                    self.setContrast(self.gammaData.luminance(end)/10, 0.5)  % while waiting, darken the screen to 1/10 of its max luminance
                end

                % create photodiode flip textures
                self.flipRect = round(self.rect(3:4).*self.flipSize);
                x = 1:self.flipRect(1);
                self.flipTex(1) = Screen('MakeTexture', self.win, x*0);
                self.flipTex(2) = Screen('MakeTexture', self.win, mod(x,2)*255);
                self.flipTex(3) = Screen('MakeTexture', self.win, x*0+255);
                self.isOpened = true;
            end
        end
        
        
        function enableContrast(self, yes)
            self.contrastEnabled = yes;
        end
        
        
        function setContrast(self, luminance, contrast, binaryGray)
            % luminance = cd/m^2
            % contrast  = Michelson contrast between 0 and 1
            % if binaryGray=true - stepwise contrast to make sine gratings appear as square gratings
            if ~self.contrastEnabled
                return
            end
            binaryGray = nargin>=4 && binaryGray;
            if isempty(self.luminance) || isepmty(self.contrast) || ...
                    contrast~=self.contrast || luminance~=self.luminance || self.binaryGray~=binaryGray

                gammaTable = self.gammaData.gammaVals(:,1);
                lumTab = self.gammaData.luminance;
                minLum = lumTab(1);
                maxLum = lumTab(end);
                minLumNew = (1 - contrast) * luminance;
                maxLumNew = 2 * luminance - minLumNew;
                x0 = 255 * (minLumNew - minLum) / (maxLum - minLum);
                x255 = 255 * (maxLumNew - minLum) / (maxLum - minLum);

                if ~binaryGray
                    ramp = linspace(x0, x255, 254);
                else
                    ramp = x0+(x255-x0)*(1:254 > 127);
                    ramp(127)=(x255+x0)/2;  % middle level to be used as background
                end
                assert(x0 >= 0 && x255 <= 255 ,'SlimStim:invalidContrast', ...
                    'Contrast/luminance combination is out of range of current monitor settings: (%.1f, %.1f) cd/m^2', ...
                    lumTab(1),lumTab(end))
                gammaTable = [0; interp1(0:255, gammaTable, ramp, 'pchip')'; 1];
                Screen('LoadNormalizedGammaTable', self.win, gammaTable * ones(1, 3));
            end
        end


        function close(self)
            if self.contrastEnabled
                disp 'restoring gamma'
                Screen('LoadNormalizedGammaTable', self.win, self.savedSettings.gammaTable);
            end
            Screen('Close',self.win);
            ShowCursor;
            self.savedSettings = [];
            sca
            self.isOpened = false;
        end

        
        function setFlipCount(self, flipCount)
            self.flipCount = flipCount;
        end
        
        function flipTimes = clearFlipTimes(self)
            flipTimes = self.flipTimes;
            self.flipTimes = [];
        end
        
        
        function flip(self, options)
            assert(all(ismember(fieldnames(options), {'logFlips', 'clearScreen', 'checkDroppedFrames'})), 'invalid option')
            logFlips   = true;
            clearScreen = true;
            checkDroppedFrames = true;
            assign(options)
                        
            if logFlips
                self.flipCount = self.flipCount + 1;
            end
            
            % draw coded photodiode flip texture
            if ~isempty(self.flipCount)
                Screen('DrawTexture', self.win, ...
                    self.flipTex(flipCode(self.flipCount)), [],...
                    [0 0 self.flipRect(1) self.flipRect(2)]);
            end
            % update screen
            when = self.prevFlipTime+self.frameStep * self.frameInterval;
            flipTime = Screen('Flip', self.win, when - 0.5*self.frameInterval, double(~clearScreen));
            if ~isempty(when)
                droppedFrames = round((flipTime - when)/self.frameInterval);
                if checkDroppedFrames
                    % indicated dropped frames by a '$" or $(n) for n dropped frames
                    if droppedFrames>3
                        fprintf('$(%d)', droppedFrames)
                    else
                        fprintf(repmat('$', 1, droppedFrames));
                    end
                end
            end
            self.prevFlipTime = flipTime;
            if logFlips
                self.flipTimes(end+1) = flipTime;
            end
        end
    end
end



function a = flipCode(n)
% Encodes the flip number n in the flip amplitude a.
% Every 32 sequential flips encode 32 21-bit flip numbers.
% Thus each n is a 21-bit flip number:
% FFFFFFFFFFFFFFFFCCCCP
% P = parity, only P=1 encode bits
% C = the position within F
% F = the current block of 32 flips
%
% a= 1 (black) or 2 (F bit=1) or 3 (white, F bit=0)

a = 1+bitand(n,1)*(2-bitget(n,bitand(floor(n/2),15)+6));
end


function assign(s)
% assigns fields from the give structure as variables in caller's workspace
for f = fieldnames(s)'
    assignin('caller', f{1}, s.(f{1}))
end
end
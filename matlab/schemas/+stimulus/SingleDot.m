%{
# single dot to map receptive field
-> stimulus.Condition
-----
bg_level                    : smallint                      # (0-255) the index of the background luminance, 0 is black
dot_level                   : smallint                      # (0-255) the index of the dot luminance, 0 is black
dot_x                       : float                         # (fraction of the x length, 0 for center, from -0.5 to 0.5) position of dot on x axis
dot_y                       : float                         # (fraction of the x length, 0 for center) position of dot on y axis
dot_xsize                   : float                         # (fraction of the x length) width of dots
dot_ysize                   : float                         # (fraction of the x length) height of dots
dot_shape                   : enum('rect','oval')           # shape of the dot
dot_time                    : float                         # (secs) time of each dot persists
%}

classdef SingleDot < dj.Manual & stimulus.core.Visual
    properties(Constant)
        version = '1'
    end
    
    methods(Static)
        function migrate
            % incremental migrate from old +vis
            special = vis.SingleDot;  % old special condition table
            trial = vis.Trial;        % old trial table
            newSpecial = stimulus.SingleDot;
            
            control = stimulus.getControl;
            control.clearAll
            scans = experiment.Scan & (preprocess.Sync*trial*special & 'trial_idx between first_trial and last_trial');
            remain = scans - stimulus.Trial * newSpecial;
            
            for scanKey = remain.fetch'
                disp(scanKey)
                params = fetch(special & (trial * preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*');
                hashes = control.makeConditions(newSpecial, rmfield(params, {'animal_id', 'psy_id', 'cond_idx'}));
                trials =  fetch(trial & special & (preprocess.Sync & scanKey & 'trial_idx between first_trial and last_trial'), '*', 'last_flip_count->last_flip');
                hashes = hashes(arrayfun(@(t) find([params.cond_idx]==t.cond_idx, 1, 'first'), trials));
                trials = rmfield(trials, {'psy_id', 'cond_idx'});
                [trials.condition_hash] = deal(hashes{:});
                insert(stimulus.Trial, dj.struct.join(trials, scanKey))
            end
        end
    end
    
    
    methods
        function showTrial(self, cond)
            Screen('FillRect', self.win, cond.bg_level, self.rect)
            width = self.rect(3);
            height = self.rect(4);
            tan = height/width;
            x_pos = cond.dot_x + 0.5;
            y_pos = cond.dot_y+0.5*tan;
            rect = [x_pos-cond.dot_xsize/2, y_pos-cond.dot_ysize/2, x_pos+cond.dot_xsize/2, y_pos+cond.dot_ysize/2]*width;
            command = struct('rect', 'FillRect', 'oval', 'FillOval');
            Screen(command.(cond.dot_shape), self.win, cond.dot_level, rect)
            self.flip(struct('clearScreen', false, 'checkDroppedFrames', false))
            WaitSecs(cond.dot_time);
            self.flip(struct('checkDroppedFrames', false))
        end
    end
end
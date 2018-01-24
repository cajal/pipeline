%{
# Bad movie clips
-> stimulus.MovieClip
---
judge                : enum('human','machine')  # was it a man or a machine?
notes                : varchar(256)             # a little explanation for the exclusion
%}

classdef ClipIgnore < dj.Manual
    methods
        function inspectClip(obj, keys, fps)
            if nargin<3
                fps = 30;
            end
            
            keys = fetch(stimulus.MovieClip - obj - movies.ClipInspected & keys);
            
            % loop through clips and replace if necessary
            for ikey = 1:length(keys)
                
                % init
                key = keys(ikey);
                fprintf('Inspecting clip: %s \n',cell2mat(reshape(...
                    [fieldnames(key) repmat({':'},2,1) cellfun(@num2str,struct2cell(key),'uni',0) repmat({'  '},2,1) ]',[],1)'))
                
                % play
                fh = play(stimulus.MovieClip & key,fps);
                
                % ask
                answer = questdlg('Too bad?','Too bad?', 'yes','no','cancel','no');
                if strcmp('yes', answer)
                    key.judge = 'human';
                    key.notes = 'Failed manual inspection';
                    insert(obj, key)
                elseif strcmp('no', answer)
                    key.judge = 'human';
                    key.notes = sprintf('Watched clip at %d fps',fps);
                    insert(movies.ClipInspected, key)
                elseif strcmp('cancel', answer)
                    fh.close
                    return
                end
                
                % close figure
                fh.close
            end
        end
    end
end
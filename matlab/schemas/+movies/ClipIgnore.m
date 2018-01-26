%{
# Bad movie clips
-> stimulus.MovieClip
---
judge                : enum('human','machine')  # was it a man or a machine?
notes                : varchar(256)             # a little explanation for the exclusion
%}

classdef ClipIgnore < dj.Manual
    methods
        function inspectClip(obj, restrict_key, fps)
            if nargin<2; restrict_key = []; end
            if nargin<3; fps = 30; end
            keys = [];
            
            % fetch unviewed keys
            updateKeys
            
            % loop through clips and replace if necessary
            while ~isempty(keys)
                
                % init
                key = keys(1);
                
                % insert as a job
                tuple = key;
                tuple.judge = 'human';
                tuple.notes = sprintf('Watched clip at %d fps',fps);
                if ~exists(movies.ClipInspected & tuple)
                    insert(movies.ClipInspected, tuple)
                else
                    updateKeys
                    continue
                end
                fprintf('Inspecting clip: %s \n',cell2mat(reshape(...
                    [fieldnames(key) repmat({':'},2,1) cellfun(@num2str,struct2cell(key),'uni',0) repmat({'  '},2,1) ]',[],1)'))
                
                % play
                fh = play(stimulus.MovieClip & key,fps);
                
                % ask
                answer = questdlg('Too bad?','Too bad?', 'yes','no','cancel','no');
                if strcmp('yes', answer)
                    key.judge = 'human';
                    key.notes = 'Failed manual inspection';
                    if ~exists(obj & key)
                        insert(obj, key)
                    end
                    if exists(movies.ClipInspected & key)
                        delQuick(movies.ClipInspected & key);
                    end
                elseif strcmp('no', answer)
                    key.judge = 'human';
                    key.notes = sprintf('Watched clip at %d fps',fps);
                    if ~exists(movies.ClipInspected & key)
                        insert(movies.ClipInspected, key)
                    end
                elseif strcmp('cancel', answer)
                    fh.close
                    if exists(movies.ClipInspected & key)
                        delQuick(movies.ClipInspected & key);
                    end
                    return
                end
                
                % close figure
                fh.close
                updateKeys
            end
            
            function updateKeys
                % update unviewed keys
                keys = fetch((stimulus.MovieClip - obj - movies.ClipInspected) & restrict_key);
            end
        end
    end
end
classdef Control < handle
    % Object to control the stimulus. It also contains the TCP/IP server to
    % accept requests from clients.
    
    properties(Constant)
        control = stimulus.core.Control  % singleton for everyone's use
        screen = stimulus.core.Visual.screen
        hashLength = 20
        flipKeyAttributes = {'animal_id'}  % the attributes of the primary key for which the flips are kept unique
    end
    
    properties(SetAccess=protected)
        conditionTable = stimulus.Condition
        trialTable = stimulus.Trial
        condCache = containers.Map
        trialQueue = stimulus.core.FIFO(1e5)
    end
    
    
    methods
        
        function hashes = makeConditions(self, specialTable, params)
            % makes conditions and stores them the specialTable
            assert(isstruct(params), 'params must be a struct')
            
            cleanup = onCleanup(@() self.conditionTable.schema.conn.cancelTransaction);
            
            hashes = cell(size(params));
            dbConn = self.conditionTable.schema.conn;
            dbConn.cancelTransaction   % reset
            for i = 1:numel(params)
                param = params(i);
                condition = struct;
                condition.stimulus_type = class(specialTable);
                condition.stimulus_version = specialTable.version;
                hash = stimulus.utils.DataHash(dj.struct.join(condition, param), ...
                    struct('Format','base64', 'Method', 'md5'));
                hash = hash(1:self.hashLength);
                
                if ~self.condCache.isKey(hash)
                    condition.condition_hash = hash;
                    if exists(self.conditionTable & condition)
                        fprintf *
                        param = fetch(specialTable & condition,'*');
                    else
                        fprintf .
                        param = specialTable.make(param);
                        specialTable.create    % to avoid implicit commits during transaction
                        dbConn.startTransaction
                        try
                            self.conditionTable.insert(condition)
                            param.condition_hash = hash;
                            specialTable.insert(param)
                            dbConn.commitTransaction
                        catch err
                            dbConn.cancelTransaction
                            rethrow(err)
                        end
                    end
                    param = specialTable.prepare(param);
                    self.condCache(hash) = setfield(param, 'obj___', specialTable); %#ok<SFLD>
                end
                hashes{i} = hash;
            end
            fprintf \n
        end
        
        
        function pushTrials(self, condition_ids)
            cellfun(@(cond) self.trialQueue.push(cond), condition_ids)
        end
        
        
        function clearTrials(self)
            self.trialQueue.reset
        end
        
        function clearCachedConditions(self)
            self.condCache.remove(self.condCache.keys);
        end
        
        function clearAll(self)
            self.clearTrials()
            self.clearCachedConditions()
        end
        
        function open(self)
            self.screen.open
            self.screen.enableContrast(false)  % change this when we calibrate monitors
            if isempty(gcp('nocreate'))
                parpool('local', 2);
            end
        end
        
        function close(self)
            self.screen.close
        end
        
        function run(self, scanKey)
            % play the trials on the trialQueue
            
            % set up clean up
            cleanupObj = onCleanup(@() self.cleanupRun());
            
            % intialize audio to be used as synchronization signal
            InitializePsychSound(1);
            audioHandle = PsychPortAudio('Open', [], [], 0, 44100, 2);
            sound = 2*ones(2,1000);
            sound(:,1:350) = 0;
            PsychPortAudio('FillBuffer',audioHandle,sound);
            PsychPortAudio('Volume',audioHandle,2);
            
            % initialize trialId
            trialId = fetch1(self.trialTable & scanKey, 'max(trial_idx) -> n')+1;
            if isnan(trialId)
                trialId = 0;
            end
            
            % initialize flip count
            flip = fetch1(self.trialTable & dj.struct.pro(scanKey, self.flipKeyAttributes{:}), 'max(last_flip) -> n')+1;
            if isnan(flip)
                flip = 0;
            end
            self.screen.setFlipCount(flip)
            self.screen.frameStep = 1;
            
            if ~stimulus.core.Visual.DEBUG
                HideCursor
                Priority(MaxPriority(self.screen.win)); % Use realtime priority for better temporal precision:
            end
            
            while ~self.trialQueue.isempty
                % emit sync audio signal used for synchronization
                PsychPortAudio('Start', audioHandle, 1, 0);
                
                %%%% SHOW TRIAL %%%%
                condition = self.condCache(self.trialQueue.pop);
                condition.obj___.showTrial(condition)
                
                % save trial
                trialRecord = scanKey;
                trialRecord.trial_idx = trialId;
                trialRecord.condition_hash = condition.condition_hash;
                trialRecord.flip_times = self.screen.clearFlipTimes();
                trialRecord.last_flip = self.screen.flipCount;
                self.trialTable.insertParallel(trialRecord)
                trialId = trialId + 1;
            end
            
        end
        
    end
    
    
    methods(Access=private)
        function cleanupRun(self)
            % used only for cleanup in run
            self.screen.flip(struct('logFlips', false, 'checkDroppedFrames', false))
            PsychPortAudio('Close');
            Priority(0);
            ShowCursor;
            disp 'cleaned up'
        end
        
    end
    
end
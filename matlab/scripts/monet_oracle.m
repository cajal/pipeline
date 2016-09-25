% animal 10469 scans 1 and 2 are 90 minute Monet recordings with 30 unique and 60 repeated trials
key = struct(...`
    'animal_id', 10469, ...`
    'session', 1, ...`
    'scan_idx', 2, ...`
    'extract_method', 2, ...`
    'spike_method', 5);

% fetch keys for repeated Monet trials
trialKey = fetch(vis.Trial * vis.Condition * vis.Monet * preprocess.Sync & key &...
    'trial_idx > first_trial and trial_idx < last_trial and rng_seed=1');

% fetch keys for all other trials
trialKey = fetchn(vis.Trial * vis.Condition * vis.Monet * preprocess.Sync & key &...
    'trial_idx > first_trial and trial_idx < last_trial and rng_seed>1',...
    'rng_seed','ORDER BY last_flip_count');

% fetch number of slices, time, and traces
nslices = fetch1(preprocess.PrepareGalvo & key,'nslices');
time = fetch1(preprocess.Sync & key, 'frame_times');
[traces, slice] = fetchn(preprocess.SpikesRateTrace*preprocess.ExtractRawGalvoROI & key, 'rate_trace', 'slice');


%% create a matrix of repeats x cells x samples for the repeated trials

% get the repeated trials keys again
key = fetch(vis.Trial * vis.Condition * vis.Monet * preprocess.Sync & key &...
    'trial_idx > first_trial and trial_idx < last_trial and rng_seed=1');

clear tMat
for i=1:length(key)
    fprintf('Processing repeat %d\n',i);
    %This fetches all the attributes
    info = fetch(vis.Trial * vis.Monet & key(i),'*');
    for j=1:length(traces)
        
        % downsample to frame rate. To get precise time of each trace, do time + (slice-1)*time_between_slices
        t = time(slice(j):nslices:end);
        
        %This gets the index into traces of the start of the trial:
        trialStartIndex = find(t>info.flip_times(1),1);
        
        %...and the end of the trial:
        trialEndIndex = find(t>info.flip_times(end),1)-1;
        
        len = length(trialStartIndex:trialEndIndex);
        if isempty(trialStartIndex) || isempty(trialEndIndex)
            j
            tMat(i,j,1:len) = nan(1,trialStartIndex:trialEndIndex);
        else
            tMat(i,j,1:len) = traces{j}(trialStartIndex:trialEndIndex);
        end
    end
end
%% oracle predictor (implemented by D.Soudry)
data = permute(tMat,[3 2 1]);
[time,neurons,trials]=size(data);

Orc=mean(data,3);
Orc=repmat(reshape(Orc,[size(Orc),1]),1,1,trials);
Orc=(Orc-data/trials)*(trials/(trials-1));
%---------------R2-------------------------%

CorrOrc=zeros(neurons,1);

OrcDataDiff=Orc-data;
R2Orc=1-squeeze(mean(var(OrcDataDiff),3)./mean(var(data),3));

%---------------Corr-------------------------%

for ii=1:neurons
    OrcN=squeeze(Orc(:,ii,:));
    dataN=squeeze(data(:,ii,:));
    CorrOrc(ii)=mean(diag(corr(dataN,OrcN)));
end

figure; hist(R2Orc,trials)
xlabel('R^2')
ylabel('count')

figure; hist(CorrOrc,trials)
xlabel('Corr')
ylabel('count')

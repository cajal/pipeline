%% choose an arbitrary pupil trace
key.animal_id=8804;
key.session=1;
key.scan_idx=3;

%% fetch timestamps, fetch and filter pupil radius trace 
r = fetchn(preprocess.EyeTrackingFrame & key,'major_r');
pupilTime = fetch1(preprocess.Eye & key,'eye_time');

% interpolate over undetected frames
rInterp = interp1(pupilTime(~isnan(r)),r(~isnan(r)),pupilTime);

% low-pass filter below 1Hz
x=rInterp;
fs = 1/median(diff(pupilTime)); %Hz
lowCut = 1;     %Hz
k = hamming(2*round(fs/lowCut)+1);
k = k/sum(k);
l = size(x,1);
n = length(k);
n = floor(n/2);
x = [x(n+2-(1:n),:); x; x(end-1-(1:n),:)]; % this pads the beginning and end with mirrored values
x = fftfilt(k,x);  % apply filter
rFilt = x(2*n+1:end,:);  % take valid values only

%% detect start and end time of dilating and constricting "pupil periods" between inflection points
increasing = diff(rFilt)>0;
decreasing = diff(rFilt)<=0;
inflection = find([0;(increasing(1:end-1) & decreasing(2:end)) | (decreasing(1:end-1) & increasing(2:end));0]);
ppStart = inflection(1:end-1);
ppEnd = inflection(2:end);

%% collect pupil period stats
for i=1:length(ppStart)
    ind = ppStart(i):ppEnd(i);  % index into pupil trace
    tuple(i).animal_id = key.animal_id;
    tuple(i).session = key.session;
    tuple(i).scan_idx = key.scan_idx;
    tuple(i).pp_ind = i; % unique index identifying each pupil period
    tuple(i).pp_mean_r = nanmean(rFilt(ind)); % mean radius
    tuple(i).pp_delta_r = rFilt(ind(end))-rFilt(ind(1)); % change in radius
    tuple(i).pp_on = pupilTime(ind(1)); % start time on behavior clock
    tuple(i).pp_dur = pupilTime(ind(end))-pupilTime(ind(1)); % duration (s)
    
    % check to see if this pupil period meets criteria for use:
    % absolute rate of change greater than 1 pixel/sec
    % total duration more than one seconds
    tuple(i).use = abs(tuple(i).pp_delta_r/tuple(i).pp_dur)>1 & tuple(i).pp_dur>1;
end

%% plot dilating and constricting periods on the pupil trace
figure
plot(pupilTime,rFilt,'k');
hold on
for i=1:length(tuple)
    if tuple(i).use
        col = {'b','r'};
        col = col{(tuple(i).pp_delta_r > 0) + 1};
        
        plot(pupilTime(ppStart(i):ppEnd(i)),rFilt(ppStart(i):ppEnd(i)),'color',col);
    end
end

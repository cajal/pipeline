% In this script, we'll fetch the pupil radius and position, and plot it
% along with a calcium trace, all on the behavior clock. The relative times
% of the eye,treadmill, and trace are precise, but the clock itself starts
% at some arbitrary offset.

% Frame times from preprocess.Sync on the visual stimulus clock can be
% used to synchronize behavioral variables with the visual stimulation.
% We'll also plot the trial start times for this scan to demonstrate this.

%% Choose an arbitrary scan
key.animal_id=8804;
key.session=1;
key.scan_idx=3;

%% Eye
% fetch pupil radius trace
r = fetchn(preprocess.EyeTrackingFrame & key,'major_r');
% ...then fetch pupil position traces
center = fetchn(preprocess.EyeTrackingFrame & key,'center');

% or fetch both together
[r,center] = fetchn(preprocess.EyeTrackingFrame & key,'major_r','center');

% undetected frames are nans in the radius trace
detectedFrames = ~isnan(r);

% convert positions from cells to array
xy = nan(length(r),2);
xy(detectedFrames,:) = cell2mat(center(~isnan(r))')';

% get pupil tracking times on the behavior clock
et = fetch1(preprocess.Eye & key,'eye_time');

% plot xy position and radius
plot(et,r)
hold on
plot(et,xy)

%% calcium traces
% choose an arbitrary calcium trace
traceKey = key;
traceKey.extract_method=2;
traceKey.trace_id=256;

% ...and fetch the trace
tr = fetch1(preprocess.ComputeTracesTrace & traceKey,'trace');

% join the trace and segmentation tables and fetch all the attributes to get more info about this trace and the mask used to generate it
tr_info = fetch(preprocess.ComputeTracesTrace * preprocess.ExtractRawGalvoROI & traceKey,'*')

% ...or just fetch the trace and slice number for the single trace from the joined tables using fetch1
[tr,slice] = fetch1(preprocess.ComputeTracesTrace * preprocess.ExtractRawGalvoROI & traceKey,'trace','slice');

% Alternatively, you can fetch all the traces for this scan using fetchn
%traces = fetchn(preprocess.ComputeTracesTrace & key,'trace');

% Fetch the imaging frame times on the behavior clock and the number of slices per scan
[ft, nslices] = fetch1(preprocess.BehaviorSync * preprocess.PrepareGalvo & key,'frame_times','nslices');

% In a single scan with 3 slices, imaging frames are collected from slice 1, 2, 3, 1, 2, 3...
% So there are nslices * length(tr) frame times
assert(nslices*length(tr)==length(ft),'You should never see this message unless the scan was aborted')

% Get the frame times for this slice
ftSlice = ft(slice:nslices:end);

% Add the trace to the pupil plot with some scaling
plot(ftSlice,tr/min(tr)*20-60)

%% vis stim

% fetch the frame times on the visual stimulus clock
vt = fetch1(preprocess.Sync & key,'frame_times');
vtSlice = vt(slice:nslices:end);

% get the trials and for this scan and their flip times
trials = fetch(vis.Trial * preprocess.Sync & key & 'trial_idx > first_trial and trial_idx < last_trial','flip_times');

for i=1:length(trials)
    % Get the imaging frame where the vis stim trial started
    startInd = find(vtSlice > trials(i).flip_times(1),1);
    
    % Use that frame to index into the times on the behavior clock
    plot(ftSlice(startInd),150,'rx')
end
%% legend
legend({'Pupil Radius (pxls)', 'Pupil X (pxls)','Pupil Y (pxls)','dF/F (scaled)', 'Vis Trial Start'})
xlabel('time on behavior clock (s)')


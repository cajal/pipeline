function [new_traces, new_times, traceKeys] = getAdjustedSpikes(key) 
% function [new_traces, new_times,traceKeys] = getAdjustedSpikes(key) 
%
% Adjusts traces for time difference between slices in a scan

[traces, slice, traceKey] = fetchn( ...
    preprocess.SpikesRateTrace * preprocess.Slice & preprocess.ExtractRawGalvoROI ...
    & key, 'rate_trace', 'slice');
nslices = fetch1(preprocess.PrepareGalvo & key, 'nslices');
uslices = unique(slice);
ca_times = fetch1(preprocess.Sync &  (experiment.Scan & key), 'frame_times');
traces = [traces{:}];
new_times = ca_times(1:nslices:end);
new_traces = nan(length(new_times),size(traces,2));
traceKeys = traceKey;
for isl = 1:length(uslices)
    islice = uslices(isl);
    slice_ca_times = ca_times(islice:nslices:end);
    X = traces(:,islice==slice);
    traceKeys(islice==slice) = traceKey(islice==slice);
    xm = min([length(slice_ca_times) length(X)]);
    X = @(t) interp1(slice_ca_times(1:xm), X(1:xm,:), t, 'linear', 'extrap');  % traces indexed by time
    
    new_traces(:,islice==slice) = X(new_times);
end
function [NewTraces, NewTimes] = getAdjustedSpikes(key) 
% function [NewTraces, NewTimes] = getSpikes(key) 
%
% Adjusts traces for time difference between slices in a scan

[Traces, slice] = fetchn( ...
    preprocess.SpikesRateTrace * preprocess.ExtractRawGalvoROI ...
    & key, 'rate_trace', 'slice' );
nslices = fetch1(preprocess.PrepareGalvo & key, 'nslices');
uslices = unique(slice);
CaTimes = fetch1(preprocess.Sync &  (experiment.Scan & key), 'frame_times');
Traces = [Traces{:}];
NewTimes = CaTimes(1:nslices:end);
NewTraces = nan(length(NewTimes),size(Traces,2));
for isl = 1:length(uslices)
    islice = uslices(isl);
    caTimes = CaTimes(islice:nslices:end);
    X = Traces(:,islice==slice);
    xm = min([length(caTimes) length(X)]);
    X = @(t) interp1(caTimes(1:xm), X(1:xm,:), t, 'linear', 'extrap');  % traces indexed by time
    
    NewTraces(:,islice==slice) = X(NewTimes);
end
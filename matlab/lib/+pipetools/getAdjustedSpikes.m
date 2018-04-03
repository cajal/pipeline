function [new_traces, new_times, traceKeys] = getAdjustedSpikes(key) 

% Adjusts traces for time difference between slices in a scan

% determine pipe
pipe = fetch1(fuse.Activity & key, 'pipe');

switch pipe
    case 'meso'
        rel = meso.ActivityTrace * meso.ScanSetUnitInfo;
        scaninfo = meso.ScanInfo;
        [nfields, nrois] = fetch1(scaninfo & key, 'nfields','nrois');
        ndepth = nfields/nrois;
    case 'reso'
        rel = reso.ActivityTrace * reso.ScanSetUnitInfo;
        scaninfo = reso.ScanInfo;
        ndepth = fetch1(scaninfo & key, 'nfields');
end
[traces, slice, delays, traceKeys] = fetchn( ...
    rel & key, 'trace', 'field','ms_delay');
delays = delays ./ 1000;  % convert to seconds
ca_times = fetch1(stimulus.Sync &  (experiment.Scan & key), 'frame_times');
traces = [traces{:}];
new_times = ca_times(1:ndepth:end);
tm = min([length(new_times) size(traces,1)]);
new_traces = nan(tm,size(traces,2));
trace_times = ca_times(1:ndepth:end) + delays;

for itrace = 1:size(traces,2)
    trace = traces(:,itrace);
    time = trace_times(itrace,:);
    X = @(t) interp1(time(1:tm), trace(1:tm), t, 'linear', 'extrap');
    new_traces(:, itrace) = X(new_times(1:tm));
end

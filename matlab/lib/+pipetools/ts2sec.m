function ts = ts2sec(ts)
% convert 10 MHz counts from the patching program (ts) to seconds 
% return: timestamps converted to seconds for each sample in the ts

% remove wraparound and convert to seconds starting at time zero
rate = 1e7;  % Hz
ts = [0; cumsum(mod(diff(double(ts(:))), 2^32))] / rate;

% Interpolate excluding areas with anomalies
timed = find(diff([-inf; ts]));
assert(max(diff(timed))==min(diff(timed)), 'unequal packet sizes')
packet_duration = median(diff(ts(timed)));
ix = find(diff(ts(timed)) < packet_duration / 1.1 | ...
    diff(ts(timed)) > packet_duration * 1.1);
if ~isempty(ix)
    warning('Found %d interruptions in packets', length(ix))
end
ts(timed(ix))=nan;
ts(timed(ix+1))=nan;
ts = interp1(timed, ts(timed), 1:length(ts),'linear','extrap');

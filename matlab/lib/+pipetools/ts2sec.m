function s = ts2sec(ts,packetLen)

% convert 10MHz timestamps from Saumil's patching program (ts) to seconds (s)
% s: timestamps converted to seconds
% packetLen: length of timestamped packets 
% start: system time (in seconds) of t=0;
% badInd: bad camera indices from 2^31:2^32 in camera timestamps prior to 4/10/13

if nargin==1
    packetLen = 0;
end

ts=double(ts);

%% check for random issue
assert(sum(ts==2^31-1) < 10,'Check for old bad timestamp issue failed')

%%  remove wraparound
wrapInd = find(diff(ts)<0);
while ~isempty(wrapInd)
    ts(wrapInd(1)+1:end)=ts(wrapInd(1)+1:end)+2^32;
    wrapInd = find(diff(ts)<0);
end
    
%% convert to seconds 
s = ts/1E7;

%% Remove offset, and if not monotonically increasing (i.e. for packeted ts), interpolate
if any(diff(s)<=0)
    % Check to make sure it's packets
    assert(packetLen == find(diff(s)>0,1,'first'));
    
    % Interpolate
    nonZero = [1 ; find(diff(s)>0)+1];
    s=interp1(nonZero,s(nonZero),1:length(s),'linear','extrap');
end

%% reshape if necessary
if size(s) ~= size(ts)
    s=s';
end
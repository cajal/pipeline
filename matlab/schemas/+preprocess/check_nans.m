function [ ok, idx ] = check_nans(Y, tolerance, beginning_or_end)
%
% Checks whether a loaded and motion corrected frame has nans. Provides an 
% index array to deal with the problem.
%
%

if nargin < 2 || isempty(tolerance)
    tolerance = 5;
end

if nargin < 3
    beginning_or_end = 0;
end

nans = squeeze(any(any(isnan(Y), 1),2));
if all(nans)
    error('All frames were NaN');
end
    
idx = logical(0*nans);                            
ok = 1;
if nans(1) && beginning_or_end == 1
    warning('Found NaN at the beginning of the block');
    idx(1:find(diff(nans),1,'first')) = 1;
    ok = -1;
elseif nans(end) && beginning_or_end == -1
    warning('Found NaN at the end of the block');
    idx(find(diff(nans),1, 'last'):end) = 1;
    ok = -1;
end


s = sum(nans(~idx)); 
if s > tolerance
    ok = 0;
    idx = nans;
elseif s > 0
    warning(['\tFound NaN frames but less than tolerance of ', tolerance]);
end


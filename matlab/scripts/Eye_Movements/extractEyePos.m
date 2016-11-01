% extract eye position traces
function [pos_x pos_y] = extractEyePos(animal_id, session, scan_idx)

s = struct('animal_id', animal_id, 'session', session, 'scan_idx', scan_idx) ;
rv = preprocess.EyeTrackingFrame & s ;
tuples = fetchn(rv, 'center') ;
pos_x = zeros(length(tuples),1) ;
pos_y = zeros(length(tuples),1) ;
for ii=1:length(tuples)
    p = tuples(ii) ;
    if (~isempty(p{1}))
       pos_x(ii) = p{1}(1) ;
       pos_y(ii) = p{1}(2) ;
    else
       pos_x(ii) = nan ;
       pos_y(ii) = nan ;
    end ;
end ;
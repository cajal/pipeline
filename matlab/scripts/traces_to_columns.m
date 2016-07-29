% This script reformats traces into a uniform format: single columns

function traces_to_columns
update_traces(preprocess.ExtractRawTrace, 'raw_trace')
update_traces(preprocess.ComputeTracesTrace, 'trace')
update_traces(preprocess.SpikesRateTrace, 'rate_trace')
end


function update_traces(table, attribute)
% convert all values of attribute in table into single column arrays.
keys = fetch(table)';
n = length(keys);
progress = -inf;
converted = 0;
for i = 1:n
    key = keys(i);
    if progress + 5 <= i/n*100
        progress = round(i/n*100);
        fprintf('Converting %s: %3d%%  converted %d traces\n', ...
            attribute, progress, converted)
    end
    trace = fetch1(table & key, attribute);
    if ~isa(trace, 'single') || ~ismatrix(trace) || size(trace,2)~=1
        update(table & key, attribute, single(trace(:)))
        converted = converted + 1;
    end
end
end
function conditions = factorize(params)
% conditions = factorize(params) makes a structure array with all
% permutations of field values in the structure params.
%
% For example,
% >> params.f1 = 1
% >> params.f2 = [1 2 3];
% >> params.f3 = {'one','two'}
% >> conditions = factorize(params)
%
%  results in
%
% conditions(1) =
%     f1: 1
%     f2: 1
%     f3: 'one'
% conditions(2) =
%     f1: 1
%     f2: 2
%     f3: 'one'
% conditions(3) =
%     f1: 1
%     f2: 3
%     f3: 'one'
% conditions(4) =
%     f1: 1
%     f2: 1
%     f3: 'two'
% conditions(5) =
%     f1: 1
%     f2: 2
%     f3: 'two'
% conditions(6) =
%     f1: 1
%     f2: 3
%     f3: 'two'


fields = fieldnames(params);

% turn all numeric arrays into cell arrays
for iField = 1:length(fields)
    n = fields{iField};
    v = params.(n);
    if ~iscell(v)
        if ischar(v)
            params.(n) = {v};
        else
            params.(n) = num2cell(v);
        end
    end
end

% cartesian product of field values
dims = structfun(@(x) size(x,2), params)';
[subs{1:length(dims)}] = ind2sub(dims, 1:prod(dims));
conditions = repmat(cell2struct(repmat({[]}, size(fields)), fields), dims);
for iField = 1:length(fields)
    field = fields{iField};
    vals = params.(field);
    for idx=1:prod(dims)
        conditions(idx).(field) = vals{subs{iField}(idx)};
    end
end
conditions = conditions(:);
end
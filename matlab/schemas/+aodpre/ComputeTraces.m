%{
aodpre.ComputeTraces (computed) # traces used for spike extraction
-> aodpre.Set
-> aodpre.PreprocessMethod
-----
%}

classdef ComputeTraces < dj.Relvar & dj.AutoPopulate

	properties
		popRel  = aodpre.Set*aodpre.PreprocessMethod
	end

	methods(Access=protected)

		function makeTuples(self, key)
			self.insert(key)
            switch fetch1(aodpre.PreprocessMethod & key, 'preprocess_name')
                case 'raw'                    
                    %for regular imaging, use channel 1.  Handle ratiometric separately
                    insert(aodpre.Trace, rmfield(fetch(aodpre.Timeseries*aodpre.PreprocessMethod & key & 'channel=1', 'trace'), 'channel'))
                case '-1pc'
                    [X, keys] = fetchn(aodpre.Timeseries*aodpre.PreprocessMethod & key & 'channel=1', 'trace');
                    keys = rmfield(keys, 'channel');
                    X = double([X{:}]);
                    % subtract 1 principal component
                    [U,D,V] = svds(X,1);
                    X = X - U*D*V';
                    for i=1:length(keys)
                        tuple = keys(i);
                        tuple.trace = single(X(:,i));
                    end
                    
                otherwise
                    error 'unknown preprocessing method'
            end
		end
	end

end
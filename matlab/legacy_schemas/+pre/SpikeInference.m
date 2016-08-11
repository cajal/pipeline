%{
pre.SpikeInference (lookup) # spike inference method
spike_inference : tinyint                # spike inference method
---
short_name                  : char(10)                      # to be used in switch statements, for example
details                     : varchar(255)                  # 
language                    : enum('matlab','python')       # the language in which the method is implemented
%}

classdef SpikeInference < dj.Relvar
    methods
        function fill(self)
            self.inserti({
                1   'rectdiff'     'thresholded forward difference'
                2   'fastoopsi'    'nonnegative sparse deconvolution from Vogelstein(2010)'
                3   'stm'          'spike triggered mixture model from Theis et al. (2016)'
                })
        end
        
        
        function X = infer_spikes(self, X, dt)
            switch self.fetch1('short_name')
                case 'rectdiff'                    
                    X = X(1:end,:) - X([1 1:end-1],:);
                    X = bsxfun(@rdivide, X, std(X));
                    X = X.*(X>2);
                case 'fastoopsi'
                    for i=1:size(X,2)
                        X(:,i) = fast_oopsi(double(X(:,i)),struct('dt',dt),struct('lambda',.3)); % changed lamda from .01 
                    end
                case 'stm'
                    error 'This deconvolution is called from python'
                otherwise
                    error 'not implemented yet'
            end
        end
        
    end
    
end

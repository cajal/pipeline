%{
aodpre.Trace (imported) #  imported calcium traces
-> aodpre.ComputeTraces
-> aodpre.ScanPoint
-----
trace : longblob  #  traces
%}

classdef Trace < dj.Relvar
    methods
        function plot(self)
            for key = fetch(aodpre.Scan & self)'
                figure
                duration = fetch1(aodpre.Scan & key, 'signal_duration');
                X = fetchn(self & key, 'trace');
                X = [X{:}];
                t = linspace(0, duration, size(X,1));
                plot(t, bsxfun(@plus,bsxfun(@rdivide,X,mean(X))/2,1:size(X,2)))
            end
        end
        
    end
end
%{
aodpre.Trace (imported) #  imported calcium traces
-> aodpre.Scan
trace_id :  smallint   # trace number within scan
channel : tinyint   # microscope channel
-----
trace : longblob  #  tracec
%}

classdef Trace < dj.Relvar
    methods
        function makeTuples(self, key)
            traces = aodReader(fetch1(aodpre.Scan & key, 'hdf5_file'), 'Functional');
            sz = traces.reshapedSize;
            for channel = 1:sz(3)
                key.channel = channel;
                t = single(traces(:,:,channel));
                for itrace = 1:sz(2)
                    key.trace_id = itrace;
                    key.trace = t(:,itrace);
                    self.insert(key);
                end
            end
        end
    end
end